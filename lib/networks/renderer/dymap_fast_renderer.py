import torch
import torch.nn.functional as F
import torch
import lib.csrc.nerfacc as nerfacc
from lib.config import cfg
from lib.networks.dymap import Network as DyMap


class Renderer:
    def __init__(self, net):
        self.net:DyMap = net
        self.max_samples = cfg.get('max_samples', 256)
        self.weight_thres = cfg.get('weight_thres', 1e-3)
        
        self.ERT = cfg.get('ERT', False)
        self.ERT_max_samples = cfg.get('ERT_max_samples', 32)
        
        self.occupancy_grids = self.net.occupancy_grids.cuda()
        self.encodings = self.net.encodings.cuda()
        
        H = int(cfg.H * cfg.ratio)
        W = int(cfg.W * cfg.ratio)
        
        self.rgb_map_all = torch.zeros((H, W, 3)).float().cuda()
        self.acc_map_all = torch.zeros((H, W, 1)).float().cuda()
        self.rgb_map_buffer = None
        self.acc_map_buffer = None

    def clear(self):
        self.rgb_map_all *= 0.
        self.acc_map_all *= 0.
        
    def set_stepsize(self, wbounds):
        self.render_step_size = torch.norm(wbounds[1] - wbounds[0], 
                       dim=-1, keepdim=False).item() / self.max_samples

    def set_params(self, batch):
        if self.encodings is None:
            latent_vector = None
        else:
            idx = batch['latent_index'][0]
            latent_vector = self.encodings[idx]
        self.params = self.net.calculate_params(
            batch, 
            latent_vector=latent_vector
        )
          
    def to_norm(self, wpts, wbounds):
        norm = (wpts - wbounds[0]) / (wbounds[1] - wbounds[0])
        return norm
    
    def make_grid(self, batch):
        idx = batch['latent_index'][0]
        binary = self.occupancy_grids[idx]
        roi_aabb =  batch['wbounds'].view(-1)
        resolution = list(binary.shape)
        contaction_type = nerfacc.grid.ContractionType.AABB
        occupancy_grid = nerfacc.grid.OccupancyGrid(
            roi_aabb=roi_aabb,
            resolution=resolution,
            contraction_type=contaction_type,
        )
        
        occupancy_grid._binary = binary
        return occupancy_grid
    
    def render_core(self, packed_info, ray_indices, norm, viewdir, batch, ERT=False):
        '''
        packed_info: [n_rays, 2]
        ray_indices: [n_samples]
        norm: [n_samples, 3]
        viewdir: [n_samples, 3]
        
        return:
            rgb_map: [n_rays, 3]
            acc_map: [n_rays]
        '''
        
        n_rays = packed_info.shape[0]
        n_samples = norm.shape[0]
        
        density_points = 0
        color_points = 0

        if ERT:
            ray_start_indices = packed_info[:, 0] # [n_rays]
            ray_remain_samples = packed_info[:, 1] # [n_rays]
            ray_transmittance = torch.ones_like(ray_start_indices).float() # [n_rays]
            active_ray_mask = torch.logical_and(
                (ray_remain_samples > 0),
                (ray_transmittance > self.weight_thres)) # [n_rays]
            ray_active = torch.sum(active_ray_mask) 
            
            while ray_active > 0:
                samples_per_ray = torch.clamp_max(ray_remain_samples[active_ray_mask], max=self.ERT_max_samples) # [ray_active]
                global_start_indices = ray_start_indices[active_ray_mask]
                global_packed_info = torch.cat([global_start_indices[..., None], samples_per_ray[..., None]], dim=-1) # [ray_active, 2]
                global_active_mask = nerfacc.unpack_mask(global_packed_info, n_samples) # [n_samples]
            
                cur_norm = norm[global_active_mask] # [pts_active, 3]
                cur_viewdir = viewdir[global_active_mask] # [pts_active, 3]
                density_points += cur_norm.shape[0]
                                
                cur_feat, cur_sigma = self.net.calculate_density(
                    cur_norm[None],
                    batch,
                    params=self.params,
                    normalize=False,
                    return_feat=True,
                )
                cur_feat = cur_feat[0]
                cur_alpha = 1. - torch.exp(-F.relu(cur_sigma) * self.render_step_size) # [n_samples, 1]
                local_start_indices = torch.cumsum(samples_per_ray, dim=0).int() - samples_per_ray
                local_packed_info = torch.cat([local_start_indices[..., None], samples_per_ray[..., None]], dim=-1) # [ray_active, 2]
                cur_ray_indices = nerfacc.unpack_info(local_packed_info)
                active_ray_transmittance = ray_transmittance[active_ray_mask][..., None]
                cur_weight = nerfacc.render_weight_from_alpha_with_transmittance(
                    packed_info=local_packed_info,
                    alphas=cur_alpha,
                    transmittance=active_ray_transmittance,
                    early_stop_eps=self.weight_thres,
                ) # [n_samples]
                
                cur_mask = cur_weight > self.weight_thres
                mask_sum = torch.sum(cur_mask).item()
                
                if mask_sum == 0:
                    break
                else:
                    color_points += mask_sum

                cur_masked_weight = cur_weight[cur_mask]
                cur_masked_ray_indices = cur_ray_indices[cur_mask]
                cur_masked_norm = cur_norm[cur_mask]
                cur_masked_viewdir = cur_viewdir[cur_mask]
                cur_masked_feat = cur_feat[cur_mask]
                
                cur_rgb = self.net.calculate_appearance(
                    wpts=cur_masked_norm[None],
                    viewdir=cur_masked_viewdir[None],
                    batch=batch,
                    params=self.params,
                    feature=cur_masked_feat,
                    normalize=False,
                ) # [n_samples, 3]
                
                cur_rgb = torch.sigmoid(cur_rgb)

                cur_rgb_map = nerfacc.accumulate_along_rays(
                    weights=cur_masked_weight,
                    ray_indices=cur_masked_ray_indices,
                    values=cur_rgb,
                    n_rays=ray_active,
                )
                
                cur_acc_map = nerfacc.accumulate_along_rays(
                    weights=cur_masked_weight,
                    ray_indices=cur_masked_ray_indices,
                    values=None,
                    n_rays=ray_active,
                )
                
                # update_render_map
                self.rgb_map_buffer[active_ray_mask] += cur_rgb_map
                self.acc_map_buffer[active_ray_mask] += cur_acc_map

                # update T
                ray_transmittance[active_ray_mask] = active_ray_transmittance[..., 0]                
                ray_start_indices[active_ray_mask] += samples_per_ray
                ray_remain_samples[active_ray_mask] -= samples_per_ray
                active_ray_mask = torch.logical_and(
                    (ray_remain_samples > 0),
                    (ray_transmittance > self.weight_thres)) # [n_rays]
                ray_active = torch.sum(active_ray_mask)
            
        else:
            density_points += norm.shape[0]
            
            feat, sigma = self.net.calculate_density(
                norm[None],
                batch,
                params=self.params,
                normalize=False,
                return_feat=True,
            )
            
            alpha = 1. - torch.exp(-F.relu(sigma) * self.render_step_size)
            ray_weight = nerfacc.render_weight_from_alpha(
                packed_info=packed_info,
                alphas=alpha,
                early_stop_eps=self.weight_thres,
            )
            
            mask = ray_weight > self.weight_thres
            mask_sum = torch.sum(mask).item()
            if mask_sum > 0:
                color_points = mask_sum
                mask_weight = ray_weight[mask]
                mask_ray_indices = ray_indices[mask]
                mask_pts = norm[mask]
                mask_viewdir = viewdir[mask]
                mask_feat = None if feat is None else feat[mask[None]]
                                
                rgb = self.net.calculate_appearance(
                    wpts=mask_pts[None],
                    viewdir=mask_viewdir[None],
                    batch=batch,
                    params=self.params,
                    feature=mask_feat,
                    normalize=False,
                )
                rgb = torch.sigmoid(rgb)
                
                self.rgb_map_buffer += nerfacc.accumulate_along_rays(
                    weights=mask_weight,
                    ray_indices=mask_ray_indices,
                    values=rgb,
                    n_rays=n_rays,
                )
                
                self.acc_map_buffer += nerfacc.accumulate_along_rays(
                    weights=mask_weight,
                    ray_indices=mask_ray_indices,
                    values=None,
                    n_rays=n_rays,          
                )
                

        
        print('density_points: ', density_points)
        print('color_points', color_points)
            
            
    def render_chunk(self, ray_o, ray_d, batch):
        wbounds = batch['wbounds'][0]
        self.set_stepsize(wbounds)
        
        self.set_params(batch)

        # ray marching
        if self.occupancy_grids is not None:
            grid = self.make_grid(batch)
        else:
            grid = None

        packed_info, t_starts, t_ends = nerfacc.ray_marching(
            rays_o=ray_o,
            rays_d=ray_d,
            scene_aabb=wbounds.view(-1),
            grid=grid,
            render_step_size=self.render_step_size
        )
                
        # rendering 
        if torch.any(packed_info[:, 1] > 0):
            ray_indices = nerfacc.unpack_info(packed_info)
            t_mid = (t_starts + t_ends) / 2.0
            wpts = ray_o[ray_indices] + t_mid * ray_d[ray_indices]
            norm = self.to_norm(wpts, wbounds)         
            viewdir = ray_d[ray_indices]
        
            self.render_core(
                packed_info=packed_info,
                ray_indices=ray_indices,
                norm=norm,
                viewdir=viewdir,
                batch=batch,
                ERT=self.ERT,
            )
        
        else:
            print('density_points: ', 0)
            print('color_points', 0)
   
    def render(self, batch):
        if 'ray_o' and 'ray_d' in batch.keys():
            ray_o = batch['ray_o'][0]
            ray_d = batch['ray_d'][0]
            mask_at_box = batch['mask_at_box'][0]
            
            H, W = batch['H'][0], batch['W'][0]
            rgb_map = self.rgb_map_all.view(-1, 3)[:H * W, :][mask_at_box]
            acc_map = self.acc_map_all.view(-1, 1)[:H * W, :][mask_at_box]
        else:
            H, W, K, R, T = batch['H'][0], batch['W'][0], batch['K'][0], batch['R'][0], batch['T'][0]
            
            from lib.utils.if_nerf import if_nerf_data_utils
            ray_o, ray_d = if_nerf_data_utils.get_rays_torch(H, W, K, R, T)
            ray_o = ray_o[0]
            ray_d = ray_d[0]
            
            self.clear()
            rgb_map = self.rgb_map_all.view(-1, 3)
            acc_map = self.acc_map_all.view(-1, 1)
               
        n_pixel = ray_o.shape[0]
        chunk = cfg.render_chunk if 'render_chunk' in cfg.keys() else n_pixel

        for i in range(0, n_pixel, chunk):   
            self.rgb_map_buffer = rgb_map[i:i+chunk]
            self.acc_map_buffer = acc_map[i:i+chunk]
    
            ray_o_chunk = ray_o[i:i + chunk]
            ray_d_chunk = ray_d[i:i + chunk]

            self.render_chunk(ray_o_chunk, ray_d_chunk, batch)            
          
        if 'bg' in batch.keys():
            assert cfg.white_bkgd == False
            bg = batch['bg'][0]
            rgb_map = rgb_map + (1. - acc_map) * bg

        elif cfg.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map) * 1.
        else:
            rgb_map = rgb_map

        if 'mask_at_box' in batch.keys():
            rgb_map = rgb_map[None]
            acc_map = acc_map[None]
        else:
            rgb_map = rgb_map.view(-1, batch['H'].item(), batch['W'].item(), 3)
            acc_map = acc_map.view(-1, batch['H'].item(), batch['W'].item(), 1)
            
        ret = {
            'rgb_map': rgb_map,
            'acc_map': acc_map,
        }

        return ret 
             