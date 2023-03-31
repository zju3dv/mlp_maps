import numpy as np
import torch.nn as nn
import torch
import torch_scatter
import torch.nn.functional as F
from lib.config import cfg
from lib.networks.dymap_utils import conv_kn_layers
from lib.networks.dymap_utils.kilonerf_layers import CudaMultiNetworkLinear, FourierEmbedding
from lib.csrc.hashencoder import HashEncoder
from lib.csrc.shencoder import SHEncoder
import kilonerf_cuda

kilonerf_cuda.init_stream_pool(16)
kilonerf_cuda.init_magma(cfg.local_rank)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        if cfg.get('sh_encoder', False):
            self.view_embedder = SHEncoder(input_dim=3, degree=cfg.view_res)
            view_outdim = self.view_embedder.output_dim
        else:
            self.view_embedder = FourierEmbedding(num_input_channels=3, num_frequencies=cfg.view_res)
            view_outdim = self.view_embedder.num_output_channels
                
        if cfg.mode == 'train':
            self.batch_size = cfg.train['batch_size']
        else:
            self.batch_size = cfg.test['batch_size']
            
        hash_level = cfg.hash.get('level', 16)
        hash_size = cfg.hash.get('size', 19)
        hash_dim = cfg.hash.get('dim', 2)
        self.hashtable_xyt = HashEncoder(
            input_dim=3, num_levels=hash_level,
            level_dim=hash_dim, base_resolution=16, 
            log2_hashmap_size=hash_size, desired_resolution=1024
        )
        self.hashtable_xzt = HashEncoder(
            input_dim=3, num_levels=hash_level,
            level_dim=hash_dim, base_resolution=16, 
            log2_hashmap_size=hash_size, desired_resolution=1024
        )
        self.hashtable_yzt = HashEncoder(
            input_dim=3, num_levels=hash_level,
            level_dim=hash_dim, base_resolution=16, 
            log2_hashmap_size=hash_size, desired_resolution=1024
        )
            
        self.input_channel = cfg.feature_channel
                
        self.mlp_layers = cfg.get('mlp_layers', 2)
        if self.mlp_layers == 2:
            self.net_layers = {
                'color':
                    ((self.input_channel + view_outdim, cfg.layer_size),
                    (cfg.layer_size, 3))
            }
        elif self.mlp_layers == 3:
            self.net_layers = {
                'color':
                    ((self.input_channel, cfg.layer_size),
                    (cfg.layer_size + view_outdim, cfg.layer_size),
                    (cfg.layer_size, 3))
            }
        else:
            raise NotImplementedError()

        num_kn_params = [
            [(v[0] + 1) * v[1] if cfg.mlp_use_bias and i != len(in_outs) - 1
             else  v[0] * v[1] for i, v in enumerate(in_outs)]
        for in_outs in self.net_layers.values()]
        self.num_kn_params = sum([sum(v) for v in num_kn_params])
        
        # set mlp config
        self.info = self.set_mlp_info(cfg.mlp_map_size)

        # build encoder
        if cfg.get('fast_render', False):
            self.register_buffer('encodings', torch.randn((cfg.num_train_frame, cfg.code_dim)).cuda())
        else:        
            if cfg.get('use_encoder', False):
                self.latent_vectors = conv_kn_layers.Encoder(ninputs=len(cfg.fixedcameras))
            else:
                self.latent_vectors = nn.Embedding(cfg.num_train_frame, cfg.code_dim)

        # build decoder
        self.plane_decoder = conv_kn_layers.PlaneDecoder(cfg.mlp_map_size, self.num_kn_params)
        
        # occupancy grid
        if cfg.get('fast_render', False):
            occ_res = [cfg.num_train_frame] + cfg.grid_cfg['grid_resolution']
            self.register_buffer('occupancy_grids', torch.ones(occ_res).bool().cuda())
            
            num_points = self.batch_size * cfg.points_buffer
            
            self.query_indices_buffer = torch.arange(0, num_points, 1).long().cuda()

                
    def set_mlp_info(self, mlp_map_size):
        info = {
            'mlp_map_size': mlp_map_size,
            'mlp_res': [mlp_map_size, mlp_map_size, mlp_map_size],
            'kn_plane_reses': {
                'xy': [mlp_map_size, mlp_map_size, 1],
                'xz': [mlp_map_size, 1, mlp_map_size],
                'yz': [1, mlp_map_size, mlp_map_size],
            },
        }

        info['kn_num_networks'] = {
            p: np.prod(info['kn_plane_reses'][p]) for p in cfg.kn_plane_types
        }
        info['num_networks'] = sum(info['kn_num_networks'].values())
        info['num_all_networks'] = info['num_networks'] * self.batch_size
        
        plane_mlp_sizes = torch.zeros((cfg.kn_num_planes, 2)).cuda()
        plane_mlp_strides = torch.zeros((cfg.kn_num_planes, 2)).cuda()
        plane_mlp_indices_offsets = torch.zeros((cfg.kn_num_planes)).cuda()
        plane_mlp_indices_bounds = torch.zeros((cfg.kn_num_planes, 2)).cuda()
        
        self.kn_plane_slices = []

        cur_mlp_ind = 0
        cur_plane_ind = 0
        for plane_type in cfg.kn_plane_types:
            plane_mlp_indices_offsets[cur_plane_ind] = cur_mlp_ind
            num_network = info['kn_num_networks'][plane_type]
            res = torch.tensor(info['kn_plane_reses'][plane_type])
            if cfg.get('mlp_overlap', False):
                assert cfg.mlp_map_size > 1
                voxel_size = 1. / (res - 1)
            else:
                voxel_size = 1. / res
                
            if plane_type == 'xy':
                plane_slice = [0, 1]
                plane_mlp_sizes[cur_plane_ind] = voxel_size[plane_slice]
                project_res = res[plane_slice]
                self.kn_plane_slices.extend(plane_slice)
            elif plane_type == 'xz':
                plane_slice = [0, 2]
                plane_mlp_sizes[cur_plane_ind] = voxel_size[plane_slice]
                project_res = res[plane_slice]
                self.kn_plane_slices.extend(plane_slice)
            else:
                plane_slice = [1, 2]
                plane_mlp_sizes[cur_plane_ind] = voxel_size[plane_slice]
                project_res = res[plane_slice]
                self.kn_plane_slices.extend(plane_slice)

            plane_mlp_strides[cur_plane_ind] = torch.tensor([project_res[1], 1])
            plane_mlp_indices_bounds[cur_plane_ind] = project_res - 1

            cur_mlp_ind += num_network
            cur_plane_ind += 1

        mlp_layers = {}
        for k in self.net_layers.keys():
            mlp_layers[k] = []
            for v in self.net_layers[k]:
                in_features = v[0]
                out_features = v[1]
                mlp_layers[k].append(
                    CudaMultiNetworkLinear(info['num_all_networks'], in_features, out_features, cfg.mlp_use_bias, True))

        self.feature_plane_slices = []
        for plane_type in cfg.feature_plane_types:
            if plane_type == 'xy':
                self.feature_plane_slices.extend([0, 1])
            elif plane_type == 'xz':
                self.feature_plane_slices.extend([0, 2])
            else:
                self.feature_plane_slices.extend([1, 2])
        
        meta = {
            'plane_mlp_sizes': plane_mlp_sizes,
            'plane_mlp_strides': plane_mlp_strides,
            'plane_mlp_indices_offsets': plane_mlp_indices_offsets,
            'plane_mlp_indices_bounds': plane_mlp_indices_bounds,
            'mlp_layers': mlp_layers,
        }
        
        info.update(meta)
        return info

    def multikn_forward(self, norm, feat, viewdir, kn_params):
        mlp_layers = self.info['mlp_layers']
    
        batch_size_per_network, query_indices, query_weight = self.get_batch_size_per_network(norm, self.info, cfg.mlp_overlap)
   
        feat = feat.view(-1, self.input_channel)
        viewdir = viewdir.view(-1, 3)
        norm = norm.view(-1, 3)
            
        feat = feat[query_indices]
        curr_param_ind = 0

        embed_viewdir = self.view_embedder(viewdir)
        embed_viewdir = embed_viewdir[query_indices]

        if self.mlp_layers == 2:
            net = torch.cat((feat, embed_viewdir), dim=1)
            for i in range(len(self.net_layers['color'])):
                in_out_ch = self.net_layers['color'][i]

                num_weight_params = in_out_ch[0] * in_out_ch[1]
                mlp_weight_params = kn_params[:, curr_param_ind:curr_param_ind +
                                        num_weight_params].view(-1, in_out_ch[0], in_out_ch[1])

                curr_param_ind = curr_param_ind + num_weight_params
                if cfg.mlp_use_bias and i != len(self.net_layers['color']) - 1:
                    num_bias_params = in_out_ch[1]
                    mlp_bias_params = kn_params[:, curr_param_ind:curr_param_ind +
                                        num_bias_params].view(-1, in_out_ch[1])
                    curr_param_ind = curr_param_ind + num_bias_params
                else:
                    mlp_bias_params = None
                multi_mlp = mlp_layers['color'][i]
                net = multi_mlp(net, mlp_weight_params, mlp_bias_params, batch_size_per_network)
                if i != len(self.net_layers['color']) - 1:
                    net = F.relu(net)
        
        elif self.mlp_layers == 3:
            net = feat
            for i in range(len(self.net_layers['color'])):
                if i == 1 and cfg.get('use_viewdir', True):
                    net = torch.cat([net, embed_viewdir], dim=-1)
        
                in_out_ch = self.net_layers['color'][i]
                num_weight_params = in_out_ch[0] * in_out_ch[1]
                mlp_weight_params = kn_params[:, curr_param_ind:curr_param_ind +
                                        num_weight_params].view(-1, in_out_ch[0], in_out_ch[1])

                curr_param_ind = curr_param_ind + num_weight_params
                if cfg.mlp_use_bias and i != len(self.net_layers['color']) - 1:
                    num_bias_params = in_out_ch[1]
                    mlp_bias_params = kn_params[:, curr_param_ind:curr_param_ind +
                                        num_bias_params].view(-1, in_out_ch[1])
                    curr_param_ind = curr_param_ind + num_bias_params
                else:
                    mlp_bias_params = None
                multi_mlp = mlp_layers['color'][i]
                
                net = multi_mlp(net, mlp_weight_params, mlp_bias_params, batch_size_per_network)
                      
                if i != len(self.net_layers['color']) - 1 and i != 0:
                    net = F.relu(net)    
        else:
            raise NotImplementedError()      

        color = net                
        multinet_output = color * query_weight[:, None]
        multinet_index = query_indices[:, None].expand(-1, 3) 
        
        rgb = torch_scatter.scatter_add(
            src=multinet_output,
            index=multinet_index,
            dim=0,
        )

        return rgb

    def get_batch_size_per_network(self, norm, info, mlp_overlap):
        '''
        point -> mlp
        norm: [B, N, 3]
        B: batch size
        N: number of points
        M: number of planes
        '''
        batch_size = norm.shape[0]
        num_points = norm.shape[1]
        batch_mlp_indices_offsets = torch.arange(0, batch_size, 1).int().cuda() * info['num_networks']
        
        n = num_points * batch_size         
        if cfg.get('fast_render', False) and self.query_indices_buffer.shape[0] >= n:
            query_indices = self.query_indices_buffer[:n]
        else:
            query_indices = torch.arange(0, num_points * batch_size, 1).long().cuda()

        if mlp_overlap:
            offset = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).int().cuda()
            norm = norm[:, :, self.kn_plane_slices].view(-1, num_points, cfg.kn_num_planes, 2)
                                    
            query_indices = query_indices[:, None].repeat(1, 4 * cfg.kn_num_planes).view(batch_size, -1) # [B, N * M * 4]

            mlp_indices_2d_float = norm / info['plane_mlp_sizes'] # [B, N, M, 2]
            mlp_indices_2d_int = mlp_indices_2d_float.int()
            mlp_indices_2d_int = torch.clamp(mlp_indices_2d_int,
                min=torch.zeros_like(info['plane_mlp_indices_bounds']),
                max=info['plane_mlp_indices_bounds']-1)
            mlp_offset_2d = mlp_indices_2d_float - mlp_indices_2d_int
            mlp_indices_2d = (mlp_indices_2d_int[:, :, :, None] + offset) # [B, N, M, 4, 2]
            mlp_indices = (mlp_indices_2d * info['plane_mlp_strides'][:, None]).sum(dim=-1) # [B, N, M, 4]
            mlp_indices += info['plane_mlp_indices_offsets'][:, None]

            query_weights_2d = torch.clamp(
                (1 - offset) + (2 * offset - 1.) * mlp_offset_2d[:, :, :, None],
                min=0., max=1.)
            query_weight = torch.prod(query_weights_2d, dim=-1, keepdim=False) # [B, N, M, 4]
            query_weight = query_weight.view(batch_size, -1) # [B, N * M * 4]

   
        else:
            norm = norm[:, :, self.kn_plane_slices].view(-1, num_points, cfg.kn_num_planes, 2)
            query_indices = query_indices[:, None].repeat(1, cfg.kn_num_planes).view(batch_size, -1) # [B, N * M]

            mlp_indices_2d_int = (norm / info['plane_mlp_sizes']).int() # [B, N, M, 2]
            mlp_indices_2d_int = torch.clamp(mlp_indices_2d_int,
                min=torch.zeros_like(info['plane_mlp_indices_bounds']),
                max=info['plane_mlp_indices_bounds'])
            mlp_indices = (mlp_indices_2d_int * info['plane_mlp_strides']).sum(dim=-1) # [B, N, M]
            mlp_indices += info['plane_mlp_indices_offsets']
            query_weight = None


        mlp_indices = mlp_indices.view(batch_size, -1).int() # [B, N * M]
        mlp_indices, reorder_indices = torch.sort(mlp_indices, dim=-1)
        mlp_indices += batch_mlp_indices_offsets[:, None]
        mlp_indices = mlp_indices.view(-1)

        contained_nets, batch_size_per_network_incomplete = torch.unique_consecutive(mlp_indices, return_counts=True)
        contained_nets = contained_nets.long()
        batch_size_per_network = torch.zeros(info['num_networks'] * self.batch_size).long().cuda()
        batch_size_per_network[contained_nets] = batch_size_per_network_incomplete
        batch_size_per_network = batch_size_per_network.cpu()

        query_indices = torch.gather(query_indices, -1, reorder_indices).view(-1)
        query_weight = torch.ones_like(query_indices) if query_weight is None else torch.gather(query_weight, -1, reorder_indices).view(-1)
 
        return batch_size_per_network, query_indices, query_weight

    def get_latent_vector(self, batch):
        if cfg.get('fast_render', False):
            latent_index = batch['latent_index']
            latent_vector = self.encodings[latent_index]
        else:
            if cfg.get('use_encoder', False):
                latent_vector, kldiv_loss = self.latent_vectors(batch['fixedcamimages'])
                self.kldiv_loss = kldiv_loss
            else:
                latent_index = batch['latent_index']
                latent_vector = self.latent_vectors(latent_index)
                
        return latent_vector
    
    def to_norm(self, wpts, batch):
        bounds = batch['wbounds']
        global_domain_min, global_domain_max = bounds[:, :1], bounds[:, 1:]
        global_domain_size = global_domain_max - global_domain_min
        norm = (wpts - global_domain_min) / global_domain_size # [B, N, 3]
        norm = torch.clamp(norm, min=0., max=1.)
        return norm

    def calculate_params(self, batch, latent_vector=None):
        if latent_vector is None:
            latent_vector = self.get_latent_vector(batch)
        
        params = self.plane_decoder(latent_vector)

        return params
    
    def calculate_feature(self, norm, batch, params):
        t = (batch['latent_index'][:, None, None] / cfg.num_train_frame) * 2. - 1.
        t = torch.broadcast_to(t, norm[..., 0:1].shape)
        norm_ = norm.detach() * 2. - 1.

        xyt_hash_feat = self.hashtable_xyt(torch.cat([norm_[..., [0, 1]], t], dim=-1))
        xzt_hash_feat = self.hashtable_xzt(torch.cat([norm_[..., [0, 2]], t], dim=-1))
        yzt_hash_feat = self.hashtable_yzt(torch.cat([norm_[..., [1, 2]], t], dim=-1))
        xyz_triplane_feat = conv_kn_layers.sample_grid_feature(
            norm=norm, 
            trip_kn_features=params['trip_kn_features'], 
            plane_slices=self.feature_plane_slices
        )

        xyz_feat = xyt_hash_feat + xzt_hash_feat + yzt_hash_feat + xyz_triplane_feat
        return xyz_feat
    
    def calculate_density(self, wpts, batch, params=None, normalize=True, return_feat=False):
        if params is None:
            params = self.calculate_params(batch)
        
        if normalize:
            norm = self.to_norm(wpts, batch)
        else:
            norm = wpts

        xyz_feat = self.calculate_feature(norm, batch, params)
        density_weight = conv_kn_layers.sample_grid_feature(
            norm=norm, 
            trip_kn_features=params['trip_kn_density_params'], 
            plane_slices=self.kn_plane_slices
        )
        
        sigma = torch.sum(density_weight * xyz_feat, dim=-1)
        sigma = sigma.view(-1)[..., None]
        
        if return_feat:
            return xyz_feat, sigma
        else:
            return sigma
        
    def calculate_appearance(self, wpts, viewdir, batch, params=None, normalize=True, feature=None):
        if params is None:
            params = self.calculate_params(batch)

        if normalize:
            norm = self.to_norm(wpts, batch)
        else:
            norm = wpts
            
        if feature is None:
            xyz_feat = self.calculate_feature(norm, batch, params)
        else:
            xyz_feat = feature
  
        rgb = self.multikn_forward(
            norm=norm, 
            feat=xyz_feat, 
            viewdir=viewdir, 
            kn_params=params['trip_kn_rgb_params']
        )

        return rgb
        
    def forward(self, wpts, viewdir, dists, batch):
        batch_size = batch['latent_index'].shape[0]
        wpts = wpts.view(batch_size, -1, 3)
        viewdir = viewdir.view(batch_size, -1, 3)
        
        params = self.calculate_params(batch)
        norm = self.to_norm(wpts, batch)
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * dists)
                    
        feat, sigma = self.calculate_density(
            wpts=norm, 
            batch=batch, 
            params=params, 
            normalize=False, 
            return_feat=True
        )
        
        rgb = self.calculate_appearance(
            wpts=norm, 
            viewdir=viewdir, 
            batch=batch, 
            params=params, 
            feature=feat,
            normalize=False
        )
        
        rgb = torch.sigmoid(rgb).view(-1, 3)

        alpha = raw2alpha(sigma[:, 0], dists)[:, None]
        raw = torch.cat((rgb, alpha), dim=1)
        ret = {'raw': raw}
        return ret
    
