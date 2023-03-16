import torch
from lib.config import cfg
from .nerf_net_utils import *
from lib.networks.dymap import Network as DyMap

class Renderer:
    def __init__(self, net):
        self.net:DyMap = net

    def batchify_rays(self, wpts, alpha_decoder, chunk=1024 * 32):
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = alpha_decoder(wpts[i:i + chunk])
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 0)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape
        pts = pts[0].view(-1, 3)
        alpha_decoder = lambda x: self.net.calculate_density(x, batch)
        alpha = self.batchify_rays(pts, alpha_decoder, 2048 * 64)
        grid = alpha[:, 0].detach().cpu().view(sh[1:-1])
        grid = grid > cfg.alpha_thres
        grid = torch.any(grid, dim=-1)
        
        encoding = self.net.get_latent_vector(batch)[0]
        
        ret = {
            'encoding': encoding,
            'occupancy_grid': grid,
        }
        
        return ret