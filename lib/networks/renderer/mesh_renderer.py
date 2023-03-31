import torch
from lib.config import cfg
from .nerf_net_utils import *
import numpy as np
import mcubes
import trimesh


class Renderer:
    def __init__(self, net):
        self.net = net

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

        if cfg.get('mesh_mask', False):
            inside = batch['inside'][0].bool()
            pts = pts[0][inside]
            inside = inside.detach().cpu().numpy()
            alpha_decoder = lambda x: self.net.calculate_density(x, batch)
            alpha = self.batchify_rays(pts, alpha_decoder, 2048 * 64)
            alpha = alpha[:, 0].detach().cpu().numpy()
            cube = np.zeros(sh[1:-1])
            cube[inside == 1] = alpha
        else:
            pts = pts[0].view(-1, 3)
            alpha_decoder = lambda x: self.net.calculate_density(x, batch)
            alpha = self.batchify_rays(pts, alpha_decoder, 2048 * 64)
            alpha = alpha.view(sh[1:-1])
            cube = alpha.detach().cpu().numpy()
    
        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret