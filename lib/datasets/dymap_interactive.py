import torch.utils.data as data
import numpy as np
import os
import tqdm
import torch
from scipy import interpolate
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils import render_utils

class Dataset(data.Dataset):
    def __init__(self, data_root, split, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.b = cfg.begin_ith_frame
        self.s = cfg.frame_interval
        self.e = self.b + cfg.num_train_frame * self.s
        self.render_frames  = np.arange(self.b, self.e)[::self.s].tolist()
        self.kwargs = kwargs
        try:
            self.load_meta()
            self.prepare_wbounds()
        except:
            return   
        
        return

    def load_meta(self):
        ann_file = self.kwargs.get('ann_file', os.path.join(self.data_root, 'annots.npy'))
        ixts, exts = render_utils.load_cam(ann_file)
        self.ixts_np = np.array(ixts).astype(np.float32).copy()
        self.exts_np = np.array(exts).astype(np.float32)
        cam_path = cfg.get('render_path', None)
        if cam_path is not None:
            self.ixts_np = self.ixts_np[cam_path]
            self.exts_np = self.exts_np[cam_path]
        
        self.cam_points = np.linalg.inv(self.exts_np)[:, :3, 3].astype(np.float32)
        self.cam_dirs = np.linalg.inv(self.exts_np)[:, :3, :3].astype(np.float32)

        self.ixt_np = np.mean(self.ixts_np, axis=0).astype(np.float32)
       
        self.H = torch.tensor([int(cfg.H * cfg.ratio)]).cuda()
        self.W = torch.tensor([int(cfg.W * cfg.ratio)]).cuda()
        
        self.known_cams = np.arange(len(self.exts_np))

        self.ixt_cuda = torch.tensor(self.ixt_np).cuda()
        self.ixts_cuda = torch.tensor(self.ixts_np).cuda()
        self.exts_cuda = torch.tensor(self.exts_np).cuda()
    
    
    def prepare_wbounds(self):
        wbounds_all = []
        for i in tqdm.trange(len(self.render_frames)):
            frame_index = self.render_frames[i]
            transform = cfg.get('transform', np.eye(4))
            wpts = self.prepare_input(frame_index)
            wpts = np.dot(wpts, transform[:3, :3].T) + transform[:3, 3]

            scale = cfg.get('interactive_scale', 1.)
            shift = cfg.get('interactive_shift', np.array([0., 0., 0.]))
            
            wbounds = if_nerf_dutils.get_bounds(wpts, scale=scale, shift=shift)
            wbounds_all.append(wbounds)

        wbounds_all = np.stack(wbounds_all, axis=0)
        self.wbounds_all = torch.tensor(wbounds_all).cuda()

    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                '{:06d}.npy'.format(i))
            
        wxyz = np.load(vertices_path).astype(np.float32)
        return wxyz

    def __getitem__(self, query):
        frame_index, w2c = query         
        transform = cfg.get('transform', np.eye(4))
        ext = np.dot(w2c, np.linalg.inv(transform))
        ext = torch.tensor(ext).cuda()
        R, T = ext[:3, :3][None], ext[:3, 3:][None]
        K = self.ixt_cuda[None]
        
        idx = (frame_index - self.b) // self.s
        wbounds = self.wbounds_all[idx][None]
        latent_index = torch.tensor([idx]).cuda()
        
        ret = {
            'H': self.H,
            'W': self.W,
            'K': K,
            'R': R,
            'T': T,
            'wbounds': wbounds,
            'latent_index': latent_index,
        }
        
        return ret
            
        
    def get_camera_up_front_center(self, index=0):
        """Return the worldup, front vectors and center of the camera
        Typically used to load camera parameters
        Extrinsic Matrix: leftmost column to rightmost column: world_down cross front, world_down, front
        [
            worldup, front, center (all column vectors)
        ]

        Args:
            index(int): which camera to load
        Returns:
            worldup(np.ndarray), front(np.ndarray), center(np.ndarray)
        """
        # TODO: loading from torch might be slow?
        ext = self.exts_np[index]
        worldup, front, center = -ext.T[:3, 1], ext.T[:3, 2], -ext[:3, :3].T @ ext[:3, 3]
        return worldup, front, center

    def get_closest_camera(self, center):
        return np.argmin(np.linalg.norm(self.cam_points - center, axis=-1))

    def get_camera_tck(self, smoothing_term=0.0):
        """Return B-spline interpolation parameters for the camera
        TODO: Actually this should be implemented as a general interpolation function
        Reference get_camera_up_front_center for the definition of worldup, front, center
        Args:
            smoothing_term(float): degree of smoothing to apply on the camera path interpolation
        """
        # - R^t @ T = cam2world translation
        # TODO: loading from torch might be slow?
        exts = self.exts_np
        # TODO: load from cam_points to avoid repeated computation
        all_cens = -np.einsum("bij,bj->bi", exts[:, :3, :3].transpose(0, 2, 1), exts[:, :3, 3]).T
        all_fros = exts[:, 2, :3].T  # (3, 21)
        all_wups = -exts[:, 1, :3].T  # (3, 21)
        cen_tck, cen_u = interpolate.splprep(all_cens, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        fro_tck, fro_u = interpolate.splprep(all_fros, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        wup_tck, wup_u = interpolate.splprep(all_wups, s=smoothing_term, per=1)  # array of u corresponds to parameters of specific camera points
        return cen_tck, cen_u, fro_tck, fro_u, wup_tck, wup_u

    @property
    def n_cams(self):
        return len(self.known_cams)

    def __len__(self):
        return len(self.render_frames)
