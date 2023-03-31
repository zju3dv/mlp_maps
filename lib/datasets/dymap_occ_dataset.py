import numpy as np
import os
import imageio
import cv2
import torch.utils.data as data
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils import base_utils

class Dataset(data.Dataset):
    def __init__(self, data_root, split, **kwargs):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        ann_file = kwargs.get('ann_file', os.path.join(self.data_root, 'annots.npy'))
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        
        self.b = cfg.get('mesh_begin_ith_frame', cfg.begin_ith_frame)
        self.s = cfg.get('mesh_frame_interval', cfg.frame_interval)
        self.e = self.b + self.s * cfg.get('mesh_num_train_frame', cfg.num_train_frame)
        
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][self.b:self.e:self.s]
        ])
        self.num_cams = 1
        
        self.Ks = np.array(self.cams['K'])[cfg.training_view].astype(np.float32)
        self.Rs = np.array(self.cams['R'])[cfg.training_view].astype(np.float32)
        self.Ts = np.array(self.cams['T'])[cfg.training_view].astype(np.float32) / 1000.
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(np.float32)
    
        if cfg.get('use_encoder', False):
            self.fixedcameras_view = [cfg.training_view[idx] for idx in cfg.fixedcameras]
            self.fixed_num_cams = len(self.fixedcameras_view)
            self.fixed_ims = np.array([
            np.array(ims_data['ims'])[self.fixedcameras_view]
                for ims_data in annots['ims'][self.b:self.e][::self.s]
            ]).ravel()
            
            self.fixed_cam_inds = np.array([
                np.arange(len(ims_data['ims']))[self.fixedcameras_view]
                for ims_data in annots['ims'][self.b:self.e][::self.s]
            ]).ravel()
            
            self.imagemean = cfg.get('imagemean', 100)
            self.imagestd = cfg.get('imagestd', 25)
            self.scale_to_255 = cfg.get('scale_to_255', True)

    def get_mask(self, img_path):
        msk_path = cfg.img_path_to_schp_path(img_path)
        msk_path = os.path.join(self.data_root, msk_path)
        msk = imageio.imread(msk_path)[..., :3].astype(np.int32)
        msk = (msk * [1, 10, 100]).sum(axis=-1)
        msk = (msk != 0).astype(np.uint8)

        return msk    
    
    def get_mask_inside(self, i, nv):
        img_path = self.ims[i, nv]
        if cfg.get('get_mask', False):
            msk = cfg.get_mask(self.data_root, img_path)
        else:
            msk = self.get_mask(img_path)
        msk = cv2.undistort(msk, self.Ks[nv], self.Ds[nv])
        return msk

    def prepare_input(self, i):
        vertices_path = os.path.join(self.data_root, cfg.vertices, '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices, '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)
        return wxyz
    
    def prepare_inside_pts(self, pts, i):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        for nv in range(self.ims.shape[1]):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([self.Rs[nv], self.Ts[nv]], axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[nv], RT)
            msk = self.get_mask_inside(i, nv)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]
            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def get_image(self, img_path, K, D):
        ori_img_path = img_path
        img_path = os.path.join(self.data_root, img_path)
        img = imageio.imread(img_path).astype(np.float32) / 255.
        if cfg.get('get_mask', False):
            msk = cfg.get_mask(self.data_root, ori_img_path)
        else:
            msk = self.get_mask(ori_img_path)

        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 1 if cfg.white_bkgd else 0
        
        return img, msk
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index][0])        
        frame_index = cfg.img_path_to_frame_ind(img_path)
        latent_index = (frame_index - cfg.begin_ith_frame) // cfg.frame_interval
        latent_index = min(latent_index, cfg.num_train_frame - 1)

        wpts = self.prepare_input(frame_index)
        transform = cfg.get('transform', np.eye(4))
        wpts = np.dot(wpts, transform[:3, :3].T) + transform[:3, 3]
        wbounds = if_nerf_dutils.get_bounds(wpts)
        
        if cfg.get('vis_mesh', False):
            pts = if_nerf_dutils.prepare_mesh_wpts(wbounds)
        elif cfg.get('occ_grid', False):
            pts = if_nerf_dutils.prepare_occupancy_wpts(wbounds)
        else:
            raise NotImplementedError()
                
        ret = {
            'pts': pts,
        }
        
        if cfg.get('mesh_mask', False):
            i = index
            inside = self.prepare_inside_pts(pts, i)
            ret['inside'] = inside
            
        meta = {
            'wbounds': wbounds,
            'latent_index': latent_index,
            'frame_index': frame_index
        }
        ret.update(meta)
        
        if cfg.get('use_encoder', False): 
            fix_latent_index = (frame_index - self.b) // self.s
            fixed_idxs = list(range(fix_latent_index * self.fixed_num_cams, 
                                    (fix_latent_index+1) * self.fixed_num_cams))
            fixedcamimages = np.zeros((3 * self.fixed_num_cams, 512, 512), dtype=np.float32)
            for i, idx in enumerate(fixed_idxs):
                fix_img_path = self.fixed_ims[idx]
                cam_idx = self.fixed_cam_inds[idx]
                K = np.array(self.cams['K'][cam_idx])
                D = np.array(self.cams['D'][cam_idx])
                fixedcamimage, _ = self.get_image(fix_img_path, K, D)
                fixedcamimage = cv2.resize(fixedcamimage, (512, 512), interpolation=cv2.INTER_AREA)
                
                fixedcamimage = fixedcamimage.transpose(2, 0, 1)
                fixedcamimages[i * 3: (i+1) * 3, :, :] = fixedcamimage
            fixedcamimages[:] = (fixedcamimages[:] * 255 - self.imagemean) / self.imagestd
            ret['fixedcamimages'] = fixedcamimages
        return ret

    def __len__(self):
        return len(self.ims)