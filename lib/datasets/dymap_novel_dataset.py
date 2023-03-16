import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
import tqdm
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils import render_utils

class Dataset(data.Dataset):
    def __init__(self, data_root, split, **kwargs):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split
        ann_file = kwargs.get('ann_file', os.path.join(self.data_root, 'annots.npy'))
        annots = np.load(ann_file, allow_pickle=True).item()

        ixts, exts = render_utils.load_cam(ann_file)
        self.ixts = np.array(ixts).astype(np.float32)
        self.exts = np.array(exts).astype(np.float32)
        self.distorts = np.array(annots['cams']['D']).astype(np.float32)

        assert cfg.fixed_view * cfg.fixed_time == 0

        self.render_ixt = self.ixts[0]
        render_path = cfg.get('render_path', None)
        if render_path is not None:
            exts = [exts[i] for i in render_path] 


        self.b = cfg.begin_ith_frame
        self.s = cfg.frame_interval
        self.e = self.b + cfg.num_train_frame * self.s

        if cfg.get('fixed_time', False):
            self.render_frames = [self.b] * cfg.render_views
            self.render_exts = render_utils.gen_path(exts, render_views=cfg.render_views)
        else:
            self.render_frames  = np.arange(self.b, self.e)[::self.s].tolist()
            self.render_exts = render_utils.gen_path(exts, render_views=cfg.num_train_frame)
        
        if cfg.get('fixed_view', False):
            novel_ext = self.render_exts[:1]
            self.render_exts = np.repeat(novel_ext, cfg.num_train_frame, axis=0)
            
        self.ims = np.array([
            np.array(ims_data['ims'])
            for ims_data in annots['ims'][self.b: self.e][::self.s]
        ])
                
        if cfg.get('use_encoder', False):
            self.fixedcameras = [cfg.training_view[idx] for idx in cfg.fixedcameras]
            self.imagemean = cfg.get('imagemean', 100)
            self.imagestd = cfg.get('imagestd', 25)
            
        wbounds_all = []
        for i in tqdm.trange(len(self.render_exts)):
            render_frame = self.render_frames[i]
            transform = cfg.get('transform', np.eye(4))
            wpts = self.prepare_input(render_frame)
            wpts = np.dot(wpts, transform[:3, :3].T) + transform[:3, 3]

            scale = cfg.get('render_scale', 1.)
            shift = cfg.get('render_shift', np.array([0., 0., 0.]))

            wbounds = if_nerf_dutils.get_bounds(wpts, scale=scale, shift=shift)
            wbounds_all.append(wbounds)
        self.wbounds_all = np.stack(wbounds_all, axis=0)

        self.h, self.w = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
                
    def get_mask(self, img_path):
        msk_path = cfg.img_path_to_schp_path(img_path)
        msk_path = os.path.join(self.data_root, msk_path)
        msk = imageio.imread(msk_path)[..., :3].astype(np.int32)
        msk = (msk * [1, 10, 100]).sum(axis=-1)
        msk = (msk != 0).astype(np.uint8)
        return msk

    def prepare_input(self, i):
        vertices_path = os.path.join(self.data_root, cfg.vertices, '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices, '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)
        return wxyz

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
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 1 if cfg.white_bkgd else 0
        
        return img, msk
    
    def get_rays_within_bounds(self, H, W, K, R, T, bounds):
        ray_o, ray_d = if_nerf_dutils.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = if_nerf_dutils.get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        
        return ray_o, ray_d, near, far, mask_at_box

    def __getitem__(self, index):
        ext = self.render_exts[index]
        frame_index = self.render_frames[index]
        latent_index = (frame_index - self.b) // self.s
        wbounds = self.wbounds_all[frame_index]
        transform = cfg.get('transform', np.eye(4))
        ext = np.dot(ext, np.linalg.inv(transform))
        R, T = ext[:3, :3], ext[:3, 3:]

        ret = {
            'cam_ind': index,
            'frame_index': frame_index,
            'latent_index': latent_index,
            'wbounds': wbounds,
        }
        
        if cfg.get('fast_render', False):
            ret.update({
                'H': self.h,
                'W': self.w,
                'K': self.render_ixt,
                'R': R,
                'T': T,
                'wbounds': wbounds,
            })
            return ret
            
        else:            
            ray_o, ray_d, near, far, mask_at_box = self.get_rays_within_bounds(
                self.h, self.w, self.render_ixt, R, T, wbounds)
            
            ret.update({
                'ray_o': ray_o,
                'ray_d': ray_d,
                'near': near,
                'far': far,
                'mask_at_box': mask_at_box,
                'H': self.h,
                'W': self.w
            })


            if cfg.get('use_encoder', False):
                ninput = len(self.fixedcameras)
                fixedcamimages = np.zeros((3 * ninput, 512, 512), dtype=np.float32)
                for i, cam_ind in enumerate(self.fixedcameras):
                    img_path = self.ims[index][cam_ind]
                    K = self.ixts[cam_ind].copy()
                    K[:2] = K[:2] / cfg.ratio
                    D = self.distorts[cam_ind]
                    fixedcamimage, _ = self.get_image(img_path, K, D)
                    fixedcamimage = cv2.resize(fixedcamimage, (512, 512), interpolation=cv2.INTER_AREA)
                    fixedcamimage = fixedcamimage.transpose(2, 0, 1)
                    fixedcamimages[i * 3: (i+1) * 3, :, :] = fixedcamimage

                fixedcamimages[:] = (fixedcamimages[:] * 255 - self.imagemean) / self.imagestd
                ret['fixedcamimages'] = fixedcamimages

        return ret

    def __len__(self):
        return len(self.render_exts)


