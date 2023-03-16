import torch.utils.data as data
import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils

class Dataset(data.Dataset):
    def __init__(self, data_root, split, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        ann_file = kwargs.get('ann_file', os.path.join(self.data_root, 'annots.npy'))
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        if split == 'train' or cfg.vis_training_view:
            view = cfg.training_view
        else:
            view = cfg.test_view if len(cfg.test_view) > 0 else [0]

        self.b = cfg.begin_ith_frame
        self.s = cfg.frame_interval
        self.e = self.b + cfg.num_train_frame * self.s
        self.f = self.s if split == 'train' else cfg.test['frame_sampler_interval'] * self.s
        
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][self.b:self.e][::self.f]
        ]).ravel()

        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][self.b:self.e][::self.f]
        ]).ravel()

        self.num_cams = len(view)
        self.nrays = cfg.N_rand
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
            self.fixed_H = 512
            self.fixed_W = 512

    def get_mask(self, img_path):
        msk_path = cfg.img_path_to_schp_path(img_path)
        msk_path = os.path.join(self.data_root, msk_path)
        msk = imageio.imread(msk_path)[..., :3].astype(np.int32)
        msk = (msk * [1, 10, 100]).sum(axis=-1)
        palette = if_nerf_dutils.get_schp_palette(20)
        face_msk = (msk == palette[2]) | (msk == palette[10]) | (
            msk == palette[13])
        msk = (msk != 0).astype(np.uint8)
        msk[face_msk == 1] = 100

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
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 1 if cfg.white_bkgd else 0

        return img, msk

    def __getitem__(self, index):
        img_path = self.ims[index]
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.
        RT = np.concatenate([R, T], axis=1)
        RT = np.concatenate([RT, [[0, 0, 0, 1]]], axis=0)
        transform = cfg.get('transform', np.eye(4))
        RT = np.dot(RT, np.linalg.inv(transform))
        R, T = RT[:3, :3], RT[:3, 3:]
        
        img, msk = self.get_image(img_path, K, D)
        H, W = img.shape[:2]
        K[:2] = K[:2] * cfg.ratio

        frame_index = cfg.img_path_to_frame_ind(img_path)
        latent_index = (frame_index - self.b) // self.s

        ret = {
            'H': H,
            'W': W,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind,
        }

        i = cfg.img_path_to_smpl_ind(img_path)
        wpts = self.prepare_input(i)
        wpts = np.dot(wpts, transform[:3, :3].T) + transform[:3, 3]
        wbounds = if_nerf_dutils.get_bounds(wpts)
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(img, msk, K, R, T, wbounds, self.nrays, self.split)

        # nerf
        ret.update({
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box,
            'wbounds': wbounds
        })
                
        if cfg.get('use_encoder', False):
            fix_latent_index = latent_index
            fixed_idxs = list(range(fix_latent_index * self.fixed_num_cams,
                                    (fix_latent_index+1) * self.fixed_num_cams))
            fixedcamimages = np.zeros((3 * self.fixed_num_cams, self.fixed_H, self.fixed_W), dtype=np.float32)
            for i, idx in enumerate(fixed_idxs):
                fix_img_path = self.fixed_ims[idx]
                cam_idx = self.fixed_cam_inds[idx]
                K = np.array(self.cams['K'][cam_idx])
                D = np.array(self.cams['D'][cam_idx])
                fixedcamimage, _ = self.get_image(fix_img_path, K, D)
                fixedcamimage = cv2.resize(fixedcamimage, (self.fixed_H, self.fixed_W), interpolation=cv2.INTER_AREA)
                fixedcamimage = fixedcamimage.transpose(2, 0, 1)
                fixedcamimages[i * 3: (i+1) * 3, :, :] = fixedcamimage

            fixedcamimages[:] = (fixedcamimages[:] * 255 - self.imagemean) / self.imagestd
            ret['fixedcamimages'] = fixedcamimages

        # mask
        if cfg.get('with_mask', True):
            occ = (msk[coord[:, 0], coord[:, 1]] != 0).astype(np.int32)
            ret.update({
                'occ': occ,
            })

            if cfg.get('filter_mask', False):
                border = 5
                kernel = np.ones((border, border), np.uint8)
                msk = (msk > 0).astype(np.float32)
                msk_erode = cv2.erode(msk.copy(), kernel)
                msk_dilate = cv2.dilate(msk.copy(), kernel)
                edge = (msk_dilate - msk_erode) == 1
                edge = edge[coord[:, 0], coord[:, 1]]
            else:
                edge = np.zeros_like(occ)
                
            ret.update({'edge': edge})

        return ret

    def __len__(self):
        return len(self.ims)
    