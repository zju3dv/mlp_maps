import torch
import numpy as np
import os
from lib.config import cfg
import imageio


def pts_to_can_pts(pts, batch):
    """transform pts from the world coordinate to the smpl coordinate"""
    Th = batch['Th']
    pts = pts - Th
    R = batch['R']
    pts = torch.matmul(pts, batch['R'])
    return pts


def pts_to_coords(pts, min_xyz):
    pts = pts.clone().detach()
    # convert xyz to the voxel coordinate dhw
    dhw = pts[..., [2, 1, 0]]
    min_dhw = min_xyz[:, [2, 1, 0]]
    dhw = dhw - min_dhw[:, None]
    dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)
    return dhw


def record_mask_depth(output, batch):
    img_path = os.path.join(batch['data_root'][0], batch['img_name'][0])
    msk_path = os.path.join('data/train_mask_depth', 'mask',
                            img_path[:-4] + '.png')
    depth_path = os.path.join('data/train_mask_depth', 'depth',
                              img_path[:-4] + '.png')

    max_depth = 10
    if os.path.exists(msk_path):
        msk = imageio.imread(msk_path)
        depth = imageio.imread(depth_path)
        depth = depth / 65535 * max_depth
    else:
        os.system("mkdir -p '{}'".format(os.path.dirname(msk_path)))
        os.system("mkdir -p '{}'".format(os.path.dirname(depth_path)))
        H, W = batch['H'].item(), batch['W'].item()
        msk = np.zeros([H, W])
        depth = np.zeros([H, W])

    coord = batch['coord'][0].detach().cpu().numpy()
    surf_z = output['surf_z'][0].detach().cpu().numpy()
    surf_mask = output['surf_mask'][0].detach().cpu().numpy()

    fg_coord = coord[surf_mask]
    bkgd_coord = coord[surf_mask == 0]

    msk[fg_coord[:, 0], fg_coord[:, 1]] = 255
    msk[bkgd_coord[:, 0], bkgd_coord[:, 1]] = 0
    msk = msk.astype(np.uint8)

    depth[fg_coord[:, 0], fg_coord[:, 1]] = surf_z[surf_mask]
    depth = depth / max_depth * 65535
    depth = depth.astype(np.uint16)

    imageio.imwrite(msk_path, msk)
    imageio.imwrite(depth_path, depth)
