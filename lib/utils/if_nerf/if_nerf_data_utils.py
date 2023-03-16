import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg
import torch

def get_rays_within_bounds_test(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    
    if cfg.center_pixel:
        xy1 = np.stack([i + 0.5, j + 0.5, np.ones_like(i)], axis=2)
    else:
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_rays_torch(H, W, K, R, T):
    K, R, T = K.float(), R.float(), T.float()
    # calculate the camera origin
    rays_o = -torch.matmul(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32),
                       indexing='xy')
    if cfg.center_pixel:
        xy1 = torch.stack([i + 0.5, j + 0.5, torch.ones_like(i)], dim=2)
    else:
        xy1 = torch.stack([i, j, torch.ones_like(i)], dim=2)
    xy1 = xy1.view(-1, 3).to(K)
    pixel_camera = torch.matmul(xy1, torch.linalg.inv(K).T)
    pixel_world = torch.matmul(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=2, keepdims=True)
    rays_o = torch.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box

def calculate_rayod_near_far(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)

    full_near = np.zeros(len(ray_o)).astype(np.float32)
    full_far = np.zeros(len(ray_d)).astype(np.float32)
    full_near[mask_at_box] = near
    full_far[mask_at_box] = far

    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)
    near = full_near.reshape(H, W)
    far = full_far.reshape(H, W)
    tminmax = np.stack([near, far], axis=2)
    box_msk = mask_at_box.reshape(H, W)

    return ray_o, ray_d, tminmax, box_msk


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    scale = cfg.get('cnn_upsample', 1)
    K_ = K.copy()
    K_[:2] = K_[:2] / scale
    ray_o, ray_d = get_rays(H, W, K_, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    # corners_3d = get_bound_corners(bounds)
    # corners_2d = base_utils.project(corners_3d, K, pose)
    # corners_2d = np.round(corners_2d).astype(int)
    # import matplotlib.pyplot as plt
    # plt.imshow(bound_mask); plt.show()

    if cfg.mask_bkgd:
        img[bound_mask != 1] = 1 if cfg.get('white_bkgd', False) else 0
    
    msk = msk * bound_mask
    coord_body = np.argwhere(msk != 0)
    coord_face = np.argwhere(msk == 100)
    coord_bound = np.argwhere(bound_mask == 1)

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body_ = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            if len(coord_face) > 0:
                coord_face_ = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = coord_bound[np.random.randint(0, len(coord_bound), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body_, coord_face_, coord], axis=0)
            else:
                coord = np.concatenate([coord_body_, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(\
                mask_at_box.reshape(H // scale, W // scale) == 1)

        if scale == 1:
            rgb = img.reshape(-1, 3).astype(np.float32)
            rgb = rgb[mask_at_box]
        else:
            rgb = img

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_bounds(xyz, padding=True, scale=1., shift=np.array([0., 0., 0.])):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    if padding:
        min_xyz -= cfg.get('box_padding', 0)
        max_xyz += cfg.get('box_padding', 0)

    size = (max_xyz - min_xyz) * scale
    center = (max_xyz + min_xyz) * 0.5 + shift
    min_xyz = center - 0.5 * size
    max_xyz = center + 0.5 * size
    
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds



def enlarge_box(bbox, img, times):
    bbox_mean = np.mean(bbox, axis=0)
    # bbox = np.round((bbox - bbox_mean) * times + bbox_mean).astype(np.int32)

    wh = (bbox[1] - bbox[0]) * times
    size = max(wh)
    min_x, min_y = bbox_mean - size // 2
    max_x, max_y = bbox_mean + size // 2
    bbox = np.array([[min_x, min_y], [max_x, max_y]])
    bbox = np.round(bbox).astype(np.int32)

    h, w = img.shape[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def get_schp_palette(num_cls=256):
    # Copied from SCHP
    """ Returns the color map for visualizing the segmentation mask.
    Inputs:
        =num_cls=
            Number of classes.
    Returns:
        The color map.
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    palette = np.array(palette)
    palette = palette.reshape(-1, 3)  # n_cls, 3
    palette = (palette * [1, 10, 100]).sum(axis=1)
    return palette

def prepare_mesh_wpts(wbounds):
    voxel_size = cfg.voxel_size
    x = np.arange(wbounds[0, 0], wbounds[1, 0] + voxel_size[0], voxel_size[0])
    y = np.arange(wbounds[0, 1], wbounds[1, 1] + voxel_size[1], voxel_size[1])
    z = np.arange(wbounds[0, 2], wbounds[1, 2] + voxel_size[2], voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    pts = pts.astype(np.float32)
    return pts

def prepare_occupancy_wpts(wbounds):
    grid_res = cfg.grid_resolution
    globol_domain_min, global_domain_max = wbounds[0], wbounds[1]
    global_domain_size = global_domain_max - globol_domain_min
    voxel_size = global_domain_size / grid_res
    voxel_offset_min = globol_domain_min
    voxel_offset_max = voxel_offset_min + voxel_size
    voxel_samples = []
    for dim in range(3):
        voxel_samples.append(np.linspace(voxel_offset_min[dim], voxel_offset_max[dim], cfg.subgrid_resolution[dim]))
    voxel_samples = np.stack(np.meshgrid(*voxel_samples, indexing='ij'), axis=-1).reshape(-1, 3)
    voxel_samples = voxel_samples[None, None, None] # [1, 1, 1, r, 3]
    
    voxel_ranges = []
    for dim in range(3):
        voxel_ranges.append(np.arange(0, grid_res[dim]))    
    voxel_grid = np.stack(np.meshgrid(*voxel_ranges, indexing='ij'), axis=-1)
    voxel_grid = (voxel_grid * voxel_size)[..., None, :] # [r1, r2, r3, -1, 3]
    
    pts = voxel_grid + voxel_samples # [r1, r2, r3, r, 3]
    pts = pts.astype(np.float32)
    return pts
