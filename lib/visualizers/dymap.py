import numpy as np
from lib.config import cfg
import os
import cv2
from termcolor import colored
from lib.utils import img_utils
import imageio

class Visualizer:
    def __init__(self):
        result_dir = cfg.result_dir
        self.result_dir = os.path.join(result_dir, cfg.vis_mode)
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))
        os.system('mkdir -p {}'.format(self.result_dir))

        self.novel_view = cfg.vis_novel_view
        
        if cfg.store_rgb:
            self.rgbs = []
        if cfg.store_depth:
            self.depths = []
        if cfg.store_acc:
            self.accs = []
        
        self.to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    def visuliaze_interactive(self, output, batch):        
        ret = {'pred': output['rgb_map'][0]}
        return ret

    def visualize(self, output, batch):
        view_index = int(batch['cam_ind'].item())
        frame_index = int(batch['frame_index'].item())
        if cfg.get('store_rgb', True):
            img_pred = img_utils.recover_shape(output['rgb_map'], batch)
            if cfg.get('adjust_hsv', False):
                img_pred = img_utils.adjust_hsv(img_pred)

            cv2.imwrite('{}/view{:04d}_{:04d}.png'.format(self.result_dir, frame_index, view_index), (img_pred[..., [2, 1, 0]] * 255))
            
            self.rgbs.append(img_pred)
                
        if cfg.get('store_diff', True) and 'rgb' in batch.keys():
            img_pred = img_utils.recover_shape(output['rgb_map'], batch)
            img_gt = img_utils.recover_shape(batch['rgb'], batch)
            img_diff = (img_gt - img_pred) ** 2
            cv2.imwrite('{}/view{:04d}_{:04d}_diff.png'.format(self.result_dir, frame_index, view_index), (img_diff[..., [2, 1, 0]] * 255))

        if cfg.get('store_depth', False) and 'depth_map' in output.keys():
            depth_pred = img_utils.recover_shape(output['depth_map'], batch)
            near = batch['near'][0].min().cpu().numpy()
            far = batch['far'][0].max().cpu().numpy()
            depth_pred = img_utils.visualize_depth_numpy(depth_pred, near, far)
            cv2.imwrite('{}/view{:04d}_{:04d}_depth.png'.format(self.result_dir, frame_index, view_index), depth_pred)
            self.depths.append(depth_pred[..., [2, 1, 0]])
            
        if cfg.get('store_acc', False) and 'acc_map' in output.keys():
            acc_pred = img_utils.recover_shape(output['acc_map'], batch)
            cv2.imwrite('{}/view{:04d}_{:04d}_acc.png'.format(self.result_dir, frame_index, view_index), acc_pred * 255)
            self.accs.append(acc_pred)
            
        if 'img' in batch or 'rgb' in batch and not self.novel_view:
            if 'img' in batch:
                img = batch['img'][0].detach().cpu().numpy()
            else:
                img = img_utils.recover_shape(batch['rgb'], batch)
            cv2.imwrite('{}/view{:04d}_{:04d}_gt.png'.format(self.result_dir, frame_index, view_index),
                        (img[..., [2, 1, 0]] * 255))

    def summarize(self):
        if not cfg.get('store_video', False):
            return
        
        if cfg.get('fixed_time', False):
            video_name = 'video_fixed_time'
        elif cfg.get('fixed_view', False):
            video_name = 'video_fixed_view'
        else:
            video_name = 'video'
        
        if cfg.get('store_rgb', True) and len(self.rgbs) > 0:
            print('saving video_rgb')
            rgbs = np.array(self.rgbs)                
            imageio.mimwrite(os.path.join(self.result_dir, '{}.rgb.mp4'.format(video_name)), self.to8b(rgbs), fps=30, quality=8)

        if cfg.get('store_depth', False) and len(self.depths) > 0:
            print('saving video_depth')
            depths = np.array(self.depths)
            imageio.mimwrite(os.path.join(self.result_dir, '{}.depth.mp4'.format(video_name)), self.to8b(depths / np.max(depths)), fps=30, quality=8)
            
        if cfg.get('store_acc', False) and len(self.accs) > 0:
            print('saving video_accumulation')
            accs = np.array(self.accs)
            imageio.mimwrite(os.path.join(self.result_dir, '{}.acc.mp4'.format(video_name)), self.to8b(accs), fps=30, quality=8)
        
        if cfg.get('clean_img', False):
            os.system('rm {}/*.png'.format(self.result_dir))