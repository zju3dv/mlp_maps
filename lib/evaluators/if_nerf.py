import numpy as np
from lib.config import cfg
from skimage.metrics import structural_similarity as compare_ssim
import os
import cv2
import lpips
import torch
from termcolor import colored
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpip = []
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()
        
        self.result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(self.result_dir))

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch, return_img=False):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        white_bkgd = int(cfg.white_bkgd)
        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred 
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt
        
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(self.result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(self.result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        # crop the object region
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]

        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, channel_axis=-1)

        if return_img:
            return ssim, img_pred, img_gt
        else:
            return ssim

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        if cfg.get('with_mask', True):
            non_edge = (batch['edge'][0] != 1).detach().cpu().numpy()
        else:
            non_edge = np.ones_like(rgb_gt[..., 0], dtype=np.bool8)
        
        mse = np.mean((rgb_pred[non_edge] - rgb_gt[non_edge])**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred[non_edge], rgb_gt[non_edge])
        self.psnr.append(psnr)

        ssim, img_pred, img_gt = self.ssim_metric(rgb_pred, rgb_gt, batch, return_img=True)
        self.ssim.append(ssim)

        pred = (torch.Tensor(img_pred)[None].permute(0, 3, 1, 2) - 0.5) * 2
        gt = (torch.Tensor(img_gt)[None].permute(0, 3, 1, 2) - 0.5) * 2
        lpip = self.loss_fn_vgg(pred.cuda(), gt.cuda()).item()
        self.lpip.append(lpip)
        
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        print('frame{}-view{}, psnr:{}, ssim:{}, lpip:{}'.format(frame_index, view_index, psnr, ssim, lpip))
        
    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpip': self.lpip}
        np.save(result_path, metrics)
        print('mse: {}'.format(np.mean(self.mse)))
        print('psnr: {}'.format(np.mean(self.psnr)))
        print('ssim: {}'.format(np.mean(self.ssim)))
        print('lpip: {}'.format(np.mean(self.lpip)))
        
        return {
            'psnr': np.mean(self.psnr),
            'ssim': np.mean(self.ssim),
            'lpip': np.mean(self.lpip),
        }



