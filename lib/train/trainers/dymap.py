import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
import tqdm


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)

        self.img2mse = lambda x, y: torch.mean((x - y)**2)
        self.mse2psnr = lambda x : -10. * torch.log10(x)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        gt = batch['rgb']
        pred = ret['rgb_map']    
        
        if batch['mask_at_box'].shape == gt.shape[:-1] and cfg.get('with_mask', True):
            mask = batch['mask_at_box']
            non_edge = batch['edge'] != 1
            mask = mask * non_edge
            img_loss = self.img2mse(pred[mask], gt[mask])
        else:
            img_loss = self.img2mse(pred, gt)

        psnr = self.mse2psnr(img_loss)
        scalar_stats.update({
            'img_loss': img_loss,
            'psnr': psnr,
        })
        loss += img_loss

        if cfg.get('with_mask', True):
            non_edge = batch['edge'] != 1
            mask_loss = self.img2mse(ret['acc_map'][non_edge], batch['occ'][non_edge].float())
        else:
            acc_map = torch.clamp(ret['acc_map'], min=1e-6, max=1-1e-6)
            mask_loss = -(acc_map *torch.log(acc_map) + (1 - acc_map) * torch.log(1 - acc_map)).mean()

        weight = cfg.mask_weight        
        scalar_stats.update({'mask_loss': mask_loss})
        loss += weight * mask_loss

        # encoder N(0, 1) KL divergence loss
        if cfg.get('use_encoder', False):
            kldiv_loss = torch.mean(self.net.kldiv_loss)
            scalar_stats.update({'kldiv_loss': kldiv_loss})
            loss += cfg.kldiv_weight * kldiv_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
