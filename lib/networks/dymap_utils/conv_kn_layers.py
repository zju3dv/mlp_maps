import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.config import cfg
from lib.networks.dymap_utils.cnn_layers import Conv2dELR, LinearELR, ConvTranspose2dELR, Reshape

def multiplane_params(trip_kn_params):
    params = []
    for plane_type in cfg.kn_plane_types:
        param = trip_kn_params[plane_type].permute(
            0, 2, 1)  # [B, num_networks, num_params]
        params.append(param)
    params = torch.cat(params, dim=1)  # [B, num_all_networks, num_params]
    num_kn_params = params.shape[-1]
    params = params.view(-1, num_kn_params)
    return params


def multiplane_features(trip_kn_features):
    features = []
    for plane_type in cfg.feature_plane_types:
        feature = trip_kn_features[plane_type]
        features.append(feature)
    features = torch.stack(features, dim=1)  # [B, M, C, H, W]
    B, M, C, H, W = features.shape
    features = features.view(-1, C, H, W)  # [B * M, C, H, W]
    return features


def sample_grid_feature(norm, trip_kn_features, plane_slices):
    pts = norm.detach() * 2. - 1.
    B, N, _ = pts.shape
    M = cfg.feature_num_planes
    pts = pts[..., plane_slices].view(B, N, M, 2)
    pts = pts.permute(0, 2, 1, 3)  # [B, M, N, 2]
    pts = pts.contiguous().view(-1, N, 2) 

    # trip_kn_features: [B * M, C, H, W]
    feature = F.grid_sample(trip_kn_features,
                            pts[:, None, :, :],
                            padding_mode='zeros',
                            align_corners=True)

    feature = feature[:, :, 0, :].permute(0, 2, 1)  # [B * M, N, C]
    feature = feature.view(B, M, N, -1)
    feature = torch.sum(feature, dim=1)  # [B, N, C]
    return feature
            
class PlaneDecoder(nn.Module):
    def __init__(self, mlp_map_size, num_kn_params):
        super(PlaneDecoder, self).__init__()

        self.plane_size = cfg.plane_size

        inch = cfg.code_dim
        outch = 64
        hstart = 4
        chstart = 256

        self.triplane_ch = cfg.feature_channel * 3

        nlayers = int(math.log2(self.plane_size)) - int(math.log2(hstart))

        lastch = chstart
        dims = (1, hstart, hstart)
        
        # backbone layer
        layers = []
        layers.append(
            LinearELR(inch, chstart * dims[1] * dims[2],
                      act=nn.LeakyReLU(0.2)))
        layers.append(Reshape(-1, chstart, dims[1], dims[2]))

        for i in range(nlayers):
            nextch = lastch if i % 2 == 0 else lastch // 2

            layers.append(
                ConvTranspose2dELR(lastch,
                                   (outch if i == nlayers - 1 else nextch),
                                   4,
                                   2,
                                   1,
                                   ub=(dims[1] * 2, dims[2] * 2),
                                   norm=None,
                                   act=nn.LeakyReLU(0.2)))

            lastch = nextch
            dims = (dims[0], dims[1] * 2, dims[2] * 2)

        self.backbone = nn.Sequential(*layers)
        
        # triplane feature
        self.feat_conv = nn.Conv2d(64,
                                   self.triplane_ch,
                                   kernel_size=3,
                                   padding=1)
        
        # density params
        self.density_ch = cfg.feature_channel * 3
        self.density_conv =  nn.Conv2d(64,
                                      self.density_ch,
                                      kernel_size=3,
                                      padding=1)
        
        # rgb params
        layers = []
        nlayers = int(math.log2(self.plane_size)) - int(math.log2(mlp_map_size))

        for i in range(nlayers):
            nextch = outch if i % 2 == 0 else outch * 2
            layers.append(
                nn.Conv2d(outch, nextch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            outch = nextch
   
        self.kn_backbone = nn.Sequential(*layers)
        self.kn_x_conv = nn.Conv2d(outch, num_kn_params, \
                                   kernel_size=3, stride=1, padding=1)
        self.kn_y_conv = nn.Conv2d(outch, num_kn_params, \
                                   kernel_size=3, stride=1, padding=1)
        self.kn_z_conv = nn.Conv2d(outch, num_kn_params, \
                                   kernel_size=3, stride=1, padding=1)

    def forward(self, enc):
        net = self.backbone(enc)

        features = self.feat_conv(net)
        ch = cfg.feature_channel
        trip_kn_features_dict = {
            'xy': features[:, :ch],
            'yz': features[:, ch:ch * 2],
            'xz': features[:, ch * 2:]
        }

        kn_net = self.kn_backbone(net)
        kn_x_params = self.kn_x_conv(kn_net)
        kn_y_params = self.kn_y_conv(kn_net)
        kn_z_params = self.kn_z_conv(kn_net)
        trip_rgb_params_dict = {
            'xy': kn_z_params.view(kn_z_params.size(0), kn_z_params.size(1), -1),
            'yz': kn_x_params.view(kn_x_params.size(0), kn_x_params.size(1), -1),
            'xz': kn_y_params.view(kn_y_params.size(0), kn_y_params.size(1), -1),
        }
        
        densitys = self.density_conv(net)
        ch = cfg.feature_channel
        trip_density_params_dict = {
                'xy': densitys[:, :ch],
                'yz': densitys[:, ch:ch * 2],
                'xz': densitys[:, ch * 2:]
        }
        
        ret = {
            'trip_kn_rgb_params': multiplane_params(trip_rgb_params_dict),
            'trip_kn_features': multiplane_features(trip_kn_features_dict),
            'trip_kn_density_params': multiplane_features(trip_density_params_dict)
        }

        return ret
    
class Encoder(torch.nn.Module):
    def __init__(self, ninputs=1, size=(512, 512), nlayers=7, conv=Conv2dELR, lin=LinearELR):
        super(Encoder, self).__init__()
        self.ninputs = ninputs
        height, width = size
        self.nlayers = nlayers

        ypad = ((height + 2 ** nlayers - 1) // 2 ** nlayers) * 2 ** nlayers - height
        xpad = ((width + 2 ** nlayers - 1) // 2 ** nlayers) * 2 ** nlayers - width
        self.pad = nn.ZeroPad2d((xpad // 2, xpad - xpad // 2, ypad // 2, ypad - ypad // 2))

        self.downwidth = ((width + 2 ** nlayers - 1) // 2 ** nlayers)
        self.downheight = ((height + 2 ** nlayers - 1) // 2 ** nlayers)

        # compile layers
        layers = []
        inch, outch = 3, 64
        for i in range(nlayers):
            layers.append(conv(inch, outch, 4, 2, 1, norm="demod", act=nn.LeakyReLU(0.2)))

            if inch == outch:
                outch = inch * 2
            else:
                inch = outch
            if outch > 256:
                outch = 256

        self.down1 = nn.ModuleList([nn.Sequential(*layers)
                for i in range(self.ninputs)])
        self.down2 = lin(256 * self.ninputs * self.downwidth * self.downheight, cfg.code_dim * 2, norm="demod", act=nn.LeakyReLU(0.2))
        self.mu = lin(cfg.code_dim * 2, cfg.code_dim)
        self.logstd = lin(cfg.code_dim * 2, cfg.code_dim)

    def forward(self, x):
        x = self.pad(x)
        x = [self.down1[i](x[:, i*3:(i+1)*3, :, :]).view(x.size(0), 256 * self.downwidth * self.downheight)
                for i in range(self.ninputs)]
        x = torch.cat(x, dim=1)
        x = self.down2(x)

        mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        if cfg.mode == 'train':
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
            kldiv_loss = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)
        else:
            z = mu
            kldiv_loss = 0.
        
        return z, kldiv_loss