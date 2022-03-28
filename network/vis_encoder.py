import torch.nn as nn
import torch

from network.ops import conv3x3, ResidualBlock, conv1x1

class DefaultVisEncoder(nn.Module):
    default_cfg={}
    def __init__(self, cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}
        norm_layer = lambda dim: nn.InstanceNorm2d(dim,track_running_stats=False,affine=True)
        self.out_conv=nn.Sequential(
            conv3x3(64, 32),
            ResidualBlock(32, 32, norm_layer=norm_layer),
            ResidualBlock(32, 32, norm_layer=norm_layer),
            conv1x1(32, 32),
        )

    def forward(self, ray_feats, imgs_feats):
        feats = self.out_conv(torch.cat([imgs_feats, ray_feats],1))
        return feats

name2vis_encoder={
    'default': DefaultVisEncoder,
}