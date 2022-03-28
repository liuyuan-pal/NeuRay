import torch
from torch import nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN
from kornia.utils import create_meshgrid

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

def homo_warp(src_feat, src_proj, ref_proj_inv, depth_values):
    # src_feat: (B, C, H, W)
    # src_proj: (B, 4, 4)
    # ref_proj_inv: (B, 4, 4)
    # depth_values: (B, D)
    # out: (B, C, D, H, W)
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device
    dtype = src_feat.dtype

    transform = src_proj @ ref_proj_inv
    R = transform[:, :3, :3] # (B, 3, 3)
    T = transform[:, :3, 3:] # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False) # (1, H, W, 2)
    ref_grid = ref_grid.to(device).to(dtype)
    ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H*W) # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
    ref_grid_d = ref_grid.unsqueeze(2) * depth_values.view(B, 1, D, 1) # (B, 3, D, H*W)
    ref_grid_d = ref_grid_d.view(B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T # (B, 3, D*H*W)
    del ref_grid_d, ref_grid, transform, R, T # release (GPU) memory
    div_val = src_grid_d[:, -1:]
    div_val[div_val<1e-4] = 1e-4
    src_grid = src_grid_d[:, :2] / div_val # divide by depth (B, 2, D*H*W)
    del src_grid_d, div_val
    src_grid[:, 0] = src_grid[:, 0]/((W - 1) / 2) - 1 # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1]/((H - 1) / 2) - 1 # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1) # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, H*W, 2)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True) # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, C, D, H, W)

    return warped_src_feat

def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth