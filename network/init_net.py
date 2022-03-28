import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inplace_abn import ABN

from network.mvsnet.modules import depth_regression
from network.mvsnet.mvsnet import MVSNet, load_ckpt
from network.ops import interpolate_feats, masked_mean_var, ResEncoder, ResUNetLight, conv3x3, ResidualBlock, conv1x1
from network.render_ops import project_points_ref_views


def depth2pts3d(depth, ref_Ks, ref_poses):
    rfn, dn, h, w = depth.shape
    coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).float().to(depth.device)
    coords = coords[:, :, (1, 0)]
    coords = coords.unsqueeze(0)  # 1,h,w,2
    coords = torch.cat([coords, torch.ones([1, h, w, 1], dtype=torch.float32, device=depth.device)], -1).unsqueeze(
        -2)  # 1,h,w,1,3
    # rfn,h,w,dn,1 1,h,w,1,3
    pts3d = depth.permute(0, 2, 3, 1).unsqueeze(-1) * coords  # rfn,h,w,dn,3
    pts3d = pts3d.reshape(rfn, h * w * dn, 3).permute(0, 2, 1)  # rfn,3,h*w*dn
    pts3d = torch.inverse(ref_Ks) @ pts3d  # rfn
    R = ref_poses[:, :, :3].permute(0, 2, 1)  # rfn,3,3
    t = -R @ ref_poses[:, :, 3:]  # rfn,3,1
    pts3d = R @ pts3d + t  # rfn,3,h*w*dn
    return pts3d.permute(0, 2, 1)  # rfn,h*w*dn,3

def get_diff_feats(ref_imgs_info, depth_in):
    imgs = ref_imgs_info['imgs']  # rfn,3,h,w
    depth_range = ref_imgs_info['depth_range']
    near = depth_range[:, 0][:, None, None]  # rfn,1,1
    far = depth_range[:, 1][:, None, None]  # rfn,1,1
    near_inv, far_inv = -1 / near[..., None], -1 / far[..., None]
    depth_in = depth_in * (far_inv - near_inv) + near_inv
    depth = -1 / depth_in
    rfn, _, h, w = imgs.shape

    pts3d = depth2pts3d(depth, ref_imgs_info['Ks'], ref_imgs_info['poses'])
    _, pts2d, pts_dpt_prj, valid_mask = project_points_ref_views(ref_imgs_info, pts3d.reshape(-1, 3))   # [rfn,rfn*h*w,2] [rfn,rfn*h*w] [rfn,rfn*h*w,1]
    pts_dpt_int = interpolate_feats(depth, pts2d, padding_mode='border', align_corners=True)         # rfn,rfn*h*w,1
    pts_rgb_int = interpolate_feats(imgs, pts2d, padding_mode='border', align_corners=True)          # rfn,rfn*h*w,3

    rgb_diff = torch.abs(pts_rgb_int - imgs.permute(0, 2, 3, 1).reshape(1, rfn * h * w, 3))  # rfn,rfn*h*w,3

    pts_dpt_int = torch.clamp(pts_dpt_int, min=1e-5)
    pts_dpt_prj = torch.clamp(pts_dpt_prj, min=1e-5)
    dpt_diff = torch.abs(-1 / pts_dpt_int + 1 / pts_dpt_prj)  # rfn,rfn*h*w,1
    near_inv, far_inv = -1 / near, -1 / far
    dpt_diff = dpt_diff / (far_inv - near_inv)
    dpt_diff = torch.clamp(dpt_diff, max=1.5)

    valid_mask = valid_mask.float().unsqueeze(-1)
    dpt_mean, dpt_var = masked_mean_var(dpt_diff, valid_mask, 0)  # 1,rfn,h,w,1
    rgb_mean, rgb_var = masked_mean_var(rgb_diff, valid_mask, 0)  # 1,rfn*h*w,3
    dpt_mean = dpt_mean.reshape(rfn, h, w, 1).permute(0, 3, 1, 2)  # rfn,1,h,w
    dpt_var = dpt_var.reshape(rfn, h, w, 1).permute(0, 3, 1, 2)  # rfn,1,h,w
    rgb_mean = rgb_mean.reshape(rfn, h, w, 3).permute(0, 3, 1, 2)  # rfn,3,h,w
    rgb_var = rgb_var.reshape(rfn, h, w, 3).permute(0, 3, 1, 2)  # rfn,3,h,w

    return torch.cat([rgb_mean, rgb_var, dpt_mean, dpt_var], 1)

def extract_depth_for_init_impl(depth_range,depth):
    rfn, _, h, w = depth.shape

    near = depth_range[:, 0][:, None, None, None]  # rfn,1,1,1
    far = depth_range[:, 1][:, None, None, None]  # rfn,1,1,1
    near_inv = -1 / near
    far_inv = -1 / far
    depth = torch.clamp(depth, min=1e-5)
    depth = -1 / depth
    depth = (depth - near_inv) / (far_inv - near_inv)
    depth = torch.clamp(depth, min=0, max=1.0)
    return depth

def extract_depth_for_init(ref_imgs_info):
    depth_range = ref_imgs_info['depth_range']  # rfn,2
    depth = ref_imgs_info['depth']  # rfn,1,h,w
    return extract_depth_for_init_impl(depth_range, depth)

class DepthInitNet(nn.Module):
    default_cfg={}
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}
        self.res_net = ResEncoder()
        self.depth_skip = nn.Sequential(
            nn.Conv2d(1, 8, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 2, 2)
        )
        self.conv_out=nn.Conv2d(16+32,32,1,1)

    def forward(self, ref_imgs_info, src_imgs_info, is_train):
        depth = extract_depth_for_init(ref_imgs_info)
        imgs = ref_imgs_info['imgs']
        diff_feats = get_diff_feats(ref_imgs_info,depth)
        # imgs [b,3,h,w] depth [b,1,h,w] diff_feats [b,8,h,w]
        feats = self.res_net(torch.cat([imgs, depth, diff_feats], 1))
        depth_feats = self.depth_skip(depth)
        return self.conv_out(torch.cat([depth_feats, feats],1))

def construct_project_matrix(x_ratio, y_ratio, Ks, poses):
    rfn = Ks.shape[0]
    scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=torch.float32, device=Ks.device)
    scale_m = torch.diag(scale_m)
    ref_prj = scale_m[None, :, :] @ Ks @ poses  # rfn,3,4
    pad_vals = torch.zeros([rfn, 1, 4], dtype=torch.float32, device=ref_prj.device)
    pad_vals[:, :, 3] = 1.0
    ref_prj = torch.cat([ref_prj, pad_vals], 1)  # rfn,4,4
    return ref_prj

def construct_cost_volume_with_src(
        ref_imgs_info, src_imgs_info, mvsnet,
        cost_volume_sn, imagenet_mean, imagenet_std, is_train):
    ref_imgs = ref_imgs_info['imgs']
    src_imgs = src_imgs_info['imgs']
    rfn, _, h, w = ref_imgs.shape
    resize = not is_train and max(h, w) >= 800

    ref_imgs_ = ref_imgs
    src_imgs_ = src_imgs
    ratio = 1.0
    if resize:
        # ref_imgs = ref_imgs[:,:,:756,:1008] # 768, 1024
        if h == 768 and w == 1024:
            ref_imgs_ = F.interpolate(ref_imgs, (576, 768), mode='bilinear')
            src_imgs_ = F.interpolate(src_imgs, (576, 768), mode='bilinear')
            ratio = 576 / 768
        elif h == 800 and w == 800:
            ref_imgs_ = F.interpolate(ref_imgs, (640, 640), mode='bilinear')
            src_imgs_ = F.interpolate(src_imgs, (640, 640), mode='bilinear')
            ratio = 640 / 800
        else:
            ref_imgs_ = ref_imgs
            src_imgs_ = src_imgs
            ratio = 1.0

    with torch.no_grad():
        nn_ids = ref_imgs_info['nn_ids']  # rfn,nn
        ref_prj = construct_project_matrix(0.25 * ratio, 0.25 * ratio, ref_imgs_info['Ks'], ref_imgs_info['poses'])
        src_prj = construct_project_matrix(0.25 * ratio, 0.25 * ratio, src_imgs_info['Ks'], src_imgs_info['poses'])
        depth_vals = get_depth_vals(ref_imgs_info['depth_range'], cost_volume_sn)  # rfn,dn
        ref_imgs_imagenet = (ref_imgs_ - imagenet_mean) / imagenet_std
        src_imgs_imagenet = (src_imgs_ - imagenet_mean) / imagenet_std
        mvsnet.eval()
        batch_num = 1 if not is_train else 2
        if not is_train:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        try:
            cost_reg = mvsnet.construct_cost_volume_with_src(ref_imgs_imagenet, src_imgs_imagenet, nn_ids, ref_prj, src_prj, depth_vals, batch_num)  # rfn,dn,h,w
        except RuntimeError:
            import ipdb; ipdb.set_trace()
        cost_reg[torch.isnan(cost_reg)] = 0
        if resize: cost_reg = F.interpolate(cost_reg, (h // 4, w // 4), mode='bilinear')
        cost_reg = F.softmax(cost_reg, 1)

    depth = depth_regression(cost_reg, depth_vals)
    return cost_reg, depth

def get_depth_vals(depth_range, dn):
    near = depth_range[:, 0]
    far = depth_range[:, 1]
    interval = (1/far - 1/near)/(dn-1) # rfn
    depth_vals = 1/(1/near[:,None] + torch.arange(0,dn-1,device=depth_range.device)[None,:]*interval[:,None]) # rfn,dn-1
    depth_vals = torch.cat([depth_vals,far[:,None]],1)
    return depth_vals # rfn,dn

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class CostVolumeInitNet(nn.Module):
    default_cfg={
        'cost_volume_sn': 64,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}

        # note we do not train MVSNet here
        self.mvsnet = MVSNet(ABN)
        load_ckpt(self.mvsnet, 'network/mvsnet/mvsnet_pl.ckpt')
        for para in self.mvsnet.parameters():
            para.requires_grad = False

        imagenet_mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406], np.float32)).cuda()[None, :, None, None]
        imagenet_std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225], np.float32)).cuda()[None, :, None, None]
        self.register_buffer('imagenet_mean', imagenet_mean)
        self.register_buffer('imagenet_std', imagenet_std)

        self.res_net = ResUNetLight(out_dim=32)
        norm_layer = lambda dim: nn.InstanceNorm2d(dim, track_running_stats=False, affine=True)
        self.volume_conv2d = nn.Sequential(
            conv3x3(self.cfg['cost_volume_sn'], 32),
            ResidualBlock(32, 32, norm_layer=norm_layer),
            conv1x1(32, 32),
        )

        in_dim = 64
        depth_dim = 32
        self.depth_conv = nn.Sequential(
            conv3x3(1, depth_dim),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer),
            conv1x1(depth_dim, depth_dim),
        )
        in_dim+=depth_dim

        self.out_conv = nn.Sequential(
            conv3x3(in_dim, 32),
            ResidualBlock(32, 32, norm_layer=norm_layer),
            conv1x1(32, 32),
        )

    def forward(self, ref_imgs_info, src_imgs_info, is_train):
        cost_reg, depth = construct_cost_volume_with_src(ref_imgs_info, src_imgs_info, self.mvsnet, self.cfg['cost_volume_sn'], self.imagenet_mean, self.imagenet_std, is_train)
        ref_feats = self.res_net(ref_imgs_info['imgs'])
        volume_feats = self.volume_conv2d(cost_reg)
        depth = extract_depth_for_init_impl(ref_imgs_info['depth_range'],depth.unsqueeze(1))
        depth_feats = self.depth_conv(depth)
        volume_feats = torch.cat([volume_feats, depth_feats],1)
        return self.out_conv(torch.cat([ref_feats, volume_feats], 1))

name2init_net={
    'depth': DepthInitNet,
    'cost_volume': CostVolumeInitNet,
}