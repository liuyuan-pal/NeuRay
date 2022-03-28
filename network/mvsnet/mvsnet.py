import torch
import torch.nn as nn
import torch.nn.functional as F
from network.mvsnet.modules import ConvBnReLU, ConvBnReLU3D, depth_regression, homo_warp
from inplace_abn import InPlaceABN

class FeatureNet(nn.Module):
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x

class CostRegNet(nn.Module):
    def __init__(self, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            norm_act(8))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x

class MVSNet(nn.Module):
    def __init__(self, norm_act=InPlaceABN):
        super(MVSNet, self).__init__()
        self.feature = FeatureNet(norm_act)
        self.cost_regularization = CostRegNet(norm_act)

    def forward(self, imgs, proj_mats, depth_values):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 4, 4)
        # depth_values: (B, D)
        B, V, _, H, W = imgs.shape
        D = depth_values.shape[1]

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        imgs = imgs.reshape(B*V, 3, H, W)
        feats = self.feature(imgs) # (B*V, F, h, w)
        del imgs
        feats = feats.reshape(B, V, *feats.shape[1:]) # (B, V, F, h, w)
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        ref_proj, src_projs = proj_mats[:, 0], proj_mats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4) # (V-1, B, F, h, w)
        src_projs = src_projs.permute(1, 0, 2, 3) # (V-1, B, 4, 4)

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1) # (B, F, D, h, w)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume

        ref_proj = torch.inverse(ref_proj)
        for src_feat, src_proj in zip(src_feats, src_projs):
            warped_volume = homo_warp(src_feat, src_proj, ref_proj, depth_values)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
        del volume_sq_sum, volume_sum
        
        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance).squeeze(1)
        prob_volume = F.softmax(cost_reg, 1) # (B, D, h, w)
        depth = depth_regression(prob_volume, depth_values)
        
        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1),
                                                      pad=(0, 0, 0, 0, 1, 2)),
                                                (4, 1, 1), stride=1).squeeze(1) # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(prob_volume,
                                           torch.arange(D,
                                                        device=prob_volume.device,
                                                        dtype=prob_volume.dtype)
                                          ).long() # (B, h, w)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(prob_volume_sum4, 1, 
                                      depth_index.unsqueeze(1)).squeeze(1) # (B, h, w)

        return depth, confidence

    def construct_cost_volume(self, ref_imgs, ref_nn_idx, ref_prjs, depth_values, batch_num=2):
        # ref_imgs rfn,3,h,w
        # ref_nn_ids: rfn,nn
        # ref_prjs: rfn,4,4    note it is already scaled!!!
        # depth_values: rfn,dn
        # return: rfn,dn,h//4,w//4
        ref_feats = self.feature(ref_imgs) # rfn,f,h,w
        ref_prjs_inv = torch.inverse(ref_prjs) # rfn,4,4
        dn = depth_values.shape[1]

        rfn, n_num = ref_nn_idx.shape
        cost_reg_all = []
        for rfi in range(0,rfn,batch_num):
            volume_sum, volume_sum_sq = ref_feats[rfi:rfi+batch_num].unsqueeze(2), ref_feats[rfi:rfi+batch_num].unsqueeze(2)**2 # 1,f,1,h,w
            volume_sum, volume_sum_sq = volume_sum.repeat(1, 1, dn, 1, 1), volume_sum_sq.repeat(1, 1, dn, 1, 1)
            for ni in range(n_num):
                warp_feats = homo_warp(ref_feats[ref_nn_idx[rfi:rfi+batch_num,ni]],ref_prjs[ref_nn_idx[rfi:rfi+batch_num,ni]],
                                       ref_prjs_inv[rfi:rfi+batch_num],depth_values[rfi:rfi+batch_num]) # 1,f,dn,h,w
                volume_sum += warp_feats
                volume_sum_sq += warp_feats**2
            volume_variance = volume_sum_sq.div_(n_num+1).sub_(volume_sum.div_(n_num+1).pow_(2)) # 1,f,dn,h,w
            del volume_sum_sq, volume_sum
             # 1,dn,h,w
            cost_reg_all.append(self.cost_regularization(volume_variance).squeeze(1))
        cost_reg_all = torch.cat(cost_reg_all,0)
        return cost_reg_all

    def construct_cost_volume_with_src(self, ref_imgs, src_imgs, ref_nn_idx, ref_prjs, src_prjs, depth_values, batch_num=2):
        # ref_imgs rfn,3,h,w
        # src_imgs srn,3,h,w
        # ref_nn_ids: rfn,nn
        # ref_prjs: rfn,4,4    note it is already scaled!!!
        # src_prjs: src,4,4    note it is already scaled!!!
        # depth_values: rfn,dn
        # return: rfn,dn,h//4,w//4
        ref_feats = self.feature(ref_imgs) # rfn,f,h,w
        src_feats = self.feature(src_imgs) # src,f,h,w
        ref_prjs_inv = torch.inverse(ref_prjs) # rfn,4,4
        dn = depth_values.shape[1]

        rfn, n_num = ref_nn_idx.shape
        cost_reg_all = []
        for rfi in range(0,rfn,batch_num):
            volume_sum, volume_sum_sq = ref_feats[rfi:rfi+batch_num].unsqueeze(2), ref_feats[rfi:rfi+batch_num].unsqueeze(2)**2 # 1,f,1,h,w
            volume_sum, volume_sum_sq = volume_sum.repeat(1, 1, dn, 1, 1), volume_sum_sq.repeat(1, 1, dn, 1, 1)
            for ni in range(n_num):
                warp_feats = homo_warp(src_feats[ref_nn_idx[rfi:rfi+batch_num,ni]],src_prjs[ref_nn_idx[rfi:rfi+batch_num,ni]],
                                       ref_prjs_inv[rfi:rfi+batch_num],depth_values[rfi:rfi+batch_num]) # 1,f,dn,h,w
                volume_sum += warp_feats
                volume_sum_sq += warp_feats**2
            volume_variance = volume_sum_sq.div_(n_num+1).sub_(volume_sum.div_(n_num+1).pow_(2)) # 1,f,dn,h,w
            del volume_sum_sq, volume_sum
             # 1,dn,h,w
            cost_reg_all.append(self.cost_regularization(volume_variance).squeeze(1))
        cost_reg_all = torch.cat(cost_reg_all,0)
        return cost_reg_all


def extract_model_state_dict(ckpt_path, prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        for k, v in checkpoint['state_dict'].items():
            if not k.startswith('model.'):
                continue
            k = k[6:] # remove 'model.'
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    else: # if it only has model weights
        for k, v in checkpoint.items():
            for prefix in prefixes_to_ignore:
                if k.startswith(prefix):
                    print('ignore', k)
                    break
            else:
                checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)