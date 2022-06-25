import time
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from skimage.io import imsave
from tqdm import tqdm

from dataset.database import parse_database_name, get_database_split
from network.aggregate_net import name2agg_net
from network.dist_decoder import name2dist_decoder
from network.init_net import name2init_net, DepthInitNet, CostVolumeInitNet
from network.ops import ResUNetLight
from network.sph_solver import SphericalHarmonicsSolver
from network.vis_encoder import name2vis_encoder
from network.render_ops import *
from utils.base_utils import to_cuda, load_cfg, color_map_backward, get_coords_mask
from utils.draw_utils import concat_images_list
from utils.imgs_info import build_imgs_info, imgs_info_to_torch, imgs_info_slice
from utils.view_select import compute_nearest_camera_indices, select_working_views, select_working_views_by_overlap


class NeuralRayBaseRenderer(nn.Module):
    base_cfg={
        'vis_encoder_type': 'default',
        'vis_encoder_cfg': {},

        'dist_decoder_type': 'mixture_logistics',
        'dist_decoder_cfg': {},

        'agg_net_type': 'default',
        'agg_net_cfg': {},

        'use_hierarchical_sampling': False,
        'fine_agg_net_cfg': {},
        'fine_dist_decoder_cfg': {},
        'fine_depth_sample_num': 64,
        'fine_depth_use_all': False,

        'ray_batch_num': 2048,
        'depth_sample_num': 64,
        'alpha_value_ground_state': -15,
        'use_dr_prediction': False,
        'use_nr_color_for_dr': False,
        'use_self_hit_prob': False,
        'use_ray_mask': True,
        'ray_mask_view_num': 2,
        'ray_mask_point_num': 8,

        'render_depth': False,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        self.vis_encoder = name2vis_encoder[self.cfg['vis_encoder_type']](self.cfg['vis_encoder_cfg'])
        self.dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](self.cfg['dist_decoder_cfg'])
        self.image_encoder = ResUNetLight(3, [1,2,6,4], 32, inplanes=16)
        self.agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg['agg_net_cfg'])
        if self.cfg['use_hierarchical_sampling']:
            self.fine_dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](self.cfg['fine_dist_decoder_cfg'])
            self.fine_agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg['fine_agg_net_cfg'])

        # if self.cfg['use_dr_prediction'] and not self.cfg['use_nr_color_for_dr']:
        self.sph_fitter = SphericalHarmonicsSolver(3)

    def predict_proj_ray_prob(self, prj_dict, ref_imgs_info, que_dists, is_fine):
        rfn, qn, rn, dn, _ = prj_dict['mask'].shape
        # decode ray prob
        if is_fine:
            prj_mean, prj_var, prj_vis, prj_aw = self.fine_dist_decoder(prj_dict['ray_feats'])
        else:
            prj_mean, prj_var, prj_vis, prj_aw = self.dist_decoder(prj_dict['ray_feats'])

        alpha_values, visibility, hit_prob = self.dist_decoder.compute_prob(
            prj_dict['depth'].squeeze(-1),que_dists.unsqueeze(0),prj_mean,prj_var,
            prj_vis, prj_aw, True, ref_imgs_info['depth_range'])
        # post process
        prj_dict['alpha'] = alpha_values.reshape(rfn,qn,rn,dn,1) * prj_dict['mask'] + \
                            (1 - prj_dict['mask']) * self.cfg['alpha_value_ground_state']
        prj_dict['vis'] = visibility.reshape(rfn,qn,rn,dn,1) * prj_dict['mask']
        prj_dict['hit_prob'] = hit_prob.reshape(rfn,qn,rn,dn,1) * prj_dict['mask']
        return prj_dict

    def predict_alpha_values_dr(self, prj_dict):
        eps = 1e-5
        # predict alpha values for query ray
        prj_alpha, prj_vis = prj_dict['alpha'], prj_dict['vis']
        alpha = torch.sum(prj_vis * prj_alpha, 0) / (torch.sum(prj_vis, 0) + eps)  # qn,rn,dn,1
        invalid_ray_mask = torch.sum(prj_dict['mask'].int().squeeze(-1), 0) == 0
        alpha = alpha * (1 - invalid_ray_mask.float().unsqueeze(-1)) + \
                invalid_ray_mask.float().unsqueeze(-1) * self.cfg['alpha_value_ground_state']
        rfn, qn, rn, dn, _ = prj_alpha.shape
        return alpha.reshape(qn, rn, dn)

    def predict_colors_dr(self,prj_dict,que_dir):
        eps = 1e-3
        prj_hit_prob, prj_rgb, prj_dir = prj_dict['hit_prob'], prj_dict['rgb'], prj_dict['dir']
        rfn, qn, rn, dn, _ = prj_rgb.shape
        pn = qn * rn * dn
        que_dir = que_dir.reshape(pn, 3)  # pn,3
        prj_dir = prj_dir.reshape(rfn, pn, 3)
        prj_rgb = prj_rgb.reshape(rfn, pn, 3)
        prj_hit_prob = prj_hit_prob.reshape(rfn,pn,1)
        prj_weights = prj_hit_prob / (torch.sum(prj_hit_prob, 0, keepdim=True) + eps) # rfn,pn,3

        # pn,k,3
        theta = self.sph_fitter(prj_dir.permute(1,0,2),
                                prj_rgb.permute(1,0,2),
                                prj_weights.squeeze(-1).permute(1,0)) # pn,rfn
        colors = self.sph_fitter.predict(que_dir.unsqueeze(1),theta)
        colors = colors.squeeze(1).reshape(qn,rn,dn,3)
        return colors

    def direct_rendering(self, prj_dict, que_dir, colors_nr=None):
        alpha_values = self.predict_alpha_values_dr(prj_dict)               # qn,rn,dn
        if self.cfg['use_nr_color_for_dr']:
            colors = colors_nr
        else:
            colors = self.predict_colors_dr(prj_dict,que_dir)               # qn,rn,dn,3
        # the alpha values is *logits* now, we decode it to *real alpha values*
        alpha_values = self.dist_decoder.decode_alpha_value(alpha_values)   # qn,rn,dn
        hit_prob = alpha_values2hit_prob(alpha_values)                      # qn,rn,dn
        pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors,2)
        return hit_prob, colors, pixel_colors

    def get_img_feats(self,ref_imgs_info, prj_dict):
        rfn, _, h, w = ref_imgs_info['imgs'].shape
        rfn, qn, rn, dn, _ = prj_dict['pts'].shape

        img_feats = ref_imgs_info['img_feats']
        prj_img_feats = interpolate_feature_map(img_feats, prj_dict['pts'].reshape(rfn, qn * rn * dn, 2),
                                                prj_dict['mask'].reshape(rfn, qn * rn * dn), h, w,)
        prj_dict['img_feats'] = prj_img_feats.reshape(rfn, qn, rn, dn, -1)
        return prj_dict

    def predict_self_hit_prob_impl(self, que_ray_feats, que_depth, que_dists, depth_range, is_fine):
        if is_fine: ops = self.fine_dist_decoder
        else: ops = self.dist_decoder
        mean, var, vis, aw = ops(que_ray_feats)  # qn,rn,1
        if aw is not None: aw = aw.unsqueeze(2)
        if vis is not None: vis = vis.unsqueeze(2)
        if mean is not None: mean = mean.unsqueeze(2)
        if var is not None: var = var.unsqueeze(2)
        # qn, rn, dn
        _, _, hit_prob_que = ops.compute_prob(que_depth, que_dists, mean, var, vis, aw, False, depth_range)
        return hit_prob_que

    def predict_self_hit_prob(self, que_imgs_info, que_depth, que_dists, is_fine):
        _, _, h, w = que_imgs_info['imgs'].shape
        qn, rn, _ = que_imgs_info['coords'].shape
        mask = torch.ones([qn, rn], dtype=torch.float32, device=que_imgs_info['coords'].device)
        que_ray_feats = interpolate_feature_map(que_imgs_info['ray_feats'], que_imgs_info['coords'], mask, h, w)  # qn,rn,f
        hit_prob_que = self.predict_self_hit_prob_impl(que_ray_feats, que_depth, que_dists, que_imgs_info['depth_range'], is_fine)
        return hit_prob_que

    def network_rendering(self, prj_dict, que_dir, is_fine):
        if is_fine:
            density, colors = self.fine_agg_net(prj_dict, que_dir)
        else:
            density, colors = self.agg_net(prj_dict, que_dir)

        alpha_values = 1.0 - torch.exp(-torch.relu(density))
        hit_prob = alpha_values2hit_prob(alpha_values)
        pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors,2)
        return hit_prob, colors, pixel_colors

    def render_by_depth(self, que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine):
        ref_imgs_info = ref_imgs_info.copy()
        que_imgs_info = que_imgs_info.copy()
        que_dists = depth2inv_dists(que_depth,que_imgs_info['depth_range'])
        que_pts, que_dir = depth2points(que_imgs_info, que_depth)

        prj_dict = project_points_dict(ref_imgs_info, que_pts)
        prj_dict = self.predict_proj_ray_prob(prj_dict, ref_imgs_info, que_dists, is_fine)
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)

        hit_prob_nr, colors_nr, pixel_colors_nr = self.network_rendering(prj_dict, que_dir, is_fine)
        outputs={'pixel_colors_nr': pixel_colors_nr, 'hit_prob_nr': hit_prob_nr}

        # direct rendering
        if self.cfg['use_dr_prediction']:
            hit_prob_dr, colors_dr, pixel_colors_dr = self.direct_rendering(prj_dict, que_dir, colors_nr)
            outputs['pixel_colors_dr'] = pixel_colors_dr
            outputs['hit_prob_dr'] = hit_prob_dr

        # predict query hit prob
        if is_train and self.cfg['use_self_hit_prob']:
            outputs['hit_prob_self'] = self.predict_self_hit_prob(que_imgs_info, que_depth, que_dists, is_fine)

        if 'imgs' in que_imgs_info:
            outputs['pixel_colors_gt'] = interpolate_feats(
                que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)

        if self.cfg['use_ray_mask']:
            outputs['ray_mask'] = torch.sum(prj_dict['mask'].int(),0)>self.cfg['ray_mask_view_num'] # qn,rn,dn,1
            outputs['ray_mask'] = torch.sum(outputs['ray_mask'],2)>self.cfg['ray_mask_point_num'] # qn,rn
            outputs['ray_mask'] = outputs['ray_mask'][...,0]

        if self.cfg['render_depth']:
            # qn,rn,dn
            outputs['render_depth'] = torch.sum(hit_prob_nr * que_depth, -1) # qn,rn
        return outputs

    def fine_render_impl(self, coarse_render_info, que_imgs_info, ref_imgs_info, is_train):
        fine_depth = sample_fine_depth(coarse_render_info['depth'], coarse_render_info['hit_prob'].detach(),
                                       que_imgs_info['depth_range'], self.cfg['fine_depth_sample_num'], is_train)

        # qn, rn, fdn+dn
        if self.cfg['fine_depth_use_all']:
            que_depth = torch.sort(torch.cat([coarse_render_info['depth'], fine_depth], -1), -1)[0]
        else:
            que_depth = torch.sort(fine_depth, -1)[0]
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, True)
        return outputs

    def render_impl(self, que_imgs_info, ref_imgs_info, is_train):
        # [qn,rn,dn]
        que_depth, _ = sample_depth(que_imgs_info['depth_range'], que_imgs_info['coords'], self.cfg['depth_sample_num'], False)
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, False)
        if self.cfg['use_hierarchical_sampling']:
            coarse_render_info= {'depth': que_depth, 'hit_prob': outputs['hit_prob_nr']}
            fine_outputs = self.fine_render_impl(coarse_render_info, que_imgs_info, ref_imgs_info, is_train)
            for k, v in fine_outputs.items():
                outputs[k + "_fine"] = v
        return outputs

    def render(self, que_imgs_info, ref_imgs_info, is_train):
        ref_img_feats = self.image_encoder(ref_imgs_info['imgs'])
        ref_imgs_info['img_feats'] = ref_img_feats
        ref_imgs_info['ray_feats'] = self.vis_encoder(ref_imgs_info['ray_feats'], ref_img_feats)

        if is_train and self.cfg['use_self_hit_prob']:
            que_img_feats = self.image_encoder(que_imgs_info['imgs'])
            que_imgs_info['ray_feats'] = self.vis_encoder(que_imgs_info['ray_feats'], que_img_feats)

        ray_batch_num = self.cfg["ray_batch_num"]
        coords = que_imgs_info['coords']
        ray_num = coords.shape[1]
        render_info_all = {}
        for ray_id in range(0,ray_num,ray_batch_num):
            que_imgs_info['coords']=coords[:,ray_id:ray_id+ray_batch_num]
            render_info = self.render_impl(que_imgs_info,ref_imgs_info,is_train)
            output_keys = [k for k in render_info.keys() if is_train or (not k.startswith('hit_prob'))]
            for k in output_keys:
                v = render_info[k]
                if k not in render_info_all:
                    render_info_all[k]=[]
                render_info_all[k].append(v)

        for k, v in render_info_all.items():
            render_info_all[k]=torch.cat(v,1)

        return render_info_all

class NeuralRayGenRenderer(NeuralRayBaseRenderer):
    default_cfg={
        'init_net_type': 'depth',
        'init_net_cfg': {},
        'use_depth_loss': False,
        'depth_loss_coords_num': 8192,
    }
    def __init__(self, cfg):
        cfg={**self.default_cfg,**cfg}
        super().__init__(cfg)
        self.init_net=name2init_net[self.cfg['init_net_type']](self.cfg['init_net_cfg'])

    def render_call(self, que_imgs_info, ref_imgs_info, is_train, src_imgs_info=None):
        ref_imgs_info['ray_feats'] = self.init_net(ref_imgs_info, src_imgs_info, is_train)
        return self.render(que_imgs_info, ref_imgs_info, is_train)

    def gen_depth_loss_coords(self,h,w,device):
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).reshape(-1, 2).to(device)
        num = self.cfg['depth_loss_coords_num']
        idxs = torch.randperm(coords.shape[0])
        idxs = idxs[:num]
        coords = coords[idxs]
        return coords

    def predict_mean_for_depth_loss(self, ref_imgs_info):
        ray_feats = ref_imgs_info['ray_feats'] # rfn,f,h',w'
        ref_imgs = ref_imgs_info['imgs'] # rfn,3,h,w
        rfn, _, h, w = ref_imgs.shape
        coords = self.gen_depth_loss_coords(h,w,ref_imgs.device) # pn,2
        coords = coords.unsqueeze(0).repeat(rfn,1,1) # rfn,pn,2

        batch_num = self.cfg['depth_loss_coords_num']
        pn = coords.shape[1]
        coords_dist_mean, coords_dist_mean_2, coords_dist_mean_fine, coords_dist_mean_fine_2 = [], [], [], []
        for ci in range(0, pn, batch_num):
            coords_ = coords[:,ci:ci+batch_num]
            mask_ = torch.ones(coords_.shape[:2], dtype=torch.float32, device=ref_imgs.device)
            coords_ray_feats_ = interpolate_feature_map(ray_feats, coords_, mask_, h, w) # rfn,pn,f
            coords_dist_mean_ = self.dist_decoder.predict_mean(coords_ray_feats_)  # rfn,pn
            coords_dist_mean_2.append(coords_dist_mean_[..., 1])
            coords_dist_mean_ = coords_dist_mean_[..., 0]

            coords_dist_mean.append(coords_dist_mean_)
            if self.cfg['use_hierarchical_sampling']:
                coords_dist_mean_fine_ = self.fine_dist_decoder.predict_mean(coords_ray_feats_)
                coords_dist_mean_fine_2.append(coords_dist_mean_fine_[..., 1])
                coords_dist_mean_fine_ = coords_dist_mean_fine_[..., 0]  # use 0 for depth supervision
                coords_dist_mean_fine.append(coords_dist_mean_fine_)

        coords_dist_mean = torch.cat(coords_dist_mean, 1)
        outputs = {'depth_mean': coords_dist_mean, 'depth_coords': coords}
        if len(coords_dist_mean_2)>0:
            coords_dist_mean_2 = torch.cat(coords_dist_mean_2, 1)
            outputs['depth_mean_2'] = coords_dist_mean_2
        if self.cfg['use_hierarchical_sampling']:
            coords_dist_mean_fine = torch.cat(coords_dist_mean_fine, 1)
            outputs['depth_mean_fine'] = coords_dist_mean_fine
            if len(coords_dist_mean_fine_2)>0:
                coords_dist_mean_fine_2 = torch.cat(coords_dist_mean_fine_2, 1)
                outputs['depth_mean_fine_2'] = coords_dist_mean_fine_2
        return outputs

    def forward(self,data):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()
        is_train = 'eval' not in data

        src_imgs_info = data['src_imgs_info'].copy() if 'src_imgs_info' in data else None
        render_outputs = self.render_call(que_imgs_info, ref_imgs_info, is_train, src_imgs_info)
        if (self.cfg['use_depth_loss'] and 'true_depth' in ref_imgs_info) or (not is_train):
            render_outputs.update(self.predict_mean_for_depth_loss(ref_imgs_info))
        return render_outputs



class NeuralRayFtRenderer(NeuralRayBaseRenderer):
    default_cfg={
        # scene
        'database_name': 'nerf_synthetic/lego/black_400',
        "database_split_type": 'val_all',

        # input config
        "ref_pad_interval": 16,
        "use_consistent_depth_range": True,

        # training related
        'gen_cfg': None, # 'configs/train/gen/ft_lr_neuray_lego.yaml'
        "use_validation": True,
        "validate_initialization": True, # visualize rendered images of inited neuray on the val set
        "init_view_num": 8, # number of neighboring views used in initialization: this should be consistent with the number used in generalization model
        "init_src_view_num": 3,

        # neighbor view selection in training
        "include_self_prob": 0.01,
        "neighbor_view_num": 8,  # number of neighboring views
        "neighbor_pool_ratio": 2,
        "train_ray_num": 512,
        "foreground_ratio": 0.5,

        # used in train from scratch
        'ray_feats_res': [200,200], # size of raw visibility feature G': H=200,W=200
        'ray_feats_dim': 32, # channel number of raw visibility feature G'

    }
    def __init__(self, cfg):
        cfg = {**self.default_cfg,**cfg}
        super().__init__(cfg)
        self.cached=False
        self.database = parse_database_name(self.cfg['database_name'])
        self.ref_ids, self.val_ids = get_database_split(self.database, self.cfg['database_split_type'])
        self.ref_ids = np.asarray(self.ref_ids)

        # build imgs_info
        self.ref_dist_idx = compute_nearest_camera_indices(self.database, self.ref_ids) # rfn,rfn
        ref_imgs_info = build_imgs_info(self.database, self.ref_ids, self.cfg['ref_pad_interval'], True, replace_none_depth=True)
        if self.cfg['use_consistent_depth_range']:
            ref_imgs_info['depth_range'][:, 0] = np.min(ref_imgs_info['depth_range'])
            ref_imgs_info['depth_range'][:, 1] = np.max(ref_imgs_info['depth_range'])
        self.ref_imgs_info = imgs_info_to_torch(ref_imgs_info)

        if self.cfg['use_validation']:
            self.val_dist_idx = compute_nearest_camera_indices(self.database, self.val_ids, self.ref_ids)
            val_imgs_info = build_imgs_info(self.database, self.val_ids, -1, True, has_depth=False)
            self.val_imgs_info = imgs_info_to_torch(val_imgs_info)
            self.val_num = len(self.val_ids)

        # init from generalization model
        self._initialization()

        # after initialization, we check the correctness of rendered images
        if self.cfg['use_validation'] and self.cfg['validate_initialization']:
            print('init validation rendering ...')
            Path(f'data/vis_val/{self.cfg["name"]}').mkdir(exist_ok=True, parents=True)
            self.eval()
            self.cuda()
            for vi in tqdm(range(self.val_num)):
                outputs = self.validate_step(vi)
                key_name = 'pixel_colors_nr_fine' if self.cfg['use_hierarchical_sampling'] else 'pixel_colors_nr'
                img_gt = self.val_imgs_info['imgs'][vi] # 3,h,w
                _, h, w = img_gt.shape
                img_gt = color_map_backward(img_gt.permute(1,2,0).numpy())
                rgb_pr = outputs[key_name].reshape(h, w, 3).cpu().numpy()
                img_pr = color_map_backward(rgb_pr)
                imsave(f'data/vis_val/{self.cfg["name"]}/init-{vi}.jpg',concat_images_list(img_gt,img_pr))

    def _init_by_depth(self, ref_id, init_net):
        # init by depth
        dist_idx = compute_nearest_camera_indices(self.database, [ref_id], self.ref_ids)[0]
        assert(self.ref_ids[dist_idx[0]] == ref_id) # view 0 is itself
        ref_imgs_info = imgs_info_slice(self.ref_imgs_info,torch.from_numpy(dist_idx[:self.cfg['init_view_num']]).long())

        with torch.no_grad():
            ray_feats_cur = init_net(to_cuda(ref_imgs_info), None, False)
            # _, r_dim, rh, rw = ray_feats_cur.shape
        ray_feats_cur = ray_feats_cur[0:1].detach().cpu()
        return ray_feats_cur

    def _init_by_cost_volume(self, ref_id, init_net):
        dist_idx = compute_nearest_camera_indices(self.database, [ref_id], self.ref_ids)[0]
        assert(self.ref_ids[dist_idx[0]] == ref_id) # view 0 is itself
        ref_imgs_info = imgs_info_slice(self.ref_imgs_info,torch.from_numpy(np.asarray([self.ref_ids.tolist().index(ref_id)])).long())
        src_num = self.cfg['init_src_view_num']
        src_imgs_info = imgs_info_slice(self.ref_imgs_info,torch.from_numpy(dist_idx[1:1+self.cfg['init_src_view_num']]).long())
        ref_imgs_info['nn_ids'] = torch.from_numpy(np.arange(src_num)).unsqueeze(0).long()

        with torch.no_grad():
            ray_feats_cur = init_net(to_cuda(ref_imgs_info), to_cuda(src_imgs_info), False)
            # _, r_dim, rh, rw = ray_feats_cur.shape
        ray_feats_cur = ray_feats_cur.detach().cpu()
        return ray_feats_cur

    def _init_raw_visibility_features(self, ref_id, init_net):
        if isinstance(init_net, DepthInitNet):
            ray_feats_cur = self._init_by_depth(ref_id, init_net)
        elif isinstance(init_net, CostVolumeInitNet):
            ray_feats_cur = self._init_by_cost_volume(ref_id, init_net)
        else:
            raise NotImplementedError
        return ray_feats_cur

    def _initialization(self):
        self.ray_feats = nn.ParameterList()
        if self.cfg['gen_cfg'] is not None:
            # load generalization model
            gen_cfg = load_cfg(self.cfg['gen_cfg'])
            name = gen_cfg['name']
            ckpt = torch.load(f'data/model/{name}/model_best.pth')
            gen_renderer = NeuralRayGenRenderer(gen_cfg).cuda()
            gen_renderer.load_state_dict(ckpt['network_state_dict'])
            gen_renderer = gen_renderer.eval()

            # init from generalization model
            print('initialization ...')
            for ref_id in tqdm(self.ref_ids):
                self.ray_feats.append(nn.Parameter(self._init_raw_visibility_features(ref_id, gen_renderer.init_net)))

            # init other parameters
            self.vis_encoder.load_state_dict(gen_renderer.vis_encoder.state_dict())
            self.dist_decoder.load_state_dict(gen_renderer.dist_decoder.state_dict())
            self.agg_net.load_state_dict(gen_renderer.agg_net.state_dict())
            self.sph_fitter.load_state_dict(gen_renderer.sph_fitter.state_dict())
            self.image_encoder.load_state_dict(gen_renderer.image_encoder.state_dict())
            if self.cfg['use_hierarchical_sampling']:
                self.fine_dist_decoder.load_state_dict(gen_renderer.fine_dist_decoder.state_dict())
                self.fine_agg_net.load_state_dict(gen_renderer.fine_agg_net.state_dict())
        else:
            print('init from scratch !')
            fh, fw = self.cfg['ray_feats_res']
            dim = self.cfg['ray_feats_dim']
            ref_num = len(self.ref_ids)
            for k in range(ref_num):
                self.ray_feats.append(nn.Parameter(torch.randn(1,dim,fh,fw)))

    def slice_imgs_info(self, ref_idx, val_idx, is_train):
        # prepare ref imgs info
        ref_imgs_info = imgs_info_slice(self.ref_imgs_info, torch.from_numpy(ref_idx).long())
        ref_imgs_info = to_cuda(ref_imgs_info)
        ref_imgs_info['ray_feats'] = torch.cat([self.ray_feats[ref_i] for ref_i in ref_idx], 0)

        # prepare que_imgs_info
        if is_train:
            que_imgs_info = imgs_info_slice(self.ref_imgs_info, torch.from_numpy(np.asarray([val_idx])).long())
            qn, _, hn, wn = que_imgs_info['imgs'].shape
            que_mask_cur = que_imgs_info['masks'][0, 0].cpu().numpy() > 0
            coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], self.cfg['foreground_ratio']).reshape([1, -1, 2])
        else:
            que_imgs_info = imgs_info_slice(self.val_imgs_info, torch.from_numpy(np.asarray([val_idx])).long())
            qn, _, hn, wn = que_imgs_info['imgs'].shape
            coords = np.stack(np.meshgrid(np.arange(wn), np.arange(hn)), -1)
            coords = coords.reshape([1, -1, 2]).astype(np.float32)

        que_imgs_info['coords'] = torch.from_numpy(coords)
        que_imgs_info = to_cuda(que_imgs_info)
        if is_train and self.cfg['use_self_hit_prob']:
            que_imgs_info['ray_feats'] = self.ray_feats[val_idx]
        return ref_imgs_info, que_imgs_info

    def validate_step(self, val_idx):
        ref_idx = self.val_dist_idx[val_idx][:self.cfg['neighbor_view_num']]
        ref_imgs_info, que_imgs_info = self.slice_imgs_info(ref_idx, val_idx, False)

        with torch.no_grad():
            render_outputs = self.render(que_imgs_info, ref_imgs_info, False)

        ref_imgs_info.pop('ray_feats')
        ref_imgs_info.pop('img_feats')
        render_outputs.update({'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info})
        return render_outputs

    def train_step(self):
        # select neighboring views for training
        que_i = np.random.randint(0,len(self.ref_ids))
        ref_idx = self.ref_dist_idx[que_i]
        if np.random.random() > self.cfg['include_self_prob']:
            ref_idx = ref_idx[1:]
        ref_idx = ref_idx[:self.cfg['neighbor_view_num']*self.cfg['neighbor_pool_ratio']]
        np.random.shuffle(ref_idx)
        ref_idx = ref_idx[:self.cfg['neighbor_view_num']]
        ref_imgs_info, que_imgs_info = self.slice_imgs_info(ref_idx, que_i, True)

        # render
        render_outputs = self.render(que_imgs_info.copy(), ref_imgs_info.copy(), True)

        # clear some values for outputs
        ref_imgs_info.pop('ray_feats')
        que_imgs_info.pop('ray_feats')
        if 'img_feats' in ref_imgs_info: ref_imgs_info.pop('img_feats')
        if 'img_feats' in que_imgs_info: que_imgs_info.pop('img_feats')
        render_outputs.update({'que_imgs_info': que_imgs_info})
        return render_outputs

    def render_pose(self, render_imgs_info, overlap_mode=False):
        # this function is used in rendering from arbitrary poses
        render_pose = render_imgs_info['poses'].cpu().numpy()
        ref_poses = self.ref_imgs_info['poses'].cpu().numpy()
        if overlap_mode:
            import ipdb; ipdb.set_trace()
            ref_Ks = self.ref_imgs_info['Ks'].cpu().numpy()
            rfn,_, h, w = self.ref_imgs_info.shape
            ref_size = (h, w)
            que_pose = render_imgs_info['poses'].cpu().numpy()[0]
            que_K = render_imgs_info['Ks'].cpu().numpy()[0]
            que_range = render_imgs_info['depth_range'].cpu().numpy()[0]
            que_size = render_imgs_info['shape']
            ref_idx = select_working_views_by_overlap(ref_poses, ref_Ks, ref_size, que_pose, que_K, que_size, que_range, self.cfg['neighbor_view_num'], 32)
        else:
            ref_idx = select_working_views(ref_poses, render_pose, self.cfg['neighbor_view_num'], True)[0]
        ref_imgs_info = to_cuda(imgs_info_slice(self.ref_imgs_info, torch.from_numpy(ref_idx).long()))
        ref_imgs_info['ray_feats'] = torch.cat([self.ray_feats[ref_i] for ref_i in ref_idx], 0)

        with torch.no_grad():
            render_outputs = self.render(render_imgs_info, ref_imgs_info, False)
        return render_outputs

    def forward(self, data):
        index = data['index']
        is_train = 'eval' not in data

        if is_train:
            return self.train_step()
        else:
            return self.validate_step(index)


name2network={
    'neuray_gen': NeuralRayGenRenderer,
    'neuray_ft': NeuralRayFtRenderer,
}
