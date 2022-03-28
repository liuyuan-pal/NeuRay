import torch.nn as nn
import torch

from network.ops import AddBias

def get_near_far_points(depth, interval, depth_range, is_ref, fixed_interval=False, fixed_interval_val=0.01):
    """                               is_ref     |  not is_ref
    :param depth:    [...,dn]      rfn,qn,rn,dn or qn,rn,dn
    :param interval: [...,dn]        1,qn,rn,dn or qn,rn,dn
    :param depth_range:                   rfn,2 or qn,2
    :param is_ref:
    :param fixed_interval:
    :param fixed_interval_val:
    :return: near far [rfn,qn,rn,dn] or [qn,rn,dn]
    """
    if is_ref:
        ref_near = depth_range[:, 0]
        ref_far = depth_range[:, 1]
        ref_near = -1 / ref_near[:, None, None, None]
        ref_far = -1 / ref_far[:, None, None, None]
        depth = torch.clamp(depth, min=1e-5)
        depth = -1 / depth
        depth = (depth - ref_near) / (ref_far - ref_near)
    else:
        que_near = depth_range[:, 0]  # qn
        que_far = depth_range[:, 1]  # qn
        que_near = -1 / que_near[:, None, None]
        que_far = -1 / que_far[:, None, None]
        depth = torch.clamp(depth, min=1e-5)
        depth = -1 / depth
        depth = (depth - que_near) / (que_far - que_near)

    if not fixed_interval:
        if is_ref:
            interval_half = interval / 2
            interval_ext = torch.cat([interval_half[..., 0:1], interval_half], -1)
            near = depth - interval_ext[..., :-1]
            far = depth + interval_ext[..., 1:]
        else:
            interval_half = interval / 2
            first = depth[..., 0] - interval_half[..., 0]
            last = depth[..., -1] + interval_half[..., -1]
            depth_ext = (depth[..., :-1] + depth[..., 1:]) / 2
            depth_ext = torch.cat([first[..., None], depth_ext, last[..., None]], -1)
            near = depth_ext[..., :-1]
            far = depth_ext[..., 1:]
    else:
        near = depth - fixed_interval_val/2
        far = depth + fixed_interval_val/2

    return near, far

class MixtureLogisticsDistDecoder(nn.Module):
    default_cfg={
        'feats_dim': 32,
        'bias_val': 0.05,
        "use_vis": True,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}
        ray_feats_dim = self.cfg["feats_dim"]
        run_dim = ray_feats_dim
        self.mean_decoder=nn.Sequential(
            nn.Linear(ray_feats_dim, run_dim),
            nn.ELU(),
            nn.Linear(run_dim, run_dim),
            nn.ELU(),
            nn.Linear(run_dim, 2),
            nn.Softplus()
        )
        self.var_decoder=nn.Sequential(
            nn.Linear(ray_feats_dim, run_dim),
            nn.ELU(),
            nn.Linear(run_dim, run_dim),
            nn.ELU(),
            nn.Linear(run_dim, 2),
            nn.Softplus(),
            AddBias(self.cfg['bias_val']),
        )
        self.aw_decoder=nn.Sequential(
            nn.Linear(ray_feats_dim, run_dim),
            nn.ELU(),
            nn.Linear(run_dim, run_dim),
            nn.ELU(),
            nn.Linear(run_dim, 1),
            nn.Sigmoid(),
        )
        if self.cfg['use_vis']:
            self.vis_decoder=nn.Sequential(
                nn.Linear(ray_feats_dim, run_dim),
                nn.ELU(),
                nn.Linear(run_dim, run_dim),
                nn.ELU(),
                nn.Linear(run_dim, 1),
                nn.Sigmoid(),
            )

    def forward(self, feats):
        prj_mean = self.mean_decoder(feats)
        prj_var = self.var_decoder(feats)
        prj_aw = self.aw_decoder(feats)
        if self.cfg['use_vis']:
            prj_vis = self.vis_decoder(feats)
        else:
            prj_vis = None
        return prj_mean, prj_var, prj_vis, prj_aw

    def compute_prob(self, depth, interval, mean, var, vis, aw, is_ref, depth_range):
        """
        :param depth:    [...,dn]      rfn,qn,rn,dn   or qn,rn,dn
        :param interval: [...,dn]        1,qn,rn,dn   or qn,rn,dn
        :param mean:     [...,1 or dn] rfn,qn,rn,dn,2 or qn,rn,1,2
        :param var:      [...,1 or dn] rfn,qn,rn,dn,2 or qn,rn,1,2
        :param vis:      [...,1 or dn] rfn,qn,rn,dn,1 or qn,rn,1,1
        :param aw:       [...,1 or dn] rfn,qn,rn,dn,1 or qn,rn,1,1
        :param is_ref:
        :param depth_range: rfn,2 or qn,2
        :return:
        """
        near, far = get_near_far_points(depth, interval, depth_range, is_ref)

        # near and far [rfn,qn,rn,dn] or [qn,rn,dn]
        mix = torch.cat([aw, 1 - aw],-1) # [...,2]
        near, far = near[...,None], far[...,None]

        d0 = (near - mean) * var # [...,2]
        d1 = (far - mean) * var  # [...,2]
        cdf0 = (0.5 + 0.5 * torch.tanh(d0))
        cdf1 = (0.5 + 0.5 * torch.tanh(d1))
        if self.cfg['use_vis']:
            cdf0, cdf1 = cdf0 * vis, cdf1 * vis
        visibility = 1 - cdf0
        hit_prob = cdf1 - cdf0
        visibility = torch.sum(visibility*mix, -1)
        hit_prob = torch.sum(hit_prob*mix, -1)

        eps = 1e-5
        alpha_value = torch.log(hit_prob / (visibility - hit_prob + eps) + eps)
        return alpha_value, visibility, hit_prob

    def decode_alpha_value(self, alpha_value):
        alpha_value = torch.sigmoid(alpha_value)
        return alpha_value

    def predict_mean(self,prj_ray_feats):
        prj_mean = self.mean_decoder(prj_ray_feats)
        return prj_mean

    def predict_aw(self,prj_ray_feats):
        return self.aw_decoder(prj_ray_feats)


name2dist_decoder={
    'mixture_logistics': MixtureLogisticsDistDecoder
}