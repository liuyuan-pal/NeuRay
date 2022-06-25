import argparse
import os
from pathlib import Path

import numpy as np
import torch
from skimage.io import imsave
from tqdm import tqdm

from dataset.database import parse_database_name, get_database_split, ExampleDatabase
from dataset.train_dataset import build_src_imgs_info_select
from network.renderer import name2network
from utils.base_utils import load_cfg, to_cuda, color_map_backward, make_dir
from utils.imgs_info import build_imgs_info, build_render_imgs_info, imgs_info_to_torch, imgs_info_slice
from utils.render_poses import get_render_poses
from utils.view_select import select_working_views_db

def prepare_render_info(database, pose_type, pose_fn, use_depth):
    # interpolate poses
    if pose_type.startswith('eval'):
        split_name = 'test' if use_depth else 'test_all'
        ref_ids, render_ids = get_database_split(database, split_name)
        que_Ks = np.asarray([database.get_K(render_id) for render_id in render_ids],np.float32)
        que_poses = np.asarray([database.get_pose(render_id) for render_id in render_ids],np.float32)
        que_shapes = np.asarray([database.get_image(render_id).shape[:2] for render_id in render_ids],np.int64)
        que_depth_ranges = np.asarray([database.get_depth_range(render_id) for render_id in render_ids],np.float32)
    else:
        que_poses = get_render_poses(database, pose_type, pose_fn)

        # prepare intrinsics, shape, depth range
        que_Ks = np.array([database.get_K(database.get_img_ids()[0]) for _ in range(que_poses.shape[0])],np.float32)
        h, w, _ = database.get_image(database.get_img_ids()[0]).shape
        que_shapes = np.array([(h,w) for _ in range(que_poses.shape[0])])

        if isinstance(database,ExampleDatabase):
            # we have sparse points to compute depth range
            que_depth_ranges = np.stack([database.compute_depth_range_impl(pose) for pose in que_poses],0)
        else:
            # just use depth range of all images
            ref_depth_range_list = np.asarray([database.get_depth_range(img_id) for img_id in database.get_img_ids()])
            near = np.min(ref_depth_range_list[:,0])
            far = np.max(ref_depth_range_list[:,1])
            que_depth_ranges = np.asarray([(near,far) for _ in range(que_poses.shape[0])],np.float32)

        ref_ids = database.get_img_ids()
        render_ids = None
    return que_poses, que_Ks, que_shapes, que_depth_ranges, ref_ids, render_ids

def save_renderings(output_dir, qi, render_info, h, w):
    def output_image(suffix):
        if f'pixel_colors_{suffix}' in render_info:
            render_image = color_map_backward(render_info[f'pixel_colors_{suffix}'].cpu().numpy().reshape([h, w, 3]))
            imsave(f'{output_dir}/{qi}-{suffix}.jpg', render_image)

    output_image('nr')
    output_image('nr_fine')

def save_depth(output_dir, qi, render_info, h, w, depth_range):
    suffix='fine'
    if f'render_depth_{suffix}' in render_info:
        depth = render_info[f'render_depth_{suffix}'].cpu().numpy().reshape([h, w])
        near, far = depth_range
        depth = np.clip(depth, a_min=near, a_max=far)
        depth = (1/depth - 1/near)/(1/far - 1/near)
        depth = color_map_backward(depth)
        imsave(f'{output_dir}/{qi}-{suffix}-depth.png', depth)

def render_video_gen(database_name: str,
                     cfg_fn='configs/gen_lr_neuray.yaml',
                     pose_type='inter', pose_fn=None,
                     render_depth=False,
                     ray_num=8192, rb=0, re=-1):
    default_render_cfg = {
        'min_wn': 8, # working view number
        'ref_pad_interval': 16, # input image size should be multiple of 16
        'use_src_imgs': False, # use source images to construct cost volume or not
        'cost_volume_nn_num': 3, # number of source views used in cost volume
        'use_depth': True, # use colmap depth in rendering or not
    }

    # load render cfg
    cfg = load_cfg(cfg_fn)
    cfg['ray_batch_num'] = ray_num
    render_cfg = cfg['train_dataset_cfg'] if 'train_dataset_cfg' in cfg else {}

    render_cfg = {**default_render_cfg, **render_cfg}

    cfg['render_depth'] = render_depth
    # load model
    renderer = name2network[cfg['network']](cfg)
    ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
    renderer.load_state_dict(ckpt['network_state_dict'])
    renderer.cuda()
    renderer.eval()
    step = ckpt["step"]

    # render poses
    database = parse_database_name(database_name)
    que_poses, que_Ks, que_shapes, que_depth_ranges, ref_ids_all, render_ids = \
        prepare_render_info(database, pose_type, pose_fn, render_cfg['use_depth'])

    # select working views
    # overlap_select = False
    # if overlap_select:
    #     ref_ids_list = []
    #     ref_size = database.get_image(ref_ids_all[0]).shape[:2]
    #     ref_poses = np.stack([database.get_pose(ref_id) for ref_id in ref_ids_all], 0)
    #     ref_Ks = np.stack([database.get_K(ref_id) for ref_id in ref_ids_all], 0)
    #     for que_pose, que_K, que_shape, que_depth_range in zip(que_poses, que_Ks, que_shapes, que_depth_ranges):
    #         ref_indices = select_working_views_by_overlap(ref_poses, ref_Ks, ref_size, que_pose, que_K, que_shape, que_depth_range, render_cfg['min_wn'])
    #         ref_ids_list.append(np.asarray(ref_ids_all)[ref_indices])
    # else:
    ref_ids_list = select_working_views_db(database, ref_ids_all, que_poses, render_cfg['min_wn'])
    output_dir = f'data/render/{database.database_name}/{cfg["name"]}-{step}-{pose_type}'
    # if overlap_select: output_dir+='-overlap'
    make_dir(output_dir)

    # render
    num = que_poses.shape[0]
    re = num if re==-1 else re
    for qi in tqdm(range(rb,re)):
        if os.path.exists(f'{output_dir}/{qi}-nr_fine.jpg'): continue

        que_imgs_info = build_render_imgs_info(que_poses[qi], que_Ks[qi], que_shapes[qi], que_depth_ranges[qi])
        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        data = {'que_imgs_info': que_imgs_info, 'eval': True}

        ref_ids = ref_ids_list[qi]
        if render_cfg['use_src_imgs']:
            ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(
                database, ref_ids, ref_ids_all, render_cfg["cost_volume_nn_num"], render_cfg["ref_pad_interval"])
            src_imgs_info = ref_imgs_info.copy()
            data['src_imgs_info'] = to_cuda(imgs_info_to_torch(src_imgs_info))

            ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)
            ref_imgs_info['nn_ids'] = ref_cv_idx
        else:
            ref_imgs_info = build_imgs_info(database, ref_ids, render_cfg["ref_pad_interval"])

        ref_imgs_info = to_cuda(imgs_info_to_torch(ref_imgs_info))
        data['ref_imgs_info']=ref_imgs_info

        with torch.no_grad():
            render_info = renderer(data)
        h, w = que_shapes[qi]
        save_renderings(output_dir, qi, render_info, h, w)
        if render_depth:
            save_depth(output_dir, qi, render_info, h, w, que_depth_ranges[qi])
        if pose_type=='eval':
            gt_dir = f'data/render/{database_name}/gt'
            Path(gt_dir).mkdir(exist_ok=True, parents=True)
            if not (Path(gt_dir)/f'{qi}.jpg').exists():
                imsave(f'{gt_dir}/{qi}.jpg',database.get_image(render_ids[qi]))

def render_video_ft(database_name, cfg_fn, pose_type, pose_fn, render_depth=False, ray_num=4096, rb=0, re=-1):
    # init network
    # default_cfg={}
    # cfg = {**default_cfg, **load_cfg(cfg_fn)}
    cfg = load_cfg(cfg_fn)
    cfg['gen_cfg'] = None
    cfg['validate_initialization'] = False
    cfg['ray_batch_num'] = ray_num
    cfg['render_depth'] = render_depth
    ckpt = torch.load(f'data/model/{cfg["name"]}/model_best.pth')
    _, dim, h, w = ckpt['network_state_dict']['ray_feats.0'].shape
    cfg['ray_feats_res'] = [h,w]
    cfg['ray_feats_dim'] = dim
    renderer = name2network[cfg['network']](cfg)
    renderer.load_state_dict(ckpt['network_state_dict'])
    step=ckpt['step']
    renderer.cuda()
    renderer.eval()

    database = parse_database_name(database_name)
    que_poses, que_Ks, que_shapes, que_depth_ranges, ref_ids, render_ids = \
        prepare_render_info(database, pose_type, pose_fn, False)
    assert(database.database_name == renderer.database.database_name)

    output_dir = f'data/render/{database.database_name}/{cfg["name"]}-{step}-{pose_type}'
    Path(output_dir).mkdir(parents=True,exist_ok=True)

    # render
    num = que_poses.shape[0]
    re = num if re==-1 else re
    for qi in tqdm(range(rb,re)):
        if os.path.exists(f'{output_dir}/{qi}.jpg'): continue
        que_imgs_info = build_render_imgs_info(que_poses[qi], que_Ks[qi], que_shapes[qi], que_depth_ranges[qi])
        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        with torch.no_grad():
            render_info = renderer.render_pose(que_imgs_info)
        h, w = que_shapes[qi]
        save_renderings(output_dir, qi, render_info, h, w)
        if render_depth:
            save_depth(output_dir, qi, render_info, h, w, que_depth_ranges[qi])

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_name', type=str, default='llff_colmap/fern/high', help='<dataset_name>/<scene_name>/<scene_setting>')
    parser.add_argument('--cfg', type=str, default='configs/gen_lr_neuray.yaml', help='config path of the renderer')
    parser.add_argument('--pose_type', type=str, default='eval', help='type of render poses')
    parser.add_argument('--pose_fn', type=str, default=None, help='file to render poses')
    parser.add_argument('--rb', type=int, default=0, help='begin index of rendering poses')
    parser.add_argument('--re', type=int, default=-1, help='end index of rendering poses')
    parser.add_argument('--render_type', type=str, default='gen', help='gen:generalization or ft:finetuning')
    parser.add_argument('--ray_num', type=int, default=4096, help='number of rays in one rendering batch')
    parser.add_argument('--depth', action='store_true', dest='depth', default=False)
    # parser.add_argument('--overlap', action='store_true', dest='overlap', default=False)
    flags = parser.parse_args()
    if flags.render_type=='gen':
        render_video_gen(flags.database_name, cfg_fn=flags.cfg, pose_type=flags.pose_type, pose_fn=flags.pose_fn,
                         render_depth=flags.depth, ray_num=flags.ray_num, rb=flags.rb,re=flags.re)
    else:
        render_video_ft(flags.database_name, cfg_fn=flags.cfg, pose_type=flags.pose_type, pose_fn=flags.pose_fn,
                        render_depth=flags.depth, ray_num=flags.ray_num, rb=flags.rb, re=flags.re)