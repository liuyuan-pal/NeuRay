import numpy as np

from dataset.database import BaseDatabase
# from utils.base_utils import pose_inverse, project_points


def compute_nearest_camera_indices(database, que_ids, ref_ids=None):
    if ref_ids is None: ref_ids = que_ids
    ref_poses = [database.get_pose(ref_id) for ref_id in ref_ids]
    ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
    que_poses = [database.get_pose(que_id) for que_id in que_ids]
    que_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])

    dists = np.linalg.norm(ref_cam_pts[None, :, :] - que_cam_pts[:, None, :], 2, 2)
    dists_idx = np.argsort(dists, 1)
    return dists_idx

def select_working_views(ref_poses, que_poses, work_num, exclude_self=False):
    ref_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in ref_poses])
    render_cam_pts = np.asarray([-pose[:, :3].T @ pose[:, 3] for pose in que_poses])
    dists = np.linalg.norm(ref_cam_pts[None, :, :] - render_cam_pts[:, None, :], 2, 2) # qn,rfn
    ids = np.argsort(dists)
    if exclude_self:
        ids = ids[:, 1:work_num+1]
    else:
        ids = ids[:, :work_num]
    return ids

# def select_working_views_by_overlap(ref_poses, ref_Ks, ref_size, que_pose, que_K, que_size, que_depth_ranges, work_num, plane_num=8):
#     near, far = que_depth_ranges[0], que_depth_ranges[1]
#     depth_vals = np.linspace(near, far, plane_num) # dn
#     depth_vals = depth_vals[None,None,:,None] # 1,1,dn,1
#     qh, qw = que_size
#     dn = plane_num
#     num = 32
#     coords2d = np.stack(np.meshgrid(np.linspace(0,qw-1,num),np.linspace(0,qh-1,num)),-1)[:,:,None,:] # qh,qw,1,2
#     pts = np.concatenate([np.tile(depth_vals,[num,num,1,1]), np.tile(coords2d, [1,1,dn,1])],-1) # qh,qw,dn,3
#     pts = pts.reshape([num*num*dn, 3])
#     pts[:,:2] *= pts[:,2:]
#
#     que_pose_inv = pose_inverse(que_pose) # 3,4
#     que_K_inv = np.linalg.inv(que_K)
#     RK = que_pose_inv[:,:3] @ que_K_inv
#     t= que_pose_inv[:,3:]
#     pts = pts @ RK.T + t.T # in world coordinate [pn,3]
#
#     rfn = ref_poses.shape[0]
#     ref_h, ref_w = ref_size
#
#     def get_valid_mask(pts2d, depth, h, w):
#         valid_mask = (pts2d[:, 0] < w) & (pts2d[:, 1] < h) & (pts2d[:, 0] >= 0) & (pts2d[:, 1] >= 0) & (depth > 0)
#         return valid_mask
#
#     global_visibility=[np.mean(get_valid_mask(*project_points(pts, ref_poses[rfi], ref_Ks[rfi]), ref_h, ref_w)) for rfi in range(rfn)]
#
#     # all points are invisible
#     cur_pts = pts
#     invisible_mask = np.ones(cur_pts.shape[0],dtype=np.bool)
#
#     cur_ref_ids = [rfi for rfi in range(rfn)]
#     resulted_ref_ids = []
#     for wi in range(min(work_num, rfn)):
#         cur_pts = cur_pts[invisible_mask]
#         if cur_pts.shape[0]/pts.shape[0]>=0.02:
#             # select by cur visibility
#             cur_visibility = [np.mean(get_valid_mask(*project_points(cur_pts, ref_poses[rfi], ref_Ks[rfi]), ref_h, ref_w)) for rfi in cur_ref_ids]
#         else:
#             # select by global visibility
#             cur_visibility = [global_visibility[rfi] for rfi in cur_ref_ids]
#
#         max_ref_index = np.argmax(np.asarray(cur_visibility))
#         max_ref_id = cur_ref_ids[max_ref_index]
#         resulted_ref_ids.append(max_ref_id)
#         cur_ref_ids.remove(max_ref_id)
#
#         # update invisible mask
#         invisible_mask = ~get_valid_mask(*project_points(cur_pts, ref_poses[max_ref_id], ref_Ks[max_ref_id]), ref_h, ref_w)
#     return resulted_ref_ids

def select_working_views_db(database: BaseDatabase, ref_ids, que_poses, work_num, exclude_self=False):
    ref_ids = database.get_img_ids() if ref_ids is None else ref_ids
    ref_poses = [database.get_pose(img_id) for img_id in ref_ids]

    ref_ids = np.asarray(ref_ids)
    ref_poses = np.asarray(ref_poses)
    indices = select_working_views(ref_poses, que_poses, work_num, exclude_self)
    return ref_ids[indices] # qn,wn