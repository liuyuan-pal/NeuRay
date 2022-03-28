import numpy as np

from dataset.database import BaseDatabase

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

def select_working_views_db(database: BaseDatabase, ref_ids, que_poses, work_num, exclude_self=False):
    ref_ids = database.get_img_ids() if ref_ids is None else ref_ids
    ref_poses = [database.get_pose(img_id) for img_id in ref_ids]

    ref_ids = np.asarray(ref_ids)
    ref_poses = np.asarray(ref_poses)
    indices = select_working_views(ref_poses, que_poses, work_num, exclude_self)
    return ref_ids[indices] # qn,wn