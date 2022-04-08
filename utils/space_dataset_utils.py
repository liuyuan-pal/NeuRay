import json
import os

import numpy as np
import math

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)

def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    _EPS = np.finfo(float).eps * 4.0
    quaternion = np.zeros((4, ), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion

class Camera(object):
    """Represents a Camera with intrinsics and world from/to camera transforms.
    Attributes:
      w_f_c: The world from camera 4x4 matrix.
      c_f_w: The camera from world 4x4 matrix.
      intrinsics: The camera intrinsics as a 3x3 matrix.
      inv_intrinsics: The inverse of camera intrinsics matrix.
    """

    def __init__(self, intrinsics, w_f_c):
        """Constructor.
        Args:
          intrinsics: A numpy 3x3 array representing intrinsics.
          w_f_c: A numpy 4x4 array representing wFc.
        """
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        self.w_f_c = w_f_c
        self.c_f_w = np.linalg.inv(w_f_c)


class View(object):
    """Represents an image and associated camera geometry.
    Attributes:
      camera: The camera for this view.
      image: The np array containing the image data.
      image_path: The file path to the image.
      shape: The 2D shape of the image.
    """

    def __init__(self, image_path, shape, camera):
        self.image_path = image_path
        self.shape = shape
        self.camera = camera
        self.image = None


def _WorldFromCameraFromViewDict(view_json):
    """Fills the world from camera transform from the view_json.
    Args:
        view_json: A dictionary of view parameters.
    Returns:
        A 4x4 transform matrix representing the world from camera transform.
    """
    transform = np.identity(4)
    position = view_json['position']
    transform[0:3, 3] = (position[0], position[1], position[2])
    orientation = view_json['orientation']
    angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
    angle = np.linalg.norm(angle_axis)
    epsilon = 1e-7
    if abs(angle) < epsilon:
        # No rotation
        return transform

    axis = angle_axis / angle
    rot_mat = quaternion_matrix(quaternion_about_axis(-angle, axis))
    transform[0:3, 0:3] = rot_mat[0:3, 0:3]
    return transform


def _IntrinsicsFromViewDict(view_params):
    """Fills the intrinsics matrix from view_params.
    Args:
        view_params: Dict view parameters.
    Returns:
        A 3x3 matrix representing the camera intrinsics.
    """
    intrinsics = np.identity(3)
    intrinsics[0, 0] = view_params['focal_length']
    intrinsics[1, 1] = (view_params['focal_length'] * view_params['pixel_aspect_ratio'])
    intrinsics[0, 2] = view_params['principal_point'][0]
    intrinsics[1, 2] = view_params['principal_point'][1]
    return intrinsics


def ReadView(base_dir, view_json):
    return View(
        image_path=os.path.join(base_dir, view_json['relative_path']),
        shape=(int(view_json['height']), int(view_json['width'])),
        camera=Camera(
            _IntrinsicsFromViewDict(view_json),
            _WorldFromCameraFromViewDict(view_json)))


def ReadScene(base_dir):
    """Reads a scene from the directory base_dir."""
    with open(os.path.join(base_dir, 'models.json')) as f:
        model_json = json.load(f)

    all_views = []
    for views in model_json:
        all_views.append([ReadView(base_dir, view_json) for view_json in views])
    return all_views
