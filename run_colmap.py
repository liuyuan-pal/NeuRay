import argparse
import numpy as np

from colmap_scripts.process import process_example_dataset, clear_project
from dataset.database import parse_database_name
from utils.base_utils import pose_inverse
from utils.draw_utils import output_points

parser = argparse.ArgumentParser()
parser.add_argument('--example_name', type=str, default='ear_cup')
parser.add_argument('--same_camera', action='store_true', dest='same_camera', default=False)
parser.add_argument('--colmap_path', type=str, default='colmap')
flags = parser.parse_args()

def visualize_camera_locations(example_name):
    database = parse_database_name(f'example/{example_name}/raw')
    img_ids = database.get_img_ids()
    cam_pts = []
    for k, img_id in enumerate(img_ids):
        pose = database.get_pose(img_id)
        cam_pt = pose_inverse(pose)[:, 3]
        cam_pts.append(cam_pt)

    output_points(f'data/example/{example_name}/cam_pts.txt', np.stack(cam_pts, 0))

process_example_dataset(flags.example_name,flags.same_camera,flags.colmap_path)
visualize_camera_locations(flags.example_name)
clear_project(flags.example_name)
