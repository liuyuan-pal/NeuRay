import abc
import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.io import imread, imsave

from asset import LLFF_ROOT, nerf_syn_val_ids, NERF_SYN_ROOT
from colmap.read_write_dense import read_array
from colmap.read_write_model import read_cameras_binary, read_images_binary, read_points3d_binary
from utils.base_utils import downsample_gaussian_blur, color_map_backward, resize_img, read_pickle, project_points, \
    save_pickle, transform_points_Rt, pose_inverse
from PIL import Image

from utils.llff_utils import load_llff_data
from utils.real_estate_utils import parse_pose_file, unnormalize_intrinsics
from utils.space_dataset_utils import ReadScene


class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self,check_depth_exist=False):
        pass

    @abc.abstractmethod
    def get_bbox(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth(self,img_id):
        pass

    @abc.abstractmethod
    def get_mask(self,img_id):
        pass

    @abc.abstractmethod
    def get_depth_range(self,img_id):
        pass

class LLFFColmapDatabase(BaseDatabase):
    def __init__(self, database_name):
        _, self.model_name, self.res_type = database_name.split('/')
        super().__init__(database_name)

        self.scale_factor = 1.0
        self.root_dir = f'{LLFF_ROOT}/{self.model_name}'
        self.cameras_colmap = read_cameras_binary(f'{self.root_dir}/sparse/cameras.bin')
        self.images_colmap = read_images_binary(f'{self.root_dir}/sparse/images.bin')

        self.img_ids = [str(k+1) for k in range(len(self.images_colmap))]
        self._cache_resolution()
        self._compute_depth_range()

    def get_resolution(self):
        if self.res_type == 'high':
            return 756, 1008
        elif self.res_type == 'low':
            return 756//2, 1008//2
        else:
            raise NotImplementedError

    def _cache_resolution(self):
        self.image_dir = f'{self.root_dir}/cache/{self.res_type}'
        Path(self.image_dir).mkdir(exist_ok=True, parents=True)
        h, w = self.get_resolution()
        ratio = w / 4032
        for img_id in self.img_ids:
            fn = self.images_colmap[int(img_id)].name
            if os.path.exists(f'{self.image_dir}/{fn}'):
                continue
            img = imread(f'{self.root_dir}/images/{fn}')
            img = downsample_gaussian_blur(img, ratio)
            img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
            imsave(f'{self.image_dir}/{fn}', img)

    def _compute_depth_range(self):
        self.bounds = np.load(f'{self.root_dir}/depth_range.npy')

    def get_img_ids(self, check_depth_exist=False):
        return self.img_ids

    def get_image(self, img_id):
        fn = self.images_colmap[int(img_id)].name
        return imread(f'{self.image_dir}/{fn}')

    def get_K(self, img_id):
        camera_colmap = self.cameras_colmap[self.images_colmap[int(img_id)].camera_id]
        height = camera_colmap.height
        width = camera_colmap.width
        h, w = self.get_resolution()
        fx, fy, cx, cy = camera_colmap.params
        K = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
        cx-=0.5; cy-=0.5
        K = np.diag([w/width, h/height, 1]) @ K
        return K.astype(np.float32)

    def get_pose(self, img_id):
        img_info = self.images_colmap[int(img_id)]
        R = img_info.qvec2rotmat()
        t = img_info.tvec
        pose = np.concatenate([R, t[:, None]], 1)
        return pose

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        return read_array(f'{self.root_dir}/colmap_depth/{img_id}.jpg.geometric.bin')

    def get_mask(self, img_id):
        h, w = self.get_resolution()
        return np.ones([h, w],dtype=np.bool_)

    def get_depth_range(self,img_id):
        return self.bounds[int(img_id)-1]

class DTUTestDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name, background_size = database_name.split('/')
        root_dir = f'data/dtu_test/{model_name}'
        self.root_dir = root_dir
        background, image_size = background_size.split('_')
        image_size = int(image_size)
        self.model_name = model_name
        self.image_size = image_size
        self.background = background
        self.ratio = image_size / 1600
        self.h, self.w = int(self.ratio*1200), int(image_size)

        self._coord_trans_world = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],dtype=np.float32,)

        rgb_paths = [x for x in glob.glob(os.path.join(root_dir, "image", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        rgb_paths = sorted(rgb_paths)

        self.depth_range=np.load(f'{self.root_dir}/depth_range.npy')

        all_cam = np.load(os.path.join(root_dir, "cameras.npz"))
        self.Rts=[]
        self.Ks=[]
        self.img_ids=[]
        for i, rgb_path in enumerate(rgb_paths):
            P = all_cam["world_mat_" + str(i)]
            P = P[:3]
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]

            scale_mtx = all_cam.get("scale_mat_" + str(i))
            if scale_mtx is not None:
                norm_trans = scale_mtx[:3, 3:]
                norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                pose[:3, 3:] -= norm_trans
                pose[:3, 3:] /= norm_scale
            pose = self._coord_trans_world @ pose # @ self._coord_trans_cam
            pose = pose[:3]
            pose = np.concatenate([pose[:,:3].T,- pose[:,:3].T @ pose[:,3:]],1)

            self.Ks.append(np.diag([self.ratio,self.ratio,1]) @ K)
            self.Rts.append(pose[:3])
            self.img_ids.append(f'{i}')


        self.img_id2imgs={}
        self.img_id2depth={}
        self.img_id2mask={}
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def get_image(self, img_id):
        if img_id in self.img_id2imgs:
            return self.img_id2imgs[img_id]
        img = imread(os.path.join(self.root_dir,'image',f'{int(img_id):06}.png'))
        if self.w != 1600:
            img = cv2.resize(downsample_gaussian_blur(img, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        if self.background=='black':
            mask = self.get_mask(img_id)
            img = img * mask.astype(np.uint8)[:,:,None]
        else:
            raise NotImplementedError
        self.img_id2imgs[img_id]=img
        return img

    def get_K(self, img_id):
        return self.Ks[int(img_id)].copy()

    def get_pose(self, img_id):
        return self.Rts[int(img_id)].copy()

    def get_img_ids(self,check_depth_exist=False):
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self,img_id):
        fn = f'{self.root_dir}/depth_maps/{img_id}.jpg.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        if img_id in self.img_id2depth:
            return self.img_id2depth[img_id]

        fn = f'{self.root_dir}/colmap_depth/{img_id}.jpg.geometric.bin'
        if os.path.exists(fn):
            depth = read_array(fn)
            depth=np.ascontiguousarray(depth, dtype=np.float32)
            if self.w != 800: depth = cv2.resize(depth, (self.w,self.h), interpolation=cv2.INTER_NEAREST)
            depth[~self.get_mask(img_id)] = 0
            self.img_id2depth[img_id] = depth
            return depth
        else:
            raise NotImplementedError

    def get_mask(self, img_id):
        if img_id in self.img_id2mask:
            return self.img_id2mask[img_id]
        mask = np.sum(imread(os.path.join(self.root_dir, 'mask', f'{int(img_id):03}.png')),-1)>0
        if self.w!=1600:
            mask = cv2.resize(mask.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0
        self.img_id2mask[img_id]=mask
        return mask

    def get_depth_range(self,img_id):
        return self.depth_range.copy()

class NeRFSyntheticDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name, background_size = database_name.split('/')
        background, size = background_size.split('_')
        self.model_name = model_name
        self.img_size = int(size)
        self.root_dir = f'{NERF_SYN_ROOT}/{model_name}'

        train_img_ids,train_poses,K = self.parse_info('train')
        test_img_ids,test_poses,K = self.parse_info('test')
        val_img_ids,val_poses,K = self.parse_info('val')

        self.K=K
        self.img_ids=train_img_ids+val_img_ids+test_img_ids
        self.poses=train_poses+val_poses+test_poses
        self.background=background
        self.range_dict={img_id:np.asarray((2.0,6.0),np.float32) for img_id in self.img_ids}
        ratio = int(size) / 800
        self.K = np.diag([ratio,ratio,1.0]).astype(np.float32) @ K
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def parse_info(self,split='train'):
        with open(f'{self.root_dir}/transforms_{split}.json','r') as f:
            img_info=json.load(f)
            focal=float(img_info['camera_angle_x'])
            img_ids,poses=[],[]
            for frame in img_info['frames']:
                img_ids.append('-'.join(frame['file_path'].split('/')[1:]))
                pose=np.asarray(frame['transform_matrix'], np.float32)
                R = pose[:3,:3].T
                t = -R @ pose[:3,3:]
                R = np.diag(np.asarray([1,-1,-1])) @ R
                t = np.diag(np.asarray([1,-1,-1])) @ t
                poses.append(np.concatenate([R,t],1))

            h,w,_=imread(f'{self.root_dir}/{self.img_id2img_path(img_ids[0])}.png').shape
            focal = .5 * w / np.tan(.5 * focal)
            K=np.asarray([[focal,0,w/2],[0,focal,h/2],[0,0,1]],np.float32)
        return img_ids, poses, K

    @staticmethod
    def img_id2img_path(img_id):
        return '/'.join(img_id.split('-'))

    def get_image(self, img_id):
        img = imread(f'{self.root_dir}/{self.img_id2img_path(img_id)}.png')
        alpha = img[:,:,3:].astype(np.float32)/255.0
        img = img[:,:,:3]
        if self.background=='black':
            img = img.astype(np.float32)/255.0
            img = img * alpha
            img = color_map_backward(img)
        elif self.background=='white':
            img = img.astype(np.float32)/255.0
            img = img*alpha + 1.0-alpha
            img = color_map_backward(img)
        else:
            raise NotImplementedError
        if self.img_size!=800:
            ratio = self.img_size/800
            img = resize_img(img, ratio)
        return img

    def get_K(self, img_id):
        return self.K.astype(np.float32).copy()

    def get_pose(self, img_id):
        return self.poses[self.img_ids.index(img_id)].astype(np.float32).copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist: return self.depth_img_ids
        return self.img_ids

    def get_bbox(self, img_id):
        alpha=imread(f'{self.root_dir}/{self.img_id2img_path(img_id)}.png')[:,:,3]
        ys,xs=np.nonzero(alpha>0)
        x_min,x_max=np.min(xs,0),np.max(xs,0)
        y_min,y_max=np.min(ys,0),np.max(ys,0)
        return [x_min,y_min,x_max-x_min+1,y_max-y_min+1]

    def _depth_existence(self,img_id):
        fn=f'{self.root_dir}/colmap_depth/{img_id}.png.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        fn=f'{self.root_dir}/colmap_depth/{img_id}.png.geometric.bin'
        if os.path.exists(fn):
            depth = read_array(fn)
            if self.img_size!=800:
                depth = cv2.resize(depth, (self.img_size,self.img_size), interpolation=cv2.INTER_NEAREST)
            return depth
        else:
            return None

    def get_mask(self, img_id):
        alpha=imread(f'{self.root_dir}/{self.img_id2img_path(img_id)}.png')[:,:,3]
        if self.img_size!=800:
            alpha = cv2.resize(alpha,(self.img_size,self.img_size),interpolation=cv2.INTER_NEAREST)
        return alpha>0

    def get_depth_range(self,img_id):
        return self.range_dict[img_id].copy()

class BlendedMVSDatabase(BaseDatabase):
    name2uid={'iron_dog': '5c1af2e2bee9a723c963d019',
              'building': '5bf18642c50e6f7f8bdbd492',
              'santa': '5be47bf9b18881428d8fbc1d',
              'dragon': '5bd43b4ba6b28b1ee86b92dd',
              'mermaid': '5ba19a8a360c7c30c1c169df',
              'laid_man': '59e75a2ca9e91f2c5526005d'}
    @staticmethod
    def load_pfm(file):
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data

    @staticmethod
    def load_mvs_cams(fn):
        with open(fn,'r') as f:
            lines=f.readlines()
            extrinsic = [line.strip() for line in lines[1:5]]
            extrinsic_mat = []
            for k in range(4):
                extrinsic_mat.append([float(v) for v in extrinsic[k].split(' ')])
            extrinsic_mat = np.asarray(extrinsic_mat)[:3]

            intrinsic = [line.strip() for line in lines[7:10]]
            intrinsic_mat = []
            for k in range(3):
                intrinsic_mat.append([float(v) for v in intrinsic[k].split(' ')])
            intrinsic_mat = np.asarray(intrinsic_mat)
            depth_min, _, _, depth_max = [float(val) for val in lines[11].split(' ')]
            return extrinsic_mat, intrinsic_mat, [depth_min*0.8,depth_max*1.2]

    def __init__(self, database_name):
        super(BlendedMVSDatabase, self).__init__(database_name)
        _, model_name, half_or_full = database_name.split('/')
        assert(half_or_full in ['half'])
        self.half = half_or_full=='half'
        self.model_name = model_name
        self.uid = self.name2uid[model_name]
        self.root = f'data/blended-mvs/{self.uid}'
        img_num = len(os.listdir(f'{self.root}/blended_images'))//2
        self.img_ids = [str(k) for k in range(img_num)]

        self.poses, self.Ks = [], []
        self.range_dict = {}
        for img_id in self.img_ids:
            pose, K, depth_range = self.load_mvs_cams(f'{self.root}/cams/{int(img_id):08d}_cam.txt')
            self.poses.append(pose)
            self.Ks.append(K)
            self.range_dict[img_id]=depth_range
        self.use_masked=True
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def get_image(self, img_id):
        if self.use_masked:
            img = imread(f'{self.root}/blended_images/{int(img_id):08d}_masked.jpg')
        else:
            img = imread(f'{self.root}/blended_images/{int(img_id):08d}.jpg')
        if self.half:
            img = resize_img(img,0.5)
        return img

    def get_K(self, img_id):
        K = self.Ks[int(img_id)].astype(np.float32()).copy()
        if self.half:
            K = np.diag([0.5,0.5,1]) @ K
        return K

    def get_pose(self, img_id):
        return self.poses[int(img_id)].astype(np.float32).copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self, img_id):
        fn = f'{self.root}/colmap_depth/{img_id}.jpg.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        return read_array(f'{self.root}/colmap_depth/{img_id}.jpg.geometric.bin')

    def get_mask(self, img_id):
        img = self.get_image(img_id)
        return np.sum(img, 2)>0

    def get_depth_range(self,img_id):
        return self.range_dict[img_id].copy()

class ExampleDatabase(BaseDatabase):
    def __init__(self, database_name):
        super(ExampleDatabase, self).__init__(database_name)
        _, scene_name, self.resolution = database_name.split('/')
        self.root_dir=f'data/example/{scene_name}'

        cameras = read_cameras_binary(f'{self.root_dir}/sparse/0/cameras.bin')
        images = read_images_binary(f'{self.root_dir}/sparse/0/images.bin')
        self.img_ids = [str(k+1) for k in range(len(images))]
        self.img_id2pose,self.img_id2K,self.img_id2fn={},{},{}
        for img_id in self.img_ids:
            img_id_int = int(img_id)
            fn = images[img_id_int].name
            self.img_id2fn[img_id] = fn

            R = images[img_id_int].qvec2rotmat()
            t = images[img_id_int].tvec
            pose = np.concatenate([R,t[:,None]],1)
            self.img_id2pose[img_id] = pose.astype(np.float32)

            f, cx, cy = cameras[images[img_id_int].camera_id].params
            self.img_id2K[img_id] = np.asarray([
                [f,0,cx],
                [0,f,cy],
                [0,0,1],
            ],np.float32)

        if self.resolution != 'raw':
            self.resolution = int(self.resolution)
            # note: we assume all images have the same size!
            h, w, _ = imread(f'{self.root_dir}/images/{self.img_id2fn[img_id]}')[:,:,:3].shape
            self.ratio = self.resolution/max(h, w)
        else:
            self.ratio = 1.0

        # self._resize_images()
        self._compute_depth_range()
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def _compute_depth_range(self):
        if os.path.exists(f'{self.root_dir}/depth_range.pkl'):
            self.range_dict=read_pickle(f'{self.root_dir}/depth_range.pkl')
        else:
            # print('estimate depth range ...')
            self.range_dict={}
            pts = read_points3d_binary(f'{self.root_dir}/sparse/0/points3D.bin')
            points = np.asarray([v.xyz for k,v in pts.items()],np.float32)
            for img_id in self.img_ids:
                K = self.get_K(img_id)
                pose = self.get_pose(img_id)
                _, depth = project_points(points,pose,K)
                far = np.percentile(depth.flatten(),98)*1.2
                near = np.percentile(depth.flatten(),2)*0.8
                self.range_dict[img_id]=np.asarray([near,far], np.float32)
            save_pickle(self.range_dict, f'{self.root_dir}/depth_range.pkl')

    def compute_depth_range_impl(self, pose):
        points = read_points3d_binary(f'{self.root_dir}/sparse/0/points3D.bin')
        points = np.asarray([v.xyz for k,v in points.items()],np.float32)
        depth = transform_points_Rt(points,pose[:3,:3],pose[:3,3])[:,2]
        far = np.percentile(depth.flatten(),98)*1.2
        near = np.percentile(depth.flatten(),2)*0.8
        return np.asarray([near,far],np.float32)

    # def _resize_images(self,):
    #     # resize images according to resolution
    #     if self.resolution=='raw': return
    #     Path(f'{self.root_dir}/images_{self.resolution}').mkdir(exist_ok=True,parents=True)
    #     Path(f'{self.root_dir}/stereo/depth_maps_{self.resolution}').mkdir(exist_ok=True,parents=True)
    #     print('resize dataset ...')
    #     for img_id in tqdm(self.img_ids):
    #         # change intrinsics
    #         img = imread(f'{self.root_dir}/images/{img_id}.jpg')
    #         ratio = self.resolution/max(img.shape)
    #         K = self.img_id2K[img_id]
    #         self.img_id2K[img_id] = np.diag([ratio, ratio,1.0]) @ K
    #
    #         if os.path.exists(f'{self.root_dir}/images_{self.resolution}/{img_id}.jpg'): continue
    #
    #         # resize images
    #         img = resize_img(img, ratio)
    #         imsave(f'{self.root_dir}/images_{self.resolution}/{img_id}.jpg',img)
    #
    #         # resize depth maps
    #         depth = read_array(f'{self.root_dir}/stereo/depth_maps/{img_id}.jpg.geometric.bin')
    #         h, w, _ = img.shape
    #         depth = cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
    #         depth = depth.astype(np.float32)
    #         np.save(f'{self.root_dir}/stereo/depth_maps_{self.resolution}/{img_id}.npy',depth)

    def get_image(self, img_id):
        img=imread(f'{self.root_dir}/images/{self.img_id2fn[img_id]}')[:,:,:3]
        if self.resolution!='raw':
            img = resize_img(img, self.ratio)
        return img

    def get_K(self, img_id):
        K = self.img_id2K[img_id].astype(np.float32).copy()
        if self.resolution!='raw':
            K = np.diag([self.ratio, self.ratio,1.0]) @ K
        return K

    def get_pose(self, img_id):
        return self.img_id2pose[img_id].astype(np.float32).copy()

    def get_img_ids(self, check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self,img_id):
        return os.path.exists(f'{self.root_dir}/dense/stereo/depth_maps/{self.img_id2fn[img_id]}.geometric.bin')

    def get_depth(self, img_id):
        depth=read_array(f'{self.root_dir}/dense/stereo/depth_maps/{self.img_id2fn[img_id]}.geometric.bin').astype(np.float32)
        if self.resolution!='raw':
            h, w = depth.shape
            depth = cv2.resize(depth, (int(w*self.ratio),int(h*self.ratio)), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, img_id):
        h, w, _ = self.get_image(img_id).shape
        return np.ones([h,w],np.bool)

    def get_depth_range(self, img_id):
        return self.range_dict[img_id].copy()

class GoogleScannedObjectDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name, background_resolution = database_name.split('/')
        background, resolution = background_resolution.split('_')
        assert(background in ['black','white'])
        self.resolution = resolution
        self.background = background
        self.prefix=f'data/google_scanned_objects/{model_name}'

        ###################compute depth range#################
        range_dict_fn = f'data/google_scanned_objects/{model_name}/depth_range.pkl'
        if os.path.exists(range_dict_fn):
            self.range_dict = read_pickle(range_dict_fn)
        else:
            # print(f'compute depth range for {model_name} ...')
            self.range_dict={}
            for img_id in self.get_img_ids():
                min_ratio = 0.1
                origin_depth = self.get_pose(img_id)[2,3]
                max_radius = 0.5 * np.sqrt(2) * 1.1
                near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
                far_depth = origin_depth + max_radius
                self.range_dict[img_id]=np.asarray([near_depth,far_depth],np.float32)
            save_pickle(self.range_dict,range_dict_fn)

    def get_image(self, img_id):
        img=imread(f'{self.prefix}/rgb/{int(img_id):06}.png')[:,:,:3]
        if self.background=='white':
            pass
        elif self.background=='black':
            mask=imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
            img[~mask]=0
        else:
            raise NotImplementedError
        if self.resolution=='raw':
            pass
        else:
            res = int(self.resolution)
            img = resize_img(img,res/512)
        return img

    def get_K(self, img_id):
        K=np.loadtxt(f'{self.prefix}/intrinsics/{int(img_id):06}.txt').reshape([4,4])[:3,:3]
        if self.resolution!='raw':
            ratio = int(self.resolution) / 512
            K = np.diag([ratio,ratio,1.0]) @ K
        return K.astype(np.float32)

    def get_pose(self, img_id):
        pose = np.loadtxt(f'{self.prefix}/pose/{int(img_id):06}.txt').reshape([4,4])[:3,:]
        R = pose[:3, :3].T
        t = R @ -pose[:3, 3:]
        return np.concatenate([R,t],-1)

    def get_img_ids(self, check_depth_exist=False):
        return [str(img_id) for img_id in range(250)]

    def get_bbox(self, img_id):
        raise NotImplementedError

    def get_depth(self, img_id):
        img = Image.open(f'{self.prefix}/depth/{int(img_id):06}.png')
        depth = np.asarray(img, dtype=np.float32) / 1000.0
        mask = imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
        depth[~mask] = 0
        if self.resolution!='raw':
            res = int(self.resolution)
            depth = cv2.resize(depth.astype(np.float32),(res,res),interpolation=cv2.INTER_NEAREST)
        return depth

    def get_mask(self, img_id):
        mask=imread(f'{self.prefix}/mask/{int(img_id):06}.png')>0
        if self.resolution!='raw':
            res = int(self.resolution)
            mask = cv2.resize(mask.astype(np.uint8),(res,res),interpolation=cv2.INTER_NEAREST)>0
        return mask

    def get_depth_range(self, img_id):
        return self.range_dict[img_id].copy()

class RealIconicDatabase(BaseDatabase):
    def __init__(self, database_name):
        super(RealIconicDatabase, self).__init__(database_name)
        recenter = True
        bd_factor = 0.75
        _, model_name, factor = database_name.split('/')
        factor = int(factor)
        self.factor=factor
        self.images, poses, self.range_dict, self.render_poses, self.test_img_id = load_llff_data(
            f'data/real_iconic_noface/{model_name}', factor, recenter, bd_factor=bd_factor)
        h, w, focal = poses[0, :3, -1]
        self.K = np.asarray([[focal,0.0,w/2],[0.0,focal,h/2],[0.0,0.0,1.0]],dtype=np.float32)
        poses = poses[:, :3, :4]
        self.poses=[]
        for k in range(len(poses)):
            pose = poses[k]
            R = pose[:3, :3].T
            t = R @ -pose[:3, 3:]

            R = np.diag(np.asarray([1, -1, -1])) @ R
            t = np.diag(np.asarray([1, -1, -1])) @ t
            self.poses.append(np.concatenate([R,t],1))

        self.img_ids = [str(k) for k in range(len(self.images))]
        self.test_img_ids=[str(self.test_img_id)]
        self.train_img_ids=[k for k in self.img_ids if k not in self.test_img_ids]
        self.range_dict={str(k):np.asarray(self.range_dict[k],np.float32) for k in range(len(self.range_dict))}
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def get_image(self, img_id):
        return self.images[int(img_id)]

    def get_K(self, img_id):
        return self.K.copy()

    def get_pose(self, img_id):
        return self.poses[int(img_id)].copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self,img_id):
        if self.factor==8:
            fn = f'data/colmap_forward_cache/{self.database_name}' \
                 f'/dense_out/stereo/depth_maps/{img_id}.png.geometric.bin'
        else:
            fn = f'data/colmap_forward_cache/{self.database_name}' \
                 f'/dense_out/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        if self.factor==8:
            fn = f'data/colmap_forward_cache/{self.database_name}' \
                 f'/dense_out/stereo/depth_maps/{img_id}.png.geometric.bin'
        else:
            fn = f'data/colmap_forward_cache/{self.database_name}' \
                 f'/dense_out/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        if not os.path.exists(fn): return None
        depth = read_array(fn)
        near, far = self.get_depth_range(img_id)
        depth = np.clip(depth,a_min=1e-5,a_max=far)
        return depth

    def get_mask(self, img_id):
        h, w = self.get_image(img_id).shape[:2]
        return np.ones([h,w],dtype=np.bool)

    def get_depth_range(self,img_id):
        return self.range_dict[img_id].copy()

class SpaceDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name = database_name.split('/')
        self.views = ReadScene(os.path.join('data/spaces_dataset','data','800',model_name))
        self.img_ids = []
        for rig_id in range(len(self.views)):
            for cam_id in range(len(self.views[rig_id])):
                self.img_ids.append(f'{rig_id}-{cam_id}')
        self.range_dict={img_id:np.asarray((0.7,100),np.float32) for img_id in self.img_ids}
        self.incorrect_intrinsics=False
        if model_name in ['scene_008','scene_038','scene_039']:
            self.incorrect_intrinsics=True
            self.name2Ks={}
            for img_id in self.img_ids:
                view = self.get_view(img_id)
                h0, w0 = imread(view.image_path).shape[:2]
                h1, w1 = view.shape
                self.name2Ks[img_id]= np.diag([w0/w1,h0/h1,1],).astype(np.float32) @ \
                                      np.asarray(view.camera.intrinsics.copy(),np.float32)
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def get_view(self,img_id):
        rig_id, cam_id = img_id.split('-')
        rig_id = int(rig_id)
        cam_id = int(cam_id)
        return self.views[rig_id][cam_id]

    # img_size = view.shape
    # image_path = view.image_path
    # intrinsics = view.camera.intrinsics
    # intrinsics_4x4 = np.eye(4)
    # intrinsics_4x4[:3, :3] = intrinsics
    # c2w = view.camera.w_f_c
    # return image_path, img_size, intrinsics_4x4, c2w

    def get_image(self, img_id):
        view=self.get_view(img_id)
        return imread(view.image_path)

    def get_K(self, img_id):
        if self.incorrect_intrinsics:
            return self.name2Ks[img_id]
        view=self.get_view(img_id)
        return np.asarray(view.camera.intrinsics.copy(),np.float32)

    def get_pose(self, img_id):
        view=self.get_view(img_id)
        c2w=view.camera.w_f_c
        pose=c2w[:3,:]
        pose=pose_inverse(pose)
        return pose.copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self, img_id):
        fn = f'data/colmap_forward_cache/{self.database_name}' \
             f'/dense_out/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        fn = f'data/colmap_forward_cache/{self.database_name}' \
             f'/dense_out/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        if not os.path.exists(fn): return None
        depth = read_array(fn)
        near, far = self.get_depth_range(img_id)
        depth = np.clip(depth,a_min=1e-5,a_max=far)
        return depth

    def get_mask(self, img_id):
        view=self.get_image(img_id)
        h,w=view.shape[:2]
        return np.ones([h,w],dtype=np.bool)

    def get_depth_range(self,img_id):
        return self.range_dict[img_id].copy()

class RealEstateDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name, img_size = database_name.split('/')
        self.model_name = model_name
        self.root_dir='data/real_estate_dataset/train'
        h, w = img_size.split('_') # 450, 800
        self.target_height, self.target_width = int(h), int(w)
        fns = os.listdir(f'{self.root_dir}/frames/{model_name}')
        self.img_ids = [fn.split('.')[0] for fn in fns]
        self.img_ids = np.asarray(self.img_ids)
        idxs = np.argsort(self.img_ids.astype(np.int32))
        self.img_ids = self.img_ids[idxs]
        self.img_ids = self.img_ids.tolist()
        self.cam_params = parse_pose_file(f'{self.root_dir}/cameras/{model_name}.txt')
        self.range_dict = {img_id: np.asarray((1.,100.),np.float32) for img_id in self.img_ids}
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def get_image(self, img_id):
        img=imread(f'{self.root_dir}/frames/{self.model_name}/{img_id}.png')
        return cv2.resize(img,(self.target_width,self.target_height),interpolation=cv2.INTER_AREA)

    def get_K(self, img_id):
        intrinsics = unnormalize_intrinsics(self.cam_params[int(img_id)].intrinsics.copy(),self.target_height,self.target_width)
        return intrinsics[:3,:3].copy()

    def get_pose(self, img_id):
        return self.cam_params[int(img_id)].w2c_mat[:3,:4].copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist: return self.depth_img_ids
        return [img_id for img_id in self.img_ids]

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self, img_id):
        model_name = self.database_name.split('/')[1]
        fn=f'data/colmap_forward_cache/real_estate/{model_name}/' \
           f'dense_out/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        assert(self.target_width==800 and self.target_height==450)
        model_name = self.database_name.split('/')[1]
        fn=f'data/colmap_forward_cache/real_estate/{model_name}/dense_out/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        if os.path.exists(fn):
            depth = read_array(fn)
            near, far = self.get_depth_range(img_id)
            depth = np.clip(depth,a_min=1e-5,a_max=far)
            return depth
        else:
            return None

    def get_mask(self, img_id):
        return np.ones([self.target_height,self.target_width],dtype=np.bool)

    def get_depth_range(self, img_id):
        return self.range_dict[img_id].copy()

class DTUTrainDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        _, model_name = database_name.split('/')
        root_dir=f'data/dtu_train/{model_name}'

        self._coord_trans_world = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],dtype=np.float32,)

        rgb_paths = [x for x in glob.glob(os.path.join(root_dir, "image", "*")) if (x.endswith(".jpg") or x.endswith(".png"))]
        rgb_paths = sorted(rgb_paths)

        all_cam = np.load(os.path.join(root_dir, "cameras.npz"))
        self.Rts=[]
        self.Ks=[]
        self.img_ids=[]
        for i, rgb_path in enumerate(rgb_paths):
            P = all_cam["world_mat_" + str(i)]
            P = P[:3]
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]

            scale_mtx = all_cam.get("scale_mat_" + str(i))
            if scale_mtx is not None:
                norm_trans = scale_mtx[:3, 3:]
                norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

                pose[:3, 3:] -= norm_trans
                pose[:3, 3:] /= norm_scale
            pose = self._coord_trans_world @ pose # @ self._coord_trans_cam
            pose = pose[:3]
            pose = np.concatenate([pose[:,:3].T,- pose[:,:3].T @ pose[:,3:]],1)

            self.Ks.append(K)
            self.Rts.append(pose[:3])
            self.img_ids.append(f'{i}')

        self.range_dict={img_id:np.asarray((0.8,4.0),np.float32) for img_id in self.img_ids}
        self.root_dir=root_dir
        self.depth_img_ids = [img_id for img_id in self.img_ids if self._depth_existence(img_id)]

    def get_image(self, img_id):
        img = imread(os.path.join(self.root_dir,'image',f'{int(img_id):06}.png'))
        return img

    def get_K(self, img_id):
        return self.Ks[int(img_id)].copy()

    def get_pose(self, img_id):
        return self.Rts[int(img_id)].copy()

    def get_img_ids(self, check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids

    def get_bbox(self, img_id):
        raise NotImplementedError

    def _depth_existence(self, img_id):
        fn=f'data/colmap_dtu_cache/{self.database_name}/dense/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        return os.path.exists(fn)

    def get_depth(self, img_id):
        fn=f'data/colmap_dtu_cache/{self.database_name}/dense/stereo/depth_maps/{img_id}.jpg.geometric.bin'
        if os.path.exists(fn):
            depth=read_array(fn)
            near, far = self.get_depth_range(img_id)
            depth = np.clip(depth, a_min=1e-5, a_max=far)
            return depth
        else:
            return None

    def get_mask(self, img_id):
        # mask = np.sum(imread(os.path.join(self.root_dir, 'mask', f'{int(img_id):03}.png')),-1)>0
        mask = np.ones([300,400],np.bool)
        return mask

    def get_depth_range(self,img_id):
        return self.range_dict[img_id].copy()


def parse_database_name(database_name:str)->BaseDatabase:
    name2database={
        # training database
        'gso': GoogleScannedObjectDatabase,
        'space': SpaceDatabase,
        'real_iconic': RealIconicDatabase,
        'real_estate': RealEstateDatabase,
        'dtu_train': DTUTrainDatabase,

        # evaluation database
        'dtu_test': DTUTestDatabase,
        'nerf_synthetic': NeRFSyntheticDatabase,
        'llff_colmap': LLFFColmapDatabase,
        'blended_mvs': BlendedMVSDatabase,
        'example': ExampleDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError

def get_database_split(database: BaseDatabase, split_type='val'):
    database_name = database.database_name
    if split_type.startswith('val'):
        splits = split_type.split('_')
        depth_valid = not(len(splits)>1 and splits[1]=='all')
        if database_name.startswith('nerf_synthetic'):
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
            val_ids = nerf_syn_val_ids
        elif database_name.startswith('llff'):
            val_ids = database.get_img_ids()[::8]
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
        elif database_name.startswith('dtu_test'):
            val_ids = database.get_img_ids()[3:-3:8]
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
        else:
            raise NotImplementedError
    elif split_type.startswith('test'):
        splits = split_type.split('_')
        depth_valid = not(len(splits)>1 and splits[1]=='all')
        if database_name.startswith('nerf_synthetic'):
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
            val_ids = [img_id for img_id in database.get_img_ids() if img_id.startswith('te')]
        elif database_name.startswith('llff'):
            val_ids = database.get_img_ids()[::8]
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
        elif database_name.startswith('dtu_test'):
            val_ids = database.get_img_ids()[3:-3:8]
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return train_ids, val_ids