import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..dataset import DatasetTemplate
from .carlaocc_eval import evaluate_detections

def _quat_to_rotmat(q):
    """q = [w, x, y, z] (nuScenes/pyquaternion convention)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

_R_OCV2SENSOR = np.array([[0, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0]], dtype=np.float64)
class CarlaOccDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.root_path = Path(self.root_path).expanduser().resolve()
        self.infos = []
        self.include_data(self.mode)
        self.num_point_features = self.point_feature_encoder.num_point_features

        self.use_camera = bool(
            dataset_cfg.get('CAMERA_CONFIG', {}).get('USE_CAMERA', False)
            if hasattr(dataset_cfg.get('CAMERA_CONFIG', {}), 'get')
            else False
        )

    def include_data(self, mode):
        info_paths = self.dataset_cfg.INFO_PATH.get(mode, [])
        for info_path in info_paths:
            path = Path(str(info_path).replace('${CARLAOCC_ROOT}', str(self.root_path)))
            if not path.is_absolute():
                path = self.root_path / path
            with path.open('rb') as f:
                self.infos.extend(pickle.load(f))
        self.logger.info(f'Loading CarlaOcc {mode}: {len(self.infos)} samples') if self.logger else None

    def get_lidar(self, info):
        path = Path(info['lidar_path'])
        if not path.is_absolute():
            path = self.root_path / path
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return points

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        current = self.get_lidar(info)
        if max_sweeps <= 1 or not info.get('sweeps'):
            return current
        points = [current]
        for sweep in info['sweeps'][:max_sweeps - 1]:
            path = Path(sweep['lidar_path'])
            if not path.is_absolute():
                path = self.root_path / path
            if not path.exists():
                continue
            pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            T = np.asarray(sweep['transform_matrix'], dtype=np.float32)
            xyz1 = np.concatenate([pts[:, :3], np.ones((len(pts), 1), dtype=np.float32)], axis=1)
            xyz = (T @ xyz1.T).T[:, :3]
            time = np.full((len(pts), 1), float(sweep['time_lag']), dtype=np.float32)
            points.append(np.hstack([xyz, pts[:, 3:4], time]))
        if points[0].shape[1] == 4 and len(points) > 1:
            points[0] = np.hstack([points[0], np.zeros((len(points[0]), 1), dtype=np.float32)])
        return np.concatenate(points, axis=0)



    def _resize_crop(self, img):
        cfg = self.dataset_cfg.CAMERA_CONFIG.IMAGE
        fH, fW = cfg.FINAL_DIM
        W, H = img.size
        if self.training:
            resize = np.random.uniform(*cfg.RESIZE_LIM_TRAIN)
        else:
            resize = cfg.RESIZE_LIM_TEST[0]
        newW, newH = int(W * resize), int(H * resize)
        assert newH >= fH, f'resized height {newH} < FINAL_DIM height {fH}'
        crop_h = newH - fH                       # keep the bottom, drop sky
        if self.training:
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        else:
            crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        img = img.resize((newW, newH)).crop(crop)
        return img, [resize, crop, False, 0.0]
    
    def load_camera(self, info, data_dict):
        image_paths, camera_imgs, image_shapes, img_process_infos = [], [], [], []
        lidar2camera, lidar2image, camera2lidar, camera2ego, camera_intrinsics = [], [], [], [], []
    
        for cam_name in sorted(info['cams']):
            cam = info['cams'][cam_name]
            path = Path(cam['data_path'])
            if not path.is_absolute():
                path = self.root_path / path
            img = Image.open(path).convert('RGB')
            image_shapes.append(np.array(img).shape[:2])      # original shape
            img, proc = self._resize_crop(img)
            camera_imgs.append(img)
            img_process_infos.append(proc)
            image_paths.append(str(path))
    
            s2l_rot = np.asarray(cam['sensor2lidar_rotation'], dtype=np.float64)
            s2l_trans = np.asarray(cam['sensor2lidar_translation'], dtype=np.float64)
    
            cam2lidar_mat = np.eye(4, dtype=np.float64)
            cam2lidar_mat[:3, :3] = s2l_rot @ _R_OCV2SENSOR   # <-- axis fix
            cam2lidar_mat[:3, 3] = s2l_trans
            camera2lidar.append(cam2lidar_mat.astype(np.float32))
            camera2ego.append(cam2lidar_mat.astype(np.float32))  # ego == lidar here
    
            lidar2cam_mat = np.linalg.inv(cam2lidar_mat)
            lidar2camera.append(lidar2cam_mat.astype(np.float32))
    
            K = np.asarray(cam.get('camera_intrinsics', cam.get('cam_intrinsic')), dtype=np.float64)
            intrin = np.eye(4, dtype=np.float64)
            intrin[:3, :3] = K
            camera_intrinsics.append(intrin.astype(np.float32))
            lidar2image.append((intrin @ lidar2cam_mat).astype(np.float32))
    
        data_dict.update({
            'camera_imgs': camera_imgs,
            'image_paths': image_paths,
            'image_shape': image_shapes,
            'lidar2camera': lidar2camera,
            'lidar2image': lidar2image,
            'camera2lidar': camera2lidar,
            'camera2ego': camera2ego,
            'camera_intrinsics': camera_intrinsics,
            'img_process_infos': img_process_infos,
        })
        return data_dict
    def __len__(self):
        if getattr(self, '_merge_all_iters_to_one_epoch', False):
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, self.dataset_cfg.MAX_SWEEPS)
        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
            'metadata': {'token': info['token']},
            'use_lead_xyz': self.point_feature_encoder.used_feature_list[0] == 'x',
        }

        if self.training or self.dataset_cfg.get('GET_ITEM_WITH_GT', True):
            gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
            gt_velocity = np.asarray(info.get('gt_velocity', np.zeros((len(gt_boxes), 2))), dtype=np.float32)
            if len(gt_boxes):
                gt_boxes = np.concatenate([gt_boxes, np.nan_to_num(gt_velocity)], axis=1)
            else:
                gt_boxes = np.zeros((0, 9), dtype=np.float32)
            input_dict['gt_boxes'] = gt_boxes
            input_dict['gt_names'] = np.asarray(info['gt_names'])
            input_dict['instance_ids'] = np.asarray(info.get('instance_ids', np.zeros(len(gt_boxes), dtype=np.int64)))

        if self.use_camera:
            input_dict = self.load_camera(info, input_dict)

        data_dict = self.prepare_data(data_dict=input_dict)
        if self.use_camera and 'camera_imgs' in data_dict:
            data_dict['camera_imgs'] = [
                img if torch.is_tensor(img) else torch.from_numpy(np.array(img))
                for img in data_dict['camera_imgs']
            ]

        data_dict.pop('gt_names', None)
        data_dict.pop('instance_ids', None)

        return data_dict

    def set_split(self, split):
        self.infos = []
        self.training = split == 'train'
        self.include_data(self.mode)

    def evaluation(self, det_annos, class_names, **kwargs):
        return evaluate_detections(det_annos, self.infos, class_names, self.point_cloud_range)
