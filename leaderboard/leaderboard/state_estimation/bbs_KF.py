import math
import os
import sys
import traceback
import numpy as np
from typing import Optional, Tuple
import carla
import threading
import queue
from PIL import Image
from leaderboard.state_estimation.sensor_dataclasses import bev_Det, trackedbb, CalculatedState, Ego3DState
from leaderboard.state_estimation.TwoDimTTC import TTC
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from leaderboard.state_estimation.UKF_bb import UKFBoundingBoxTracker, ObjectUKF
import leaderboard.state_estimation.hyperparams as hp
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

DEBUG_PRINT = False
FIX_FRAMES = True

def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def build_camera_calibration(lidar2ego: np.ndarray = None) -> dict:
    """
    lidar2ego: optional 4x4 (right-handed lidar frame -> right-handed ego frame).
               Only used for 'camera2ego'; identity if omitted.
    Returns float32 arrays, each of shape (6, 4, 4), in CAM_ORDER.
    """
    if lidar2ego is None:
        lidar2ego = np.eye(4)

    fx = (hp.SRC_W / 2.0) / math.tan(math.radians(hp.FOV_DEG) / 2.0)
    fy = fx                                    
    sx, sy = hp.OUT_W / hp.SRC_W, hp.OUT_H / hp.SRC_H
    K = np.eye(4)
    K[0, 0], K[1, 1] = fx * sx, fy * sy
    K[0, 2], K[1, 2] = (hp.SRC_W / 2.0) * sx, (hp.SRC_H / 2.0) * sy

    out = {k: [] for k in ("camera2lidar", "lidar2camera", "lidar2image",
                           "camera_intrinsics", "camera2ego")}
    for name in hp.CAM_ORDER:
        x, y, z, yaw = hp.CAM_RIG[name]
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = _rot_z(-yaw) @ hp.R_CAM_TO_LIDAR   # yaw flips sign in a right-handed frame
        cam2lidar[:3, 3] = (x, -y, z)                        # y flips too
        lidar2cam = np.linalg.inv(cam2lidar)
        out["camera2lidar"].append(cam2lidar)
        out["lidar2camera"].append(lidar2cam)
        out["camera_intrinsics"].append(K.copy())
        out["lidar2image"].append(K @ lidar2cam)
        out["camera2ego"].append(lidar2ego @ cam2lidar)

    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}

def lidar2ego_from_offset(lx: float, ly_carla: float, lz: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (lx, -ly_carla, lz)
    return m

def lidar_offset(lidar_transform) -> np.ndarray:
    if lidar_transform is None:
        return np.zeros(3)
    if hasattr(lidar_transform, "location"):
        l = lidar_transform.location
        return np.array([l.x, l.y, l.z], dtype=np.float64)
    return np.asarray(lidar_transform, dtype=np.float64)[:3, 3].copy()

def points_in_rotated_box(points_xyz: np.ndarray, center_x: float, center_y: float, center_z: float,
                           length: float, width: float, height: float, yaw_deg: float, margin: float = 0.1) -> np.ndarray:
    boxes = np.array([[center_x, center_y, center_z, length + (margin * 2), width + (margin * 2),
                       height + (margin * 2), np.deg2rad(yaw_deg)]])
    points_tensor = torch.from_numpy(points_xyz).float()
    boxes_tensor = torch.from_numpy(boxes).float()
    box_point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points_tensor, boxes_tensor)
    return box_point_masks[0].numpy() > 0

def _dbg(*a):
    if DEBUG_PRINT:
        print(*a, flush=True)

class _DummyLogger:
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass

class _SingleFrameDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training: bool = False):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training)
        self._points = np.zeros((0, 4), dtype=np.float32)
        self._imgs = None                       # (N_cam, H, W, 3) uint8 RGB
        self._calib = build_camera_calibration()

    def __len__(self): return 1

    def set_lidar_to_ego(self, lidar2ego: np.ndarray):
        self._calib = build_camera_calibration(lidar2ego=lidar2ego)

    def load_data(self, points, imgs):
        n_src = len(getattr(self.point_feature_encoder, "src_feature_list", [])) or points.shape[1]
        if points.shape[1] < n_src:
            pad = np.zeros((points.shape[0], n_src - points.shape[1]), dtype=np.float32)
            points = np.hstack([points, pad])
        elif points.shape[1] > n_src:
            points = points[:, :n_src]
        self._points = np.ascontiguousarray(points, dtype=np.float32)
        self._imgs = imgs

    def _crop_images(self, pil_imgs, ori_shape):
        """Test-time branch of NuScenesDataset.crop_image."""
        W, H = ori_shape
        img_cfg = self.dataset_cfg.get("IMAGE", None)
        if img_cfg is None:
            fH, fW, resize = H, W, 1.0
        else:
            fH, fW = img_cfg.FINAL_DIM
            resize = float(np.mean(img_cfg.RESIZE_LIM_TEST))
        newW, newH = int(W * resize), int(H * resize)
        crop_h = newH - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        out, infos = [], []
        for img in pil_imgs:
            out.append(img.resize((newW, newH)).crop(crop))
            infos.append([resize, crop, False, 0])
        return out, infos

    def __getitem__(self, index):
        pil = [Image.fromarray(np.ascontiguousarray(im)) for im in self._imgs]
        ori_shape = pil[0].size                              # (W, H)
        cropped, infos = self._crop_images(pil, ori_shape)
        c = self._calib
        data_dict = {
            "points": self._points,
            "frame_id": 0,
            "camera_imgs": cropped,
            "ori_shape": ori_shape,
            "img_process_infos": infos,
            "camera_intrinsics": c["camera_intrinsics"],
            "camera2lidar": c["camera2lidar"],
            "lidar2camera": c["lidar2camera"],
            "lidar2image": c["lidar2image"],
            "camera2ego": c["camera2ego"],
            "lidar_aug_matrix": np.eye(4, dtype=np.float32)
        }
        return self.prepare_data(data_dict=data_dict)


class OpenPCDetDetector:
    def __init__(self, cfg_file: str = hp.OPENPCDET_CFG_FILE, ckpt_file: str = hp.OPENPCDET_CKPT_FILE,
                 device: str = hp.OPENPCDET_DEVICE, openpcdet_tools_dir: str = hp.OPENPCDET_TOOLS_DIR):
        prev_cwd = os.getcwd()
        try:
            if openpcdet_tools_dir:
                os.chdir(openpcdet_tools_dir)
            cfg_from_yaml_file(cfg_file, cfg)
        finally:
            os.chdir(prev_cwd)
        _dbg("starting the detector | cfg:", cfg_file, "| ckpt:", ckpt_file, "| device:", device)
        self.class_names = cfg.CLASS_NAMES
        self.dataset = _SingleFrameDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=self.class_names)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=self.dataset)
        self.model.load_params_from_file(filename=ckpt_file, logger=_DummyLogger(), to_cpu=(device == "cpu"))
        self.device = device
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def infer(self, points_xyzi: np.ndarray, imgs: np.ndarray) -> list[dict]:
        if imgs is None:
            _dbg("no camera images, skipping inference")
            return []
        self.dataset.load_data(points_xyzi.astype(np.float32), imgs)
        data_dict = self.dataset.collate_batch([self.dataset[0]])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = self.model.forward(data_dict)
        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].cpu().numpy()
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy()
        _dbg(f"inference done: {len(boxes)} raw boxes, max score "  f"{float(scores.max()) if len(scores) else 0.0:.3f}")

        results = []
        for box, score, label in zip(boxes, scores, labels):
            if score < hp.OPENPCDET_SCORE_THRESH:
                continue
            entry = {
                "class_name": self.class_names[int(label) - 1],
                "score": float(score),
                "x": float(box[0]), "y": float(box[1]), "z": float(box[2]),
                "length": float(box[3]), "width": float(box[4]), "height": float(box[5]),
                "heading": float(box[6]), "vx": float(box[7]), "vy": float(box[8])
            }
            results.append(entry)
        return results

def ego_to_world_matrix(ego_state) -> np.ndarray:
    return np.array(carla.Transform(
        carla.Location(x=ego_state.x, y=ego_state.y, z=ego_state.z),
        carla.Rotation(pitch=math.degrees(ego_state.pitch_rad), yaw=math.degrees(ego_state.yaw_rad),
                       roll=math.degrees(ego_state.roll_rad))).get_matrix())

def _match_radar_points(radar_points: np.ndarray, center_x: float, center_y: float, center_z: float,
                        length: float, width: float, height: float, yaw_deg: float, margin: float = 1.0) -> np.ndarray:
    if not radar_points.size:
        return np.array([])
    min_x, max_x = center_x - length / 2 - margin, center_x + length / 2 + margin
    min_y, max_y = center_y - width / 2 - margin, center_y + width / 2 + margin
    min_z, max_z = center_z - height / 2 - margin, center_z + height / 2 + margin
    coarse_mask = (radar_points[:, 0] >= min_x) & (radar_points[:, 0] <= max_x) & \
                  (radar_points[:, 1] >= min_y) & (radar_points[:, 1] <= max_y) & \
                  (radar_points[:, 2] >= min_z) & (radar_points[:, 2] <= max_z)
    candidate_points = radar_points[coarse_mask]
    if len(candidate_points) == 0:
        return np.array([])
    inside = points_in_rotated_box(candidate_points[:, :3], center_x, center_y, center_z,
                                    length, width, height, yaw_deg, margin)
    return candidate_points[inside]


def _vertical_overlap(z_a: float, h_a: float, z_b: float, h_b: float) -> bool:
    a_lo, a_hi = z_a - h_a / 2.0, z_a + h_a / 2.0
    b_lo, b_hi = z_b - h_b / 2.0, z_b + h_b / 2.0
    return a_lo <= b_hi and b_lo <= a_hi


def time_to_road_departure(carla_map: carla.Map, ego: Ego3DState, dims: np.array, dt: float = hp.ROAD_EXIT_STEP_S,
                           max_time: float = hp.ROAD_EXIT_LOOKAHEAD_S) -> float:
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    ax_w = c * ego.long_accel - s * ego.lat_accel      # check the sign of lat_accel in your EKF
    ay_w = s * ego.long_accel + c * ego.lat_accel
    x = np.array([ego.x, ego.y, ego.vx_world, ego.vy_world, ax_w, ay_w,
                  ego.yaw, ego.yaw_rate, dims[0], dims[1], ego.z, dims[2]])

    steps = max(1, int(max_time / dt))
    for i in range(steps + 1):
        t = i * dt
        if i > 0:
            x = ObjectUKF.fx(x, dt)

        px, py, pz = x[0], x[1], x[10]
        wp = carla_map.get_waypoint(carla.Location(x=px, y=py, z=pz), project_to_road=False, lane_type=carla.LaneType.Driving)
        if wp is None:
            return -1.0 if i == 0 else t

        wp_loc = wp.transform.location
        right = wp.transform.get_right_vector()
        lateral = (px - wp_loc.x) * right.x + (py - wp_loc.y) * right.y
        if abs(lateral) + dims[1] / 2 > wp.lane_width / 2.0:
            return -1.0 if i == 0 else t

        x[10] = wp_loc.z

    return float('inf')

def run_detection(half_length, half_width, state: CalculatedState, master_cloud: np.ndarray,
                  radar_points, cameras, step, detector: OpenPCDetDetector, lidar_offset: np.ndarray = None):
    if lidar_offset is None:
        lidar_offset = np.zeros(3)

    ex = master_cloud[:, 0] + lidar_offset[0]
    ey = -master_cloud[:, 1] + lidar_offset[1]
    ez = master_cloud[:, 2] + lidar_offset[2]
    obstacle_mask = (ex > -hp.MAX_LOCKBACK) & (ex < hp.MAX_LOOKFORWARD) & \
                    (np.abs(ey) < hp.MAX_LOOKSIDEWAYS) & (ez > -hp.GROUND_CLEARANCE) & \
                    ((np.abs(ex) > half_length) | (np.abs(ey) > half_width))
    obstacle_pts = master_cloud[obstacle_mask]

    if len(obstacle_pts) < hp.MIN_NUMBER_LIDAR_HITS:
        return [], [], obstacle_pts

    T = ego_to_world_matrix(state.ego_state)
    R = T[:3, :3]
    network_detections: list[bev_Det] = []
    network_static: list[trackedbb] = []
    for box in detector.infer(obstacle_pts, cameras):
        _dbg("detected box:", box)
        class_name = box["class_name"].strip().lower()
        if class_name not in hp.OPENPCDET_DYNAMIC_CLASSES and class_name not in hp.OPENPCDET_STATIC_CLASSES:
            continue

        if FIX_FRAMES:
            local = np.array([box["x"] + lidar_offset[0], -box["y"] + lidar_offset[1],
                              box["z"] + lidar_offset[2], 1.0])
            heading = -box["heading"]
            vel_local = np.array([box["vx"], -box["vy"], 0.0])
        else:
            local = np.array([box["x"], box["y"], box["z"], 1.0])
            heading = box["heading"]
            vel_local = np.array([box["vx"], box["vy"], 0.0])

        center_global = T @ local
        heading_dir_global = R @ np.array([math.cos(heading), math.sin(heading), 0.0])
        vel_global = R @ vel_local if FIX_FRAMES else vel_local

        center_x, center_y, center_z = float(center_global[0]), float(center_global[1]), float(center_global[2])
        yaw_deg = math.degrees(math.atan2(heading_dir_global[1], heading_dir_global[0]))
        length, width, height = box["length"], box["width"], box["height"]
        vx, vy = float(vel_global[0]), float(vel_global[1])

        if class_name in hp.OPENPCDET_DYNAMIC_CLASSES:
            matched_radars = _match_radar_points(radar_points, center_x, center_y, center_z,
                                                 length, width, height, yaw_deg)
            network_detections.append(bev_Det(
                center_x=center_x, center_y=center_y, length=length, width=width, vx=vx, vy=vy,
                yaw=yaw_deg, step=step, center_z=center_z, height=height, radar_points=matched_radars, score=box["score"]
            ))
        else:
            network_static.append(trackedbb(
                px=center_x, py=center_y, moveable=False,
                vx=0.0, vy=0.0, speed=0.0, acceleration=0.0, ax=0.0, ay=0.0,
                yaw=math.radians(yaw_deg), yaw_rate=0.0,
                len=length, wid=max(width, 0.3),
                z=center_z, height=height, score= box["score"],
                pos_history=[], slip_angle_deg=0.0
            ))
    return network_detections, network_static, obstacle_pts


def radar_measurement_to_array(radar_data) -> np.ndarray:
    if radar_data is None:
        return np.empty((0, 4), dtype=np.float32)
    if isinstance(radar_data, np.ndarray):
        return radar_data.astype(np.float32).reshape(-1, 4)
    arr = np.array([[d.depth, d.azimuth, d.altitude, d.velocity] for d in radar_data], dtype=np.float32)
    return arr.reshape(-1, 4)


def process_radar(state: CalculatedState, radar_arr: np.ndarray, sensor_transform: np.ndarray) -> np.ndarray:
    if GameTime.get_time() < 0.5 or radar_arr.shape[0] == 0:
        return np.empty((0, 7), dtype=np.float32)
    e = state.ego_state
    T_sensor_to_world = ego_to_world_matrix(e) @ sensor_transform
    R_sensor_to_world = T_sensor_to_world[:3, :3]
    v_ego_world = np.array([e.vx_world, e.vy_world, 0.0], dtype=np.float64)

    r, az, alt, v_r = (radar_arr[:, i].astype(np.float64) for i in range(4))
    cos_alt = np.cos(alt)
    u_sensor = np.stack([cos_alt * np.cos(az), cos_alt * np.sin(az), np.sin(alt)], axis=1)   
    p_sensor = np.hstack([u_sensor * r[:, None], np.ones((len(r), 1))])                       
    p_world = (T_sensor_to_world @ p_sensor.T).T[:, :3]
    u_world = u_sensor @ R_sensor_to_world.T
    v_r_abs = v_r + u_world @ v_ego_world
    return np.column_stack([p_world, v_r_abs, u_world]).astype(np.float32)

def detect_from_sensors(state: CalculatedState, lidar_stream, radar_streams, cameras, step,
                        radar_transforms, half_length, half_width, detector: OpenPCDetDetector,
                        lidar_offset: np.ndarray = None):
    if cameras is None:
        return None
    radar_points = np.empty((0, 7), dtype=np.float32)
    for sensor_name, radar_arr in radar_streams.items():
        radar_points = np.vstack([radar_points, process_radar(state, radar_arr, radar_transforms[sensor_name])])

    pts_sensor = np.asarray(lidar_stream, dtype=np.float32).reshape(-1, 4)
    ranges = np.linalg.norm(pts_sensor[:, :3], axis=1)
    pts_sensor = pts_sensor[ranges > hp.MIN_VALID_RANGE_M]
    pts_sensor[:, 1] = -pts_sensor[:, 1]         

    return run_detection(half_length, half_width, state, pts_sensor, radar_points, cameras, step,
                         detector, lidar_offset)


def compute_ttcs(tracked, state: CalculatedState, carla_map: carla.Map, half_length, half_width,
                 half_height: float = hp.DEFAULT_EGO_HALF_HEIGHT):
    """Runs in the main thread every step, against the CURRENT ego state and extrapolated tracks."""
    if state.ego_state.speed > 1e-3:
        hx = state.ego_state.vx_world / state.ego_state.speed
        hy = state.ego_state.vy_world / state.ego_state.speed
    else:
        hx = math.cos(state.ego_state.yaw)
        hy = math.sin(state.ego_state.yaw)
    ego_ttc = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.vx_world,
                        state.ego_state.vy_world, hx, hy, half_length * 2, half_width * 2])

    obs_ttc = []
    for track in tracked:
        if track.moveable and track.speed > 1e-3:
            hx = track.vx / track.speed
            hy = track.vy / track.speed
        else:
            hx = math.cos(track.yaw)
            hy = math.sin(track.yaw)
        obs_ttc.append((track.px, track.py, track.vx, track.vy, hx, hy, track.len, track.wid))

    if len(obs_ttc) > 0:
        obs_ttc = np.array(obs_ttc)
        ttcs = TTC(ego_ttc, obs_ttc)
        ego_z_center = state.ego_state.z + half_height
        ego_h = half_height * 2.0
        for i, track in enumerate(tracked):
            if ttcs[i] == -1 and not _vertical_overlap(ego_z_center, ego_h, track.z, track.height):
                ttcs[i] = 0.0
    else:
        ttcs = np.array([])

    #road_exit_ttc = time_to_road_departure(carla_map, state.ego_state,
    #                                       np.array([half_length * 2, half_width * 2, half_height * 2]))
    #ttcs = np.append(ttcs, road_exit_ttc)
    if len(ttcs) == 0:
        return ttcs, float('inf')
    return ttcs, float(np.min(ttcs))

"""def get_lidar_radar_info(state: CalculatedState, lidar_stream, radar_streams, cameras, step,
                         radar_transforms, half_length, half_width, tracker, carla_map: carla.Map,
                         detector: OpenPCDetDetector, half_height: float = hp.DEFAULT_EGO_HALF_HEIGHT,
                         lidar_offset: np.ndarray = None):
    radar_points = np.empty((0, 7), dtype=np.float32)
    for sensor_name, radar_arr in radar_streams.items():
        radar_points = np.vstack([radar_points, process_radar(state, radar_arr, radar_transforms[sensor_name])])

    pts_sensor = np.asarray(lidar_stream, dtype=np.float32).reshape(-1, 4)
    ranges = np.linalg.norm(pts_sensor[:, :3], axis=1)
    pts_sensor = pts_sensor[ranges > hp.MIN_VALID_RANGE_M]
    pts_sensor[:, 1] = -pts_sensor[:, 1]          # CARLA (left-handed) -> right-handed, y left

    tracked, pts = build_bbs(half_length=half_length, half_width=half_width,
                             state=state, master_cloud=pts_sensor, radar_points=radar_points,
                             cameras=cameras, step=step, tracker=tracker, detector=detector,
                             lidar_offset=lidar_offset)

    if state.ego_state.speed > 1e-3:
        hx = state.ego_state.vx_world / state.ego_state.speed
        hy = state.ego_state.vy_world / state.ego_state.speed
    else:
        hx = math.cos(state.ego_state.yaw)
        hy = math.sin(state.ego_state.yaw)
    ego_ttc = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.vx_world,
                        state.ego_state.vy_world, hx, hy, half_length * 2, half_width * 2])

    obs_ttc = []
    for track in tracked:
        if track.moveable:
            if track.speed > 1e-3:
                hx = track.vx / track.speed
                hy = track.vy / track.speed
            else:
                hx = math.cos(track.yaw)
                hy = math.sin(track.yaw)
        else:
            hx = math.cos(track.yaw)
            hy = math.sin(track.yaw)
        obs_ttc.append((track.px, track.py, track.vx, track.vy, hx, hy, track.len, track.wid))

    if len(obs_ttc) > 0:
        obs_ttc = np.array(obs_ttc)
        ttcs = TTC(ego_ttc, obs_ttc)

        ego_z_center = state.ego_state.z + half_height
        ego_h = half_height * 2.0
        for i, track in enumerate(tracked):
            if ttcs[i] == -1 and not _vertical_overlap(ego_z_center, ego_h, track.z, track.height):
                ttcs[i] = 0.0
    else:
        ttcs = np.array([])

    road_exit_ttc = time_to_road_departure(carla_map, state.ego_state,
                                           np.array([half_length * 2, half_width * 2, half_height * 2]))
    ttcs = np.append(ttcs, road_exit_ttc)
    min_ttc = float(np.min(ttcs))

    return tracked, ttcs, min_ttc, pts
"""

class BBsWorker(threading.Thread):
    """
    The detector runs in this thread. The UKF tracker lives in the caller's thread.
      push_frame()        hands the newest frame to the detector and never blocks.
      get_latest_result() applies finished detections as UKF updates, extrapolates tracks
                          to the current step, and computes TTCs.
    """
    def __init__(self, lidar_transform, radar_transforms, half_length, half_width, half_height: float = hp.DEFAULT_EGO_HALF_HEIGHT):
        super().__init__()
        self.tracker = UKFBoundingBoxTracker()
        self.detector = OpenPCDetDetector()
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        self.running = True
        self.daemon = True
        self.lidar_transform = lidar_transform
        self.lidar_offset = lidar_offset(lidar_transform)
        self.detector.dataset.set_lidar_to_ego(lidar2ego_from_offset(*self.lidar_offset))
        self.radar_transforms: dict[str, np.ndarray] = {name: np.array(t.get_matrix(), dtype=np.float64) for name, t in radar_transforms.items()}
        self.half_length = half_length
        self.half_width = half_width
        self.half_height = half_height
        self.carla_map = CarlaDataProvider.get_map()
        self._cur_state = None
        self._cur_step = None
        self._static: list[trackedbb] = []
        self._pts = None
        self._done = threading.Event()
        self._done.set()

    def push_frame(self, state, lidar_np: np.ndarray, radar_dict: dict, cameras: np.ndarray, step):
        self._cur_state, self._cur_step = state, step
        radar_arrays = {name: radar_measurement_to_array(m) for name, m in radar_dict.items()}
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        self.input_queue.put((state, lidar_np, radar_arrays, cameras, step))

    def get_latest_result(self) -> Optional[Tuple[list[trackedbb], np.ndarray, float, np.ndarray]]:
        while True:                                   # detections -> UKF updates, in capture order
            try:
                det_step, dets, statics, pts = self.output_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self.tracker.update(dets, det_step)
            except Exception:
                traceback.print_exc()
                sys.stderr.flush()
            self._static, self._pts = statics, pts
        if self._cur_state is None:
            return None
        tracked = self.tracker.predict_to(self._cur_step) + self._static   # extrapolated to NOW
        ttcs, min_ttc = compute_ttcs(tracked, self._cur_state, self.carla_map,
                                     self.half_length, self.half_width, self.half_height)
        return tracked, ttcs, min_ttc, self._pts

    def run(self):
        while self.running:
            try:
                (state, lidar_np, radar_dict, cameras, step) = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                out = detect_from_sensors(state, lidar_np, radar_dict, cameras, step,
                                          self.radar_transforms, self.half_length, self.half_width,
                                          self.detector, self.lidar_offset)
                if out is not None:
                    dets, statics, pts = out
                    self.output_queue.put((step, dets, statics, pts))
            except Exception:
                traceback.print_exc()
                sys.stderr.flush()
            finally:
                self._done.set()

    def stop(self):
        self.running = False
