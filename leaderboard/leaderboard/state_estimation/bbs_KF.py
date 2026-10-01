import math
import os
import numpy as np
from typing import Optional, Tuple
import carla
import threading
import queue
from leaderboard.leaderboard.state_estimation.sensor_dataclasses import bev_Det, trackedbb, CalculatedState, Ego3DState
from leaderboard.leaderboard.state_estimation.TwoDimTTC import TTC
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from UKF_bb import ObjectUKF
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from leaderboard.leaderboard.state_estimation.UKF_bb import UKFBoundingBoxTracker
import hyperparams as hp

class _DummyLogger:
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass

class _SingleFrameDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training: bool = False):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training)
        self._points = np.zeros((0, 4), dtype=np.float32)
        self._camera_imgs = None
        self._camera_intrinsics = None
        self._camera2lidar = None

    def __len__(self):  return 1

    def __getitem__(self, index): 
        data_dict = {
            "points": self._points,
            "frame_id": 0,
            "camera_imgs": self._camera_imgs,            
            "camera_intrinsics": self._camera_intrinsics,
            "camera2lidar": self._camera2lidar           
        }
        return self.prepare_data(data_dict=data_dict)

    def load_data(self, points, imgs):
        self._points = points
        self._camera_imgs = imgs
        self._camera_intrinsics = hp.intrinsics
        self._camera2lidar = hp.extrinsics

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

        self.class_names = cfg.CLASS_NAMES
        self.dataset = _SingleFrameDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=self.class_names)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=self.dataset)
        self.model.load_params_from_file(filename=ckpt_file, logger=_DummyLogger(), to_cpu=(device == "cpu"))
        self.device = device
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def infer(self, points_xyzi: np.ndarray, imgs: np.ndarray) -> list[dict]:
        self.dataset.load_data(points_xyzi.astype(np.float32), imgs)
        data_dict = self.dataset.collate_batch([self.dataset[0]])
        load_data_to_gpu(data_dict)
        pred_dicts, _ = self.model.forward(data_dict)
        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].cpu().numpy()
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy()

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
            if box.shape[0] >= 9:
                entry["vx"], entry["vy"] = float(box[7]), float(box[8])
            results.append(entry)
        return results

def ego_to_world_matrix(ego_state) -> np.ndarray:
    return np.array(carla.Transform(
        carla.Location(x=ego_state.x, y=ego_state.y, z=ego_state.z),
        carla.Rotation(pitch=math.degrees(ego_state.pitch_rad),
                        yaw=math.degrees(ego_state.yaw_rad),
                        roll=math.degrees(ego_state.roll_rad))).get_matrix())

def _points_in_rotated_box(points_xyz: np.ndarray, center_x: float, center_y: float, center_z: float,
                            length: float, width: float, height: float, yaw_deg: float, margin: float = 0.1) -> np.ndarray:
    boxes = np.array([[center_x, center_y, center_z, length + (margin * 2), width + (margin * 2), height + (margin * 2), np.deg2rad(yaw_deg)]])
    points_tensor = torch.from_numpy(points_xyz).float()
    boxes_tensor = torch.from_numpy(boxes).float()
    box_point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(boxes_tensor, points_tensor)
    return box_point_masks[0].numpy() > 0

def _match_radar_points(radar_points: np.ndarray, center_x: float, center_y: float, center_z: float,
                         length: float, width: float, height: float, yaw_deg: float, margin: float = 1.0) -> np.ndarray:
    if not radar_points.size:
        return np.array([])
    min_x, max_x = center_x - length / 2 - margin, center_x + length / 2 + margin
    min_y, max_y = center_y - width / 2 - margin, center_y + width / 2 + margin
    min_z, max_z = center_z - height / 2 - margin, center_z + height / 2 + margin
    coarse_mask = (radar_points[:, 0] >= min_x) & (radar_points[:, 0] <= max_x) & \
        (radar_points[:, 1] >= min_y) & (radar_points[:, 1] <= max_y) &\
        (radar_points[:, 2] >= min_z) & (radar_points[:, 2] <= max_z)
    candidate_points = radar_points[coarse_mask]
    if len(candidate_points) == 0:
        return np.array([])
    inside = _points_in_rotated_box(candidate_points[:, :3], center_x, center_y, center_z, length, width, height, yaw_deg, margin)
    return candidate_points[inside]

def _vertical_overlap(z_a: float, h_a: float, z_b: float, h_b: float) -> bool:
    a_lo, a_hi = z_a - h_a / 2.0, z_a + h_a / 2.0
    b_lo, b_hi = z_b - h_b / 2.0, z_b + h_b / 2.0
    return a_lo <= b_hi and b_lo <= a_hi

def time_to_road_departure(carla_map: carla.Map, ego: Ego3DState,
                            dims: np.array, dt: float = hp.ROAD_EXIT_STEP_S,
                            max_time: float = hp.ROAD_EXIT_LOOKAHEAD_S) -> float:    
    x = np.array([ego.x, ego.y, ego.vx_world, ego.vy_world, ego.long_accel, ego.lat_accel, ego.yaw, ego.yaw_rate, dims[0],dims[1], ego.z, dims[2]])

    steps = max(1, int(max_time / dt))
    for i in range(steps + 1):
        t = i * dt
        if i > 0:
            x = ObjectUKF.fx(x, dt)

        px, py, pz = x[0], x[1], x[9]
        wp = carla_map.get_waypoint(carla.Location(x=px, y=py, z=pz),
                                     project_to_road=False, lane_type=carla.LaneType.Driving)
        if wp is None:
            return -1.0 if i == 0 else t

        wp_loc = wp.transform.location
        right = wp.transform.get_right_vector()
        lateral = (px - wp_loc.x) * right.x + (py - wp_loc.y) * right.y
        if abs(lateral) + dims[1]/2 > wp.lane_width / 2.0:
            return -1.0 if i == 0 else t

        x[9] = wp_loc.z

    return float('inf')

def build_bbs(half_length, half_width, state: CalculatedState, master_cloud: np.ndarray,
              radar_points, cameras, step, tracker: UKFBoundingBoxTracker,
              detector: OpenPCDetDetector) -> list[trackedbb]:
    
    obstacle_mask = (master_cloud[:, 0] > -hp.MAX_LOCKBACK) & (master_cloud[:, 0] < hp.MAX_LOOKFORWARD) & \
        (np.abs(master_cloud[:, 1]) < hp.MAX_LOOKSIDEWAYS) & (master_cloud[:, 2] > -hp.GROUND_CLEARANCE) & \
        ((np.abs(master_cloud[:, 0]) > half_length) | (np.abs(master_cloud[:, 1]) > half_width))
    obstacle_pts = master_cloud[obstacle_mask]

    if len(obstacle_pts) < hp.MIN_NUMBER_LIDAR_HITS:
        moving_tracks = tracker.update([], step)
        return [t.get_all_data() for t in moving_tracks]

    T = ego_to_world_matrix(state.ego_state)    
    network_detections: list[bev_Det] = []
    network_static: list[trackedbb] = []
    for box in detector.infer(obstacle_pts, cameras):
        class_name = box["class_name"].strip().lower()
        if class_name not in hp.OPENPCDET_DYNAMIC_CLASSES and class_name not in hp.OPENPCDET_STATIC_CLASSES:  
            continue 
        center_h = np.array([box["x"], box["y"], box["z"], 1.0])
        center_global = T @ center_h
        heading_dir_global = T[:3, :3] @ np.array(
            [math.cos(box["heading"]), math.sin(box["heading"]), 0.0])
        center_x, center_y, center_z = float(center_global[0]), float(center_global[1]), float(center_global[2])
        yaw_deg = math.degrees(math.atan2(heading_dir_global[1], heading_dir_global[0]))
        length, width, height, vx, vy = box["length"], box["width"], box["height"], box["vx"], box["vy"]
        if class_name in hp.OPENPCDET_DYNAMIC_CLASSES:
            matched_radars = _match_radar_points(radar_points, center_x, center_y, center_z, length, width, height, yaw_deg)
            network_detections.append(bev_Det(
                center_x=center_x, center_y=center_y, length=length, width=width, vx=vx, vy=vy,
                yaw=yaw_deg, step=step, center_z=center_z, height=height, radar_points=matched_radars
            ))
        else:
            network_static.append(trackedbb(
                px=center_x, py=center_y, moveable=False,
                vx=0.0, vy=0.0, speed=0.0, acceleration=0.0, ax=0.0, ay=0.0,
                yaw=math.radians(yaw_deg), yaw_rate=0.0,
                len=length, wid=max(width, 0.3),
                z=center_z, height=height,
                pos_history=[], slip_angle_deg=0.0
            ))

    moving_tracks = tracker.update(network_detections, step)
    tracked: list[trackedbb] = [t.get_all_data() for t in moving_tracks]
    tracked.extend(network_static)
    return tracked

def process_radar(state: CalculatedState, radar_data: carla.RadarMeasurement, sensor_transform: np.ndarray) -> np.ndarray:
    if GameTime.get_time() < 0.5:
        return np.empty((0, 7), dtype=np.float32)
    e = state.ego_state
    T_ego_to_world = ego_to_world_matrix(state.ego_state)
    T_sensor_to_world = T_ego_to_world @ sensor_transform
    R_sensor_to_world = T_sensor_to_world[:3, :3]
    v_ego_world = np.array([e.vx_world, e.vy_world, 0], dtype=np.float32)

    parsed_points = []
    for det in radar_data:
        r, az, alt, v_r = det[0], det[1], det[2], det[3]
        cos_alt = math.cos(alt)

        p_sensor = np.array([r * cos_alt * math.cos(az), r * cos_alt * math.sin(az), r * math.sin(alt), 1.0], dtype=np.float32)
        p_world = (T_sensor_to_world @ p_sensor)[:3]

        u_sensor = np.array([cos_alt * math.cos(az), cos_alt * math.sin(az), math.sin(alt)], dtype=np.float32)
        u_world = R_sensor_to_world @ u_sensor

        v_ego_radial = np.dot(v_ego_world, u_world)
        v_r_abs = v_r + v_ego_radial
        parsed_points.append([p_world[0], p_world[1], p_world[2], v_r_abs, u_world[0], u_world[1], u_world[2]])
    if not parsed_points:
        return np.empty((0, 7), dtype=np.float32)
    return np.array(parsed_points, dtype=np.float32)

def get_lidar_radar_info(state: CalculatedState, lidar_stream, radar_streams, cameras, step,
                        radar_transforms, half_length, half_width, tracker, carla_map: carla.Map,
                        detector: OpenPCDetDetector, half_height: float = hp.DEFAULT_EGO_HALF_HEIGHT):
    radar_points = np.ndarray((0, 7))
    for sensor_name, radar_data in radar_streams.items():
        radar_points = np.vstack([radar_points, process_radar(state, radar_data, radar_transforms[sensor_name])])

    # pts = np.ndarray((0, 4))
    pts_sensor = np.reshape(lidar_stream, (int(len(lidar_stream) / 4), 4))
    ranges = np.linalg.norm(pts_sensor[:, :3], axis=1)
    pts_sensor = pts_sensor[ranges>hp.MIN_VALID_RANGE_M]
    pts_sensor[:, 1] = -pts_sensor[:, 1]
    #num_pts = len(pts_sensor)
    #xyz_homo = np.hstack((pts_sensor[:, :3], np.ones((num_pts, 1))))
    #xyz_ego = (lidar_transform @ xyz_homo.T).T[:, :3]
    #pts = np.vstack((pts, np.column_stack((xyz_ego, pts_sensor[:, 3]))))

    #print("points : ",len(pts),flush=True)
    
    tracked: list[trackedbb] = build_bbs(half_length=half_length, half_width=half_width,
                    state=state, master_cloud=pts_sensor, radar_points=radar_points,
                    cameras=cameras, step=step, tracker=tracker, detector=detector)

    if state.ego_state.speed > 1e-3:
        hx = state.ego_state.vx_world / state.ego_state.speed
        hy = state.ego_state.vy_world / state.ego_state.speed
    else:
        hx = math.cos(state.ego_state.yaw)
        hy = math.sin(state.ego_state.yaw)
    ego_ttc = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.vx_world, state.ego_state.vy_world, hx, hy, half_length * 2, half_width * 2])

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
                                np.array([half_length*2, half_width*2, half_height*2]))
    ttcs = np.append(ttcs, road_exit_ttc)
    min_ttc = float(np.min(ttcs))

    return tracked, ttcs, min_ttc

class BBsWorker(threading.Thread):
    def __init__(self, lidar_transform, radar_transforms, half_length, half_width,
                 half_height: float = hp.DEFAULT_EGO_HALF_HEIGHT):
        super().__init__()
        self.tracker = UKFBoundingBoxTracker()
        try:
            self.detector = OpenPCDetDetector()
        except Exception as exc:
            print(f"OpenPCDetDetector unavailable: {exc}", flush=True)
            exit(1)
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        self.running = True
        self.daemon = True
        self.lidar_transform = lidar_transform
        self.radar_transforms = radar_transforms
        self.half_length = half_length
        self.half_width = half_width
        self.half_height = half_height
        self.carla_map = CarlaDataProvider.get_map()

    def push_frame(self, state, lidar_np: np.ndarray, radar_dict: dict, cameras: np.array, step):
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        self.input_queue.put((state, lidar_np, radar_dict, cameras, step))

    def get_latest_result(self) -> Optional[Tuple[list[trackedbb], np.ndarray, float]]:
        latest = None
        while not self.output_queue.empty():
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def run(self):
        while self.running:
            try:
                (state, lidar_np, radar_dict, cameras, step) = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            result = get_lidar_radar_info(state, lidar_np, radar_dict, cameras, step,
                                           self.radar_transforms, self.half_length, self.half_width, self.tracker,
                                           self.carla_map, self.detector, self.half_height)
            self.output_queue.put(result)

    def stop(self):
        self.running = False
