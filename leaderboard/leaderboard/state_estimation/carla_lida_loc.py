from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import carla
from scipy.spatial.transform import Rotation

from leaderboard.state_estimation.sensor_dataclasses import SlamUpdateResult, Keyframe, SlamOutput
import leaderboard.state_estimation.hyperparams as hp



def translation_cov_shape(info6: np.ndarray, A_R: np.ndarray, cap: float) -> np.ndarray:
    """
    Turn the Open3D ICP information matrix into a *shape* for the translation covariance.
    Open3D orders the 6-vector as [rotation(3), translation(3)] and expresses it in the
    target (keyframe/submap) frame. The best-constrained direction gets factor 1, weakly
    constrained directions (e.g. along a straight road) are inflated up to `cap`.
    The result is rotated into the current vehicle frame (A_R maps current -> target).
    """
    try:
        lam, V = np.linalg.eigh(0.5 * (info6[3:6, 3:6] + info6[3:6, 3:6].T))
    except np.linalg.LinAlgError:
        return np.eye(3)
    lam = np.maximum(lam, 0.0)
    lmax = float(lam.max())
    if lmax <= 1e-9:
        return np.eye(3) * cap
    s = lmax / np.maximum(lam, lmax / cap)
    shape_target = (V * s) @ V.T
    return A_R.T @ shape_target @ A_R


class CarlaPoseGraphSLAM:
    def __init__(self, initial_transform: np.ndarray, voxel_size: float = 0.5,
                 normal_radius: float = 2.0, normal_max_nn: int = 30,
                 extrinsic_R: Optional[np.ndarray] = None, extrinsic_t: Optional[np.ndarray] = None):
        self.initial_transform = initial_transform.astype(float)
        self.voxel_size = voxel_size
        self._normal_params = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
        self.R_ext = np.eye(3) if extrinsic_R is None else np.asarray(extrinsic_R, float)
        self.t_ext = np.zeros(3) if extrinsic_t is None else np.asarray(extrinsic_t, float)

        self.keyframes: List[Keyframe] = []
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self._odometry_graph = np.eye(4)

        self.current_pose_world = self.initial_transform.copy()
        self._last_keyframe_world_pose = self.initial_transform.copy()

        self.map_epoch = 0
        self.last_extras = {"shape": np.eye(3), "epoch": 0}
        self._submap: Optional[o3d.geometry.PointCloud] = None

        reg = o3d.pipelines.registration
        self._estimator = reg.TransformationEstimationPointToPlane()
        if hasattr(reg, "TukeyLoss"):
            self._estimator = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=hp.ICP_ROBUST_K))

    @staticmethod
    def _parse_raw_lidar(lidar_data: carla.LidarMeasurement) -> np.ndarray:
        raw = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        pts = np.reshape(raw, (-1, 4))
        xyz = pts[:, 0:3].astype(float)
        xyz[:, 1] = -xyz[:, 1]  # CARLA LH -> EKF FLU RH
        return xyz

    def _filter_static(self, xyz_sensor: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(xyz_sensor, axis=1)                     
        xyz = xyz_sensor[(r > hp.SELF_HIT_RANGE) & (r < hp.MAX_SENSOR_RANGE)]
        return xyz @ self.R_ext.T + self.t_ext

    def _prepare(self, pcd):
        pcd.estimate_normals(search_param=self._normal_params)
        self._estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        return pcd

    def _make_pcd(self, lidar_data) -> Optional[o3d.geometry.PointCloud]:
        xyz = self._filter_static(self._parse_raw_lidar(lidar_data))
        if xyz.shape[0] < 50:
            return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return self._prepare(pcd)

    def _icp(self, source, target, max_dist, init, max_iter):
        reg = o3d.pipelines.registration
        return reg.registration_icp(source, target, max_dist, init, self._estimator,
                                    reg.ICPConvergenceCriteria(max_iteration=max_iter))

    def _pairwise_registration(self, source, target, init: np.ndarray):
        coarse = self._icp(source, target, hp.EDGE_CORR_DIST_COARSE, init, 30)
        fine = self._icp(source, target, hp.EDGE_CORR_DIST_FINE, coarse.transformation, 30)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, hp.EDGE_CORR_DIST_FINE, fine.transformation)
        return fine.transformation, np.asarray(info), fine.fitness, fine.inlier_rmse

    def _try_loop_closure(self, new_id: int) -> bool:
        current_pos = self.keyframes[new_id].world_position_hint
        current_pcd = self.keyframes[new_id].pcd
        closed_any = False
        for kf in self.keyframes[: max(0, new_id - hp.LOOP_MIN_KEYFRAME_GAP)]:
            if np.linalg.norm(kf.world_position_hint - current_pos) > hp.LOOP_SEARCH_RADIUS_M:
                continue
            init_k_to_j = (np.linalg.inv(self.pose_graph.nodes[new_id].pose) @ self.pose_graph.nodes[kf.id].pose)
            T, info, fitness, rmse = self._pairwise_registration(kf.pcd, current_pcd, init_k_to_j)
            if fitness < hp.LOOP_FITNESS_MIN or rmse > hp.LOOP_RMSE_MAX:
                continue
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(kf.id, new_id, T, info, uncertain=True))
            closed_any = True
        return closed_any

    def _rebuild_submap(self):
        """Merge the last N keyframes into the frame of the newest keyframe (scan-to-submap)."""
        j = len(self.keyframes) - 1
        Pj_inv = np.linalg.inv(self.pose_graph.nodes[j].pose)
        merged = o3d.geometry.PointCloud()
        for i in range(max(0, j - hp.SUBMAP_KEYFRAMES + 1), j + 1):
            pc = o3d.geometry.PointCloud(self.keyframes[i].pcd)
            pc.transform(Pj_inv @ self.pose_graph.nodes[i].pose)
            merged += pc
        merged = merged.voxel_down_sample(voxel_size=self.voxel_size)
        self._submap = self._prepare(merged)

    def _target(self):
        return self._submap if self._submap is not None else self.keyframes[-1].pcd

    def _insert_keyframe(self, current_pcd, A: np.ndarray) -> bool:
        j = len(self.keyframes)
        prev_kf = self.keyframes[-1]

        T, info, _, _ = self._pairwise_registration(prev_kf.pcd, current_pcd, np.linalg.inv(A))
        self._odometry_graph = T @ self._odometry_graph
        new_node_pose = np.linalg.inv(self._odometry_graph)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(new_node_pose))
        self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(j - 1, j, T, info, uncertain=False))

        new_kf_world_pos = (self.initial_transform @ new_node_pose)[0:3, 3]
        self.keyframes.append(Keyframe(id=j, pcd=current_pcd, world_position_hint=new_kf_world_pos))
        loop_closed = self._try_loop_closure(j)
        if loop_closed:
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=hp.EDGE_CORR_DIST_FINE,
                edge_prune_threshold=hp.GLOBAL_OPT_EDGE_PRUNE_THRESHOLD,
                reference_node=0)
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
            self.map_epoch += 1          # SLAM frame jumped: relative fusion must not bridge this

        optimized_pose_local = self.pose_graph.nodes[j].pose
        self._last_keyframe_world_pose = self.initial_transform @ optimized_pose_local
        self.current_pose_world = self._last_keyframe_world_pose.copy()
        if loop_closed:
            for kf in self.keyframes:
                kf.world_position_hint = (self.initial_transform @ self.pose_graph.nodes[kf.id].pose)[0:3, 3]
        self._rebuild_submap()
        return loop_closed

    def _insert_first_keyframe(self, pcd) -> None:
        self.keyframes.append(Keyframe(id=0, pcd=pcd, world_position_hint=self.initial_transform[0:3, 3]))
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        self._odometry_graph = np.eye(4)
        self._last_keyframe_world_pose = self.initial_transform.copy()
        self.current_pose_world = self.initial_transform.copy()
        self._rebuild_submap()

    @staticmethod
    def _pose_to_pq(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = pose[0:3, 3]
        x, y, z, w = Rotation.from_matrix(pose[0:3, 0:3]).as_quat()
        return p, np.array([w, x, y, z], dtype=float)

    def process_lidar(self, lidar_data, predicted_pose: Optional[np.ndarray] = None) -> SlamUpdateResult:
        current_pcd = self._make_pcd(lidar_data)
        if current_pcd is None:
            return SlamUpdateResult(success=False, reason="too few points after filtering")

        if not self.keyframes:
            self._insert_first_keyframe(current_pcd)
            p, q = self._pose_to_pq(self.current_pose_world)
            self.last_extras = {"shape": np.eye(3), "epoch": self.map_epoch}
            return SlamUpdateResult(success=True, p_slam=p, q_slam=q, num_keyframes=1)

        target = self._target()
        init = np.linalg.inv(self._last_keyframe_world_pose) @ (
            predicted_pose if predicted_pose is not None else self.current_pose_world)
        track = self._icp(current_pcd, target, hp.TRACK_CORR_DIST, init, 25)

        if track.fitness < hp.TRACK_FITNESS_MIN or track.inlier_rmse > hp.TRACK_RMSE_MAX:
            return SlamUpdateResult(success=False, fitness=track.fitness, inlier_rmse=track.inlier_rmse,
                                    num_keyframes=len(self.keyframes), reason="tracking ICP quality below threshold")

        A = track.transformation        # current_local -> last_keyframe_local
        self.current_pose_world = self._last_keyframe_world_pose @ A
        info = np.asarray(o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            current_pcd, target, hp.TRACK_CORR_DIST, A))
        shape = translation_cov_shape(info, A[0:3, 0:3], hp.SLAM_INFO_COND_CAP)

        loop_closed = False
        if (float(np.linalg.norm(A[0:3, 3])) > hp.KEYFRAME_TRANSLATION_M or
                float(np.degrees(Rotation.from_matrix(A[0:3, 0:3]).magnitude())) > hp.KEYFRAME_ROTATION_DEG):
            loop_closed = self._insert_keyframe(current_pcd, A)

        self.last_extras = {"shape": shape, "epoch": self.map_epoch}
        p, q = self._pose_to_pq(self.current_pose_world)
        return SlamUpdateResult(success=True, p_slam=p, q_slam=q, fitness=track.fitness,
                                inlier_rmse=track.inlier_rmse, info_matrix=info,
                                loop_closed=loop_closed, num_keyframes=len(self.keyframes))


class AsyncSlamWorker(threading.Thread):
    def __init__(self, slam_engine: CarlaPoseGraphSLAM):
        super().__init__()
        self.slam = slam_engine
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        self.running = True
        self.daemon = True

    def push_frame(self, lidar_data, predicted_pose: np.ndarray, frame_id: int = 0):
        item = (lidar_data, predicted_pose, frame_id)
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()          # drop stale frame
            except queue.Empty:
                pass
        self.input_queue.put(item)

    def get_latest_result(self) -> Optional[SlamOutput]:
        latest = None
        while not self.output_queue.empty():
            try:
                latest = self.output_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def _process(self, item):
        lidar_data, predicted_pose, frame_id = item
        res = self.slam.process_lidar(lidar_data, predicted_pose=predicted_pose)
        if res.success:
            ex = self.slam.last_extras
            self.output_queue.put(SlamOutput(
                frame_id=frame_id, p=np.asarray(res.p_slam, float), q=np.asarray(res.q_slam, float),
                trans_shape=ex["shape"], map_epoch=ex["epoch"],
                fitness=float(res.fitness or 0.0), inlier_rmse=float(res.inlier_rmse or 0.0)))

    def run(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process(item)

    def stop(self):
        self.running = False
