from typing import Optional, List, Tuple
import threading
import queue
import numpy as np
import open3d as o3d
import carla
from scipy.spatial.transform import Rotation
from leaderboard.state_estimation.sensor_dataclasses import SlamUpdateResult, Keyframe
import leaderboard.state_estimation.hyperparams as hp

class CarlaPoseGraphSLAM:
    def __init__(self, initial_transform: np.ndarray, voxel_size: float = 0.5,
            normal_radius: float = 2.0, normal_max_nn: int = 30,):
        self.initial_transform = initial_transform.astype(float)
        self.voxel_size = voxel_size
        self._normal_params = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)

        self.keyframes: List[Keyframe] = []
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self._odometry_graph = np.eye(4)   # accumulator, mirrors Open3D tutorial pattern

        self.current_pose_world = self.initial_transform.copy()
        self._last_keyframe_world_pose = self.initial_transform.copy()

    @staticmethod
    def _parse_raw_lidar(lidar_data: carla.LidarMeasurement) -> np.ndarray:
        raw = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        pts = np.reshape(raw, (-1, 4))
        xyz = pts[:, 0:3].copy()
        xyz[:, 1] = -xyz[:, 1]  # CARLA LH -> EKF FLU RH
        return xyz

    def _filter_static(self, xyz: np.ndarray, bbs: Optional[np.ndarray]) -> np.ndarray:
        r = np.linalg.norm(xyz, axis=1)
        keep = (r > hp.SELF_HIT_RANGE) & (r < hp.MAX_SENSOR_RANGE)
        if bbs is not None: pass # remove the known bounding boxes
        return xyz[keep]

    def _make_pcd(self, lidar_data) -> Optional[o3d.geometry.PointCloud]:
        xyz = self._filter_static(self._parse_raw_lidar(lidar_data), None)
        if xyz.shape[0] < 50:   return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pcd.estimate_normals(search_param=self._normal_params)
        return pcd

    def _pairwise_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, init: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        coarse = o3d.pipelines.registration.registration_icp(
            source, target, hp.EDGE_CORR_DIST_COARSE, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        
        fine = o3d.pipelines.registration.registration_icp(
            source, target, hp.EDGE_CORR_DIST_FINE, coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, hp.EDGE_CORR_DIST_FINE, fine.transformation)
        return fine.transformation, np.asarray(info), fine.fitness, fine.inlier_rmse

    def _try_loop_closure(self, new_id: int) -> bool:
        current_pos = self.keyframes[new_id].world_position_hint
        current_pcd = self.keyframes[new_id].pcd
        closed_any = False
        for kf in self.keyframes[: max(0, new_id - hp.LOOP_MIN_KEYFRAME_GAP)]:
            if np.linalg.norm(kf.world_position_hint - current_pos) > hp.LOOP_SEARCH_RADIUS_M: continue
            init_k_to_j = (np.linalg.inv(self.pose_graph.nodes[new_id].pose) @ self.pose_graph.nodes[kf.id].pose)
            transformation_icp, information_icp, fitness, rmse = self._pairwise_registration(kf.pcd, current_pcd, init_k_to_j)
            if fitness < hp.LOOP_FITNESS_MIN or rmse > hp.LOOP_RMSE_MAX:  continue
            self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(kf.id, new_id, transformation_icp, information_icp, uncertain=True))
            closed_any = True # Don't break -- multiple simultaneous loop edges only help the optimizer, and candidates are cheap to check with the radius gate.
        return closed_any

    def _insert_keyframe(self, current_pcd: o3d.geometry.PointCloud, A: np.ndarray) -> bool:
        j = len(self.keyframes)              
        prev_kf = self.keyframes[-1]

        transformation_icp, information_icp, _, _ = self._pairwise_registration(
            prev_kf.pcd, current_pcd, np.linalg.inv(A)) # A maps j->i, so inverse maps i->j

        self._odometry_graph = transformation_icp @ self._odometry_graph
        new_node_pose_graph_local = np.linalg.inv(self._odometry_graph)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(new_node_pose_graph_local))
        self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(j-1, j, transformation_icp, information_icp, uncertain=False))

        new_kf_world_pos = (self.initial_transform @ new_node_pose_graph_local)[0:3, 3]
        self.keyframes.append(Keyframe(id=j, pcd=current_pcd, world_position_hint=new_kf_world_pos))
        loop_closed = self._try_loop_closure(j)
        if loop_closed:
            option = o3d.pipelines.registration.GlobalOptimizationOption(
                        max_correspondence_distance=hp.EDGE_CORR_DIST_FINE,
                        edge_prune_threshold=hp.GLOBAL_OPT_EDGE_PRUNE_THRESHOLD,
                        reference_node=0)
            o3d.pipelines.registration.global_optimization(self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)

        # Update live pose / keyframe anchor from the (possibly just-optimized) graph.
        optimized_pose_local = self.pose_graph.nodes[j].pose
        self._last_keyframe_world_pose = self.initial_transform @ optimized_pose_local
        self.current_pose_world = self._last_keyframe_world_pose.copy()

        if loop_closed:
            for kf in self.keyframes:
                kf.world_position_hint = (self.initial_transform @ self.pose_graph.nodes[kf.id].pose)[0:3, 3]

        return loop_closed

    def _insert_first_keyframe(self, pcd: o3d.geometry.PointCloud) -> None:
        self.keyframes.append(Keyframe(id=0, pcd=pcd, world_position_hint=self.initial_transform[0:3, 3]))
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        self._odometry_graph = np.eye(4)
        self._last_keyframe_world_pose = self.initial_transform.copy()
        self.current_pose_world = self.initial_transform.copy()

    @staticmethod
    def _pose_to_pq(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = pose[0:3, 3]
        scipy_quat = Rotation.from_matrix(pose[0:3, 0:3]).as_quat()  # [x, y, z, w]
        q = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]], dtype=float)
        return p, q

    def process_lidar(self, lidar_data: carla.LidarMeasurement, predicted_pose: Optional[np.ndarray] = None) -> SlamUpdateResult:
        current_pcd = self._make_pcd(lidar_data)
        if current_pcd is None: return SlamUpdateResult(success=False, reason="too few points after filtering")

        if not self.keyframes:
            self._insert_first_keyframe(current_pcd)
            p, q = self._pose_to_pq(self.current_pose_world)
            return SlamUpdateResult(success=True, p_slam=p, q_slam=q, num_keyframes=1)

        last_kf = self.keyframes[-1]
        track = o3d.pipelines.registration.registration_icp(
            source=current_pcd,
            target=last_kf.pcd,
            max_correspondence_distance=hp.TRACK_CORR_DIST,
            init=np.linalg.inv(self._last_keyframe_world_pose) @ (predicted_pose if predicted_pose is not None else self.current_pose_world),  # curr_local -> kf_local guess,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
        
        if track.fitness < hp.TRACK_FITNESS_MIN or track.inlier_rmse > hp.TRACK_RMSE_MAX:
            return SlamUpdateResult(success=False, fitness=track.fitness, inlier_rmse=track.inlier_rmse,
                num_keyframes=len(self.keyframes), reason="tracking ICP quality below threshold")

        A = track.transformation    # maps current_local -> last_keyframe_local.
        self.current_pose_world = self._last_keyframe_world_pose @ A
        loop_closed = False
        if float(np.linalg.norm(A[0:3, 3])) > hp.KEYFRAME_TRANSLATION_M or float(np.degrees(Rotation.from_matrix(A[0:3, 0:3]).magnitude())) > hp.KEYFRAME_ROTATION_DEG:
            loop_closed = self._insert_keyframe(current_pcd, A)

        p, q = self._pose_to_pq(self.current_pose_world)
        return SlamUpdateResult(
            success=True, p_slam=p, q_slam=q,
            fitness=track.fitness, inlier_rmse=track.inlier_rmse,
            info_matrix=np.asarray(o3d.pipelines.registration.get_information_matrix_from_point_clouds(current_pcd, last_kf.pcd, hp.TRACK_CORR_DIST, A)),
            loop_closed=loop_closed,
            num_keyframes=len(self.keyframes))

class AsyncSlamWorker(threading.Thread):
    def __init__(self, slam_engine):
        super().__init__()
        self.slam = slam_engine
        self.input_queue = queue.Queue(maxsize=1) 
        self.output_queue = queue.Queue()           
        self.running = True
        self.daemon = True                          

    def push_frame(self, lidar_data: carla.LidarMeasurement, predicted_pose: np.ndarray):
        if self.input_queue.full():
            try:    self.input_queue.get_nowait() # Remove stale frame
            except queue.Empty: pass
        self.input_queue.put((lidar_data, predicted_pose))

    def get_latest_result(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        latest = None
        while not self.output_queue.empty():
            try:    latest = self.output_queue.get_nowait()
            except queue.Empty: break
        return latest

    def run(self):
        while self.running:
            try:    lidar_data, predicted_pose = self.input_queue.get(timeout=0.1)
            except queue.Empty: continue
            result = self.slam.process_lidar(lidar_data, predicted_pose=predicted_pose)
            if result.success:  self.output_queue.put((result.p_slam, result.q_slam))

    def stop(self): self.running = False

