import math
import carla
import numpy as np
import time
#from leaderboard_codes.metrics_calc_with_sensors.ego_UKF import EgoUnscentedKalmanFilterStateEstimator
#from leaderboard_codes.metrics_calc_with_sensors.ego_EKF import EgoExtendedKalmanFilterStateEstimator
from leaderboard.state_estimation.ego_EKF_3D import  EgoErrorStateEKF3D
from leaderboard.state_estimation.sensor_dataclasses import CalculatedState
from leaderboard.state_estimation.bbs_KF import BBsWorker
from leaderboard.state_estimation.real_test_BBs import OnlineEvaluator
from leaderboard.state_estimation.traffic_light_worker import TrafficLightWorker
from leaderboard.state_estimation.real_test_ego import EgoStateOnlineEvaluator
Gravity = 9.81
min_valid_range_m = 0.5

class CarlaStateEstimatorEngine:
    """
        Computes derived vehicle state from raw CARLA sensor data provided on each call.
    """
    _last_state: CalculatedState | None = None

    def __init__(
            self,
            vehicle: carla.Vehicle,
            world: carla.World,
            world_map: carla.Map, 
            lidar_transform: carla.Transform,
            radar_transforms: dict[str, carla.Transform] = None):
        self.map = world_map
        bb = vehicle.bounding_box
        self.ego_to_world_transform = np.array(vehicle.get_transform().get_matrix(), dtype=np.float32)
        loc = bb.location
        self.z = loc.z
        ext = bb.extent
        self.half_width = ext.y
        self.half_length = ext.x
        self.state = CalculatedState()
        type(self)._last_state = self.state
        self.tlw = TrafficLightWorker()
        self.ekf = EgoErrorStateEKF3D(vehicle, world_map)
        self.tracker = BBsWorker(lidar_transform, radar_transforms, self.half_length, self.half_width, ext.z)
        self.radar_transforms : dict[str, np.array] = {}
        if radar_transforms is None:
            raise ValueError("RADAR transforms must be provided for each RADAR sensor.")
        for name, transform in radar_transforms.items():
            self.radar_transforms[name] = np.array(transform.get_matrix())

        self.lidar_transform = np.array(carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + ext.z + 0.2),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)).get_matrix())
        self.bbs_evaluator = OnlineEvaluator(world=world, ego_actor_id=vehicle.id)
        self.ego_evaluator = EgoStateOnlineEvaluator(world, vehicle.id)
        self.tlw.start()
        self.tracker.start()
        self._last_result = None

    def get_rest_info(self, state : CalculatedState):
        cr, sr = math.cos(state.ego_state.roll_rad), math.sin(state.ego_state.roll_rad)
        cp, sp = math.cos(state.ego_state.pitch_rad), math.sin(state.ego_state.pitch_rad)
        cy, sy = math.cos(state.ego_state.yaw_rad), math.sin(state.ego_state.yaw_rad)
        state.forward_vector = np.array([cy * cp, sy * cp, -sp])
        state.right_vector = np.array([
            cy * sp * sr + sy * cr,
            sy * sp * sr - cy * cr,
            cp * sr])
        state.up_vector = np.array([
            cy * sp * cr - sy * sr,
            sy * sp * cr + cy * sr,
            cp * cr])
        
        state.g_force_longitudinal =  state.ego_state.long_accel / Gravity
        state.g_force_lateral = state.ego_state.lat_accel / Gravity

        waypoint = self.map.get_waypoint(
            location= carla.Location(state.ego_state.x,state.ego_state.y,self.z), 
            project_to_road=True, 
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            state.road_id = -1
            state.lane_id = -1
            state.lane_width_m = 0.0
            state.cross_track_error_m = float('inf')
            state.heading_error_deg = 0.0
            state.dist_nxt_junc = float('inf')
            return

        state.road_id = waypoint.road_id
        state.lane_id = waypoint.lane_id
        state.lane_width_m = waypoint.lane_width
        state.speed_limit_ms = self.speed_limit / 3.6

        wp_trans = waypoint.transform
        wp_dir = wp_trans.get_forward_vector()

        disp = np.array([state.ego_state.x - wp_trans.location.x, state.ego_state.y - wp_trans.location.y])
        wp_vec = np.array([wp_dir.x, wp_dir.y])

        state.cross_track_error_m = float(wp_vec[0] * disp[1] - wp_vec[1] * disp[0])
        wp_yaw_rad = math.radians(wp_trans.rotation.yaw)
        diff_yaw = math.atan2(math.sin(state.ego_state.yaw_rad - wp_yaw_rad), math.cos(state.ego_state.yaw_rad - wp_yaw_rad))
        state.heading_error_deg = math.degrees(diff_yaw)

        curr_wp = waypoint
        dist_acc = 0.0
        state.dist_nxt_junc = float('inf')
        for _ in range(20):
            if curr_wp.is_junction:
                state.dist_nxt_junc = dist_acc
                break
            next_wps = curr_wp.next(1.0)
            if not next_wps:
                break
            curr_wp = next_wps[0]
            dist_acc += 1.0

    def compute(self, dt: float, vehicle: carla.Vehicle, sensors_data: dict, step):
        imu_data = sensors_data["imu"]
        gnss_data = sensors_data["gnss"]
        lidar_data = sensors_data["lidar"]
        radar_datas = sensors_data["radars"]
        cameras = sensors_data["stacked_cameras"]
        frame = sensors_data["yolo_camera"]
        if (imu_data == None):
            raise ValueError("Sensor data IMU must be provided for computation.")
        state = CalculatedState()
        self.speed_limit = vehicle.get_speed_limit()
        state.ego_state = self.ekf.compute(dt, vehicle, imu_data, gnss_data, lidar_data, step)
        lidar_np = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        self.tracker.push_frame(state, lidar_np, radar_datas, cameras, step)
        result = self.tracker.get_latest_result()
        if result is not None:
            state.detected_objects, state.ttcs, state.min_ttc, pts = result
            self._last_result = result
        elif self._last_result is not None:
            state.detected_objects, state.ttcs, state.min_ttc, pts = self._last_result
        else:
            state.detected_objects, state.ttcs, state.min_ttc, pts = [], np.array([]), float('inf'), None
        if frame is not None: self.tlw.push_frame(frame)
        self.state.light = self.tlw.light_state()
        self.get_rest_info(state)
        self.state = state
        type(self)._last_state = state
        self.ego_evaluator.add_frame(frame_id=str(step),ego_state=self.state.ego_state)
        self.ego_evaluator.print_report()
        self.bbs_evaluator.add_frame(frame_id=str(step),lidar_points_ego=pts, track_bbs=state.detected_objects,
                ego_transform=carla.Transform(carla.Location(state.ego_state.x,state.ego_state.y,state.ego_state.z),
                                    carla.Rotation(state.ego_state.pitch,state.ego_state.yaw,state.ego_state.roll)))
        self.evaluator.print_report()

    @staticmethod
    def get_calculated_state() -> CalculatedState:
        return CarlaStateEstimatorEngine._last_state
