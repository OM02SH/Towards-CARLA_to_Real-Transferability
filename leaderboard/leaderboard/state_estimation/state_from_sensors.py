import math
import carla
import numpy as np
import time
#from leaderboard_codes.metrics_calc_with_sensors.ego_UKF import EgoUnscentedKalmanFilterStateEstimator
#from leaderboard_codes.metrics_calc_with_sensors.ego_EKF import EgoExtendedKalmanFilterStateEstimator
from leaderboard.leaderboard.state_estimation.ego_EKF_3D import  EgoErrorStateEKF3D
from leaderboard.leaderboard.state_estimation.sensor_dataclasses import CalculatedState
from leaderboard.leaderboard.state_estimation.bbs_KF import BBsWorker
from leaderboard.leaderboard.state_estimation.TwoDimTTC import TTC
from leaderboard.leaderboard.state_estimation.traffic_light_worker import TrafficLightWorker
Gravity = 9.81
min_valid_range_m = 0.5

class CarlaStateEstimatorEngine:
    """
        Computes derived vehicle state from raw CARLA sensor data provided on each call.
    """
    def __init__(
            self,
            vehicle: carla.Vehicle, 
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
        self.tlw = TrafficLightWorker()
        self.ekf = EgoErrorStateEKF3D(vehicle, world_map)
        self.tracker = BBsWorker(lidar_transform, radar_transforms, self.half_length, self.half_width,)
        self.radar_transforms : dict[str, np.array] = {}
        if radar_transforms is None:
            raise ValueError("RADAR transforms must be provided for each RADAR sensor.")
        for name, transform in radar_transforms.items():
            self.radar_transforms[name] = np.array(transform.get_matrix())

        self.lidar_transform = np.array(carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + ext.z + 0.2),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)).get_matrix())

    def get_rest_info(self, state : CalculatedState):
        cr, sr = math.cos(state.ego_state.roll_rad), math.sin(state.ego_state.roll_rad)
        cp, sp = math.cos(state.ego_state.pitch_rad), math.sin(state.ego_state.pitch_rad)
        cy, sy = math.cos(state.ego_state.yaw_rad), math.sin(state.ego_state.yaw_rad)
        state.forward_vector_est = np.array([cy * cp, sy * cp, -sp])
        state.right_vector = np.array([
            cy * sp * sr + sy * cr,
            sy * sp * sr - cy * cr,
            cp * sr])
        state.up_vector = np.array([
            cy * sp * cr - sy * sr,
            sy * sp * cr + cy * sr,
            cp * cr])
        #state.forward_vector_est = np.array([cy, sy, 0])
        #state.right_vector = np.array([sy,-cy ,0])
        #state.up_vector = np.array([0,0,0])
        
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

    @staticmethod
    def transformation_matrix(px, py, pz, roll, pitch, yaw):
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ])
        Ry = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ])
        Rz = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [px, py, pz]
        return T
 
    def compute(self, dt: float, vehicle: carla.Vehicle, sensors_data: dict, step):
        imu_data = sensors_data["imu"]
        gnss_data = sensors_data["gnss"]
        lidar_data = sensors_data["lidar"]
        radar_datas = sensors_data["radars"]
        cameras = sensors_data["stacked_cameras"]
        if (imu_data == None or gnss_data == None or lidar_data == None or radar_datas == None):
            raise ValueError("All sensor data (IMU, GNSS, LiDAR, RADAR) must be provided for computation.")
        state = CalculatedState()
        self.speed_limit = vehicle.get_speed_limit()
        state.ego_state = self.ekf.compute(dt, vehicle, imu_data, gnss_data, lidar_data, step)
        
        self.get_rest_info(state)
        self.state = state

    @staticmethod
    def get_calculated_state() -> CalculatedState:
        return CarlaStateEstimatorEngine.state

    