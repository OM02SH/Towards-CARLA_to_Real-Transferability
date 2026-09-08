import math
import carla
import numpy as np
import time
#from leaderboard_codes.metrics_calc_with_sensors.ego_UKF import EgoUnscentedKalmanFilterStateEstimator
#from leaderboard_codes.metrics_calc_with_sensors.ego_EKF import EgoExtendedKalmanFilterStateEstimator
from leaderboard.leaderboard.state_estimation.ego_EKF_3D import  EgoErrorStateEKF3D
from leaderboard.leaderboard.state_estimation.sensor_dataclasses import CalculatedState
from leaderboard.leaderboard.state_estimation.bbs_KF import UKFBoundingBoxTracker, trackedbb , detect_cars_from_lidar
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
        self.tracker = UKFBoundingBoxTracker()
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
    
    def process_radar(self, state: CalculatedState, radar_data: carla.RadarMeasurement, sensor_transform: np.ndarray) -> np.ndarray:
        e = state.ego_state
        T_ego_to_world = self.transformation_matrix(e.x,e.y,self.z,0,0,e.yaw_rad)
        T_sensor_to_world = T_ego_to_world @ sensor_transform
        R_sensor_to_world = T_sensor_to_world[:3, :3]
        v_ego_world = np.array([e.vx_world, e.vy_world, 0], dtype=np.float32)

        parsed_points = []
        for det in radar_data:
            r, az, alt, v_r = det.depth, det.azimuth, det.altitude, det.velocity
            cos_alt = math.cos(alt)

            p_sensor = np.array([r * cos_alt * math.cos(az), r * cos_alt * math.sin(az), r * math.sin(alt), 1.0], dtype=np.float32)
            p_world = (T_sensor_to_world @ p_sensor)[:3]

            u_sensor = np.array([cos_alt * math.cos(az), cos_alt * math.sin(az), math.sin(alt)], dtype=np.float32)
            u_world = R_sensor_to_world @ u_sensor

            v_ego_radial = np.dot(v_ego_world, u_world)
            v_r_abs = v_r + v_ego_radial

            parsed_points.append([
                p_world[0], p_world[1], p_world[2], v_r_abs,
                u_world[0], u_world[1], u_world[2]])
        if not parsed_points:
            return np.empty((0, 7), dtype=np.float32)

        return np.array(parsed_points, dtype=np.float32)

    def get_radar_info(self, state : CalculatedState, radar_streams: dict[str, carla.RadarMeasurement]):
        self.radar_points = np.ndarray((0, 7))
        for sensor_name, radar_data in radar_streams.items():
            if radar_data is None:
                continue
            self.radar_points = np.vstack([self.radar_points, self.process_radar(state, radar_data, self.radar_transforms[sensor_name])])

    def get_lidar_info(self, state : CalculatedState, lidar_stream: carla.LidarMeasurement,step):
        pts = np.ndarray((0, 4))
        raw_pts = np.frombuffer(lidar_stream.raw_data, dtype=np.dtype('f4'))
        pts_sensor = np.reshape(raw_pts, (int(len(raw_pts) / 4), 4))
        ranges = np.linalg.norm(pts_sensor[:, :3], axis=1)
        pts_sensor = pts_sensor[ranges > min_valid_range_m]
        num_pts = len(pts_sensor)
        xyz_homo = np.hstack((pts_sensor[:, :3], np.ones((num_pts, 1))))
        xyz_ego = (self.lidar_transform @ xyz_homo.T).T[:, :3]

        pts = np.vstack((pts, np.column_stack((xyz_ego, pts_sensor[:, 3]))))
        time1 = time.perf_counter_ns()
        tracked : list[trackedbb] = detect_cars_from_lidar(self.half_length,self.half_width, state,pts,self.radar_points,step,self.tracker)
        time2 = time.perf_counter_ns()
        diff_ns = time2 - time1
        print(f"Time taken in tracking: {diff_ns:,} ns")

        state.detected_objects = tracked
        if state.ego_state.speed > 1e-3:
            hx = state.ego_state.vx_world / state.ego_state.speed
            hy = state.ego_state.vy_world / state.ego_state.speed
        else:
            hx = math.cos(state.ego_state.yaw)
            hy = math.sin(state.ego_state.yaw)
        ego_ttc = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.vx_world, state.ego_state.vy_world, hx, hy, self.half_length * 2, self.half_width * 2])
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
            state.ttcs = TTC(ego_ttc,obs_ttc)
            state.min_ttc = float(np.min(state.ttcs))

    def compute(
        self,
        dt: float,
        vehicle: carla.Vehicle,
        imu_data: carla.IMUMeasurement,
        gnss_data: carla.GnssMeasurement,
        lidar_data: carla.LidarMeasurement,
        radar_datas: dict[str, carla.RadarMeasurement],
        
        step
    ) -> None:
        if (imu_data == None or gnss_data == None or lidar_data == None or radar_datas == None):
            raise ValueError("All sensor data (IMU, GNSS, LiDAR, RADAR) must be provided for computation.")
        state = CalculatedState()
        self.speed_limit = vehicle.get_speed_limit()
        state.ego_state = self.ekf.compute(dt, vehicle, imu_data, gnss_data, lidar_data, step)
        self.get_rest_info(state)
        #self.get_radar_info(state, radar_datas)
        #self.get_lidar_info(state, lidar_data,step)
        #return state
        self.state = state

    @staticmethod
    def get_calculated_state() -> CalculatedState:
        return CarlaStateEstimatorEngine.state

    