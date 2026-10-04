import math
import carla
import numpy as np
from leaderboard.state_estimation.ego_EKF_3D import  EgoErrorStateEKF3D
from leaderboard.state_estimation.sensor_dataclasses import CalculatedState, Ego3DState
from leaderboard.state_estimation.bbs_KF import BBsWorker
from leaderboard.state_estimation.real_test_BBs import OnlineEvaluator
from leaderboard.state_estimation.traffic_light_worker import TrafficLightWorker
from leaderboard.state_estimation.real_test_ego import EgoStateOnlineEvaluator
import leaderboard.state_estimation.hyperparams as hp
from leaderboard.state_estimation.ekf_tuning import SensorRecorder

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
        #self.rec = SensorRecorder(self.map.transform_to_geolocation(carla.Location(0,0,0)))
        self.geo_ref = self.map.transform_to_geolocation(carla.Location(0,0,0))

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
        lidar_np = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        state = CalculatedState()
        perp_state = CalculatedState()
        self.speed_limit = vehicle.get_speed_limit()
        rec = SensorRecorder(self.geo_ref)
        wheel = EgoErrorStateEKF3D.get_simulated_speedometer(vehicle)   
        state.ego_state = self.ekf.compute(dt, vehicle, imu_data, gnss_data, lidar_data, step, wheel_speed= wheel)
        rec.add(step, dt, imu_data, gnss_data, lidar_data is not None, wheel, vehicle)
        rec.save(f"route_{step}.npz", self.ekf) 
        self.ego_evaluator.add_frame(frame_id=str(step),ego_state=self.state.ego_state)
        if step % 100 == 0: self.ego_evaluator.print_report()

        def _vec(v) -> np.ndarray:
            return np.array([v.x, v.y, v.z], dtype=float)

        def ego_state_from_vehicle() -> Ego3DState:
            tf = vehicle.get_transform()
            R_wb = np.asarray(tf.get_inverse_matrix(), dtype=float)[:3, :3]   # world -> body rotation

            v_w = _vec(vehicle.get_velocity())
            a_w = _vec(vehicle.get_acceleration())
            w_b = R_wb @ np.radians(_vec(vehicle.get_angular_velocity()))      # CARLA gives deg/s
            v_b = R_wb @ v_w
            a_b = R_wb @ a_w

            roll, pitch, yaw = (math.radians(a) for a in (tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw))
            c, s = math.cos(yaw), math.sin(yaw)
            nan = float("nan")

            return Ego3DState(
                x=tf.location.x, y=tf.location.y, z=tf.location.z,
                roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw,
                roll=tf.rotation.roll, pitch=tf.rotation.pitch, yaw=tf.rotation.yaw,
                roll_rate=w_b[0], pitch_rate=w_b[1], yaw_rate=w_b[2],
                vx_body=v_b[0], vy_body=v_b[1], vz_body=v_b[2],
                ax_body=a_b[0], ay_body=a_b[1], az_body=a_b[2],
                speed=float(np.linalg.norm(v_w)),
                vx_world=v_w[0], vy_world=v_w[1], vz_world=v_w[2],
                long_accel=c * a_w[0] + s * a_w[1],
                lat_accel=-s * a_w[0] + c * a_w[1],
                vert_accel=a_w[2],
                bg_yaw=nan, bg_pitch=nan, bg_roll=nan)

        gt_ego_state = ego_state_from_vehicle()
        perp_state.ego_state = gt_ego_state
        self.tracker.push_frame(perp_state, lidar_np, radar_datas, cameras, step)
        result = self.tracker.get_latest_result()

        if result is not None:
            state.detected_objects, state.ttcs, state.min_ttc, pts = result
            self._last_result = result
        elif self._last_result is not None:
            state.detected_objects, state.ttcs, state.min_ttc, pts = self._last_result
        else:
            state.detected_objects, state.ttcs, state.min_ttc, pts = [], np.array([]), float('inf'), None

        if frame is not None: self.tlw.push_frame(frame)
        state.light = self.tlw.light_state()  
        self.get_rest_info(state)
        self.state = state
        type(self)._last_state = state    
        
        pts_sensor = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
        ranges = np.linalg.norm(pts_sensor[:, :3], axis=1)
        pts_sensor = pts_sensor[ranges>hp.MIN_VALID_RANGE_M]
        self.bbs_evaluator.add_frame(frame_id=str(step), pts_sensor=pts_sensor,
            track_bbs=[b for b in state.detected_objects if b.moveable], ego_state=gt_ego_state)
        if step % 100 == 0: self.bbs_evaluator.print_report()

    @staticmethod
    def get_calculated_state() -> CalculatedState:
        return CarlaStateEstimatorEngine._last_state
