import math
import numpy as np
from scipy.stats import chi2
from matplotlib.path import Path  # add to imports at top of file

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.optimize import linear_sum_assignment
from leaderboard_codes.metrics_calc_with_sensors.sensor_dataclasses import UKFBoundingBox2D, trackedbb, CalculatedState
from sklearn.cluster import DBSCAN
import cv2
from scipy.linalg import block_diag


GROUND_CLEARANCE = 0.5
MAX_LOCKBACK = 20.0
MAX_LOOKFORWARD = 50.0
MAX_LOOKSIDEWAYS = 10.0
MAX_DISTANCE_BETWEEN_LIDAR_POINTS = 0.5
MIN_NUMBER_LIDAR_HITS = 10

def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def robust_sqrt(matrix: np.ndarray) -> np.ndarray:
    sym_matrix = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(sym_matrix)
    except np.linalg.LinAlgError:
        print("EEEEEEEEEEERRRRROOOOOORRRRRR")
        U, s, Vh = np.linalg.svd(sym_matrix)
        s = np.maximum(s, 1e-8)
        return U @ np.diag(np.sqrt(s))

class ObjectUKF:
    """
        x = [px, py, v, a, beta, yaw, yaw_rate, l, h]
    """
    def __init__(
        self,
        x0:                     float,
        y0:                     float,
        yaw0:                   float,
        dt:                     float,
        l0:                     float,
        w0:                     float,
        position_std:           float = 0.5,
        yaw_std:                float = math.radians(10.0),
        beta_std:               float = math.radians(8.0),
        speed_std:              float = 4.0,
        accel_std:              float = 4.0,
        yaw_rate_std:           float = math.radians(20.0),
        l_std:                  float = 0.125,
        w_std:                  float = 0.125,
        process_jerk_std:       float = 5,
        process_beta_rate_std:  float = math.radians(10.0),
        process_yaw_accel_std:  float = math.radians(20.0),
        process_l_std:          float = 0.05,
        process_w_std:          float = 0.05,
        lidar_yaw_std:          float = math.radians(5.0),
        lidar_l_std:            float = 0.25,
        lidar_w_std:            float = 0.25,
        radar_velocity_std:     float = 0.25,
        min_speed_for_slip:     float = 0.5
    ):
        self.process_yaw_accel_std = process_yaw_accel_std
        self.process_jerk_std = process_jerk_std
        self.process_beta_rate_std = process_beta_rate_std
        self.process_l_std = process_l_std
        self.process_w_std = process_w_std
        self.min_speed_for_slip = min_speed_for_slip
        self.dt = dt

        sigma_points = MerweScaledSigmaPoints(n=9, alpha=0.1, beta=2.0, kappa=0.0,sqrt_method=robust_sqrt)

        self.ukf = UnscentedKalmanFilter(dim_x=9, dim_z=5, dt=dt, fx=self.fx, hx=self.hx_lidar, 
                    points=sigma_points,x_mean_fn=self.state_mean,residual_x=self.state_residual)

        self.ukf.x = np.array([x0, y0, 0.0, 0.0, 0.0, yaw0, 0.0, l0, w0],dtype=float)
        self.ukf.P = np.diag([position_std**2,position_std**2,speed_std**2,accel_std**2,beta_std**2,yaw_std**2,yaw_rate_std**2,l_std**2,w_std**2])
        self.ukf.Q = self.make_process_noise(dt)
        self.lidarR = np.diag([position_std**2,position_std**2,lidar_yaw_std**2, lidar_l_std**2, lidar_w_std**2])
        self.radarR = np.array([[radar_velocity_std**2]])
        self.ukf.R = self.lidarR

    def stabilize_covariance(self):
        P = self.ukf.P
        P[:] = 0.5 * (P + P.T)
        if np.any(np.diag(P) <= 0):
            P[:] += np.eye(9) * 1e-8

    @staticmethod
    def state_mean(sigmas: np.ndarray,weights: np.ndarray) -> np.ndarray:
        x = np.zeros(9)
        for i in (0, 1, 2, 3, 6, 7, 8):
            x[i] = np.sum(sigmas[:, i] * weights)
        x[4] = math.atan2(np.sum(np.sin(sigmas[:, 4]) * weights),np.sum(np.cos(sigmas[:, 4]) * weights),)
        x[5] = math.atan2(np.sum(np.sin(sigmas[:, 5]) * weights),np.sum(np.cos(sigmas[:, 5]) * weights),)
        return x

    @staticmethod
    def z_mean(sigmas, weights):
        m = np.sum(sigmas * weights[:, None], axis=0)
        m[2] = math.atan2(np.sum(np.sin(sigmas[:, 2]) * weights),np.sum(np.cos(sigmas[:, 2]) * weights),)
        return m

    @staticmethod
    def state_residual(a: np.ndarray,b: np.ndarray) -> np.ndarray:
        y = a - b
        y[4] = normalize_angle(y[4])
        y[5] = normalize_angle(y[5])
        return y

    @staticmethod
    def lidar_residual(a: np.ndarray,b: np.ndarray) -> np.ndarray:
        y = a - b
        y[2] = normalize_angle(y[2])
        return y
    
    @staticmethod
    def radar_residual(a: np.ndarray,b: np.ndarray) -> np.ndarray:
        return a - b
    # Constant Turn Rate and Acceleration
    @staticmethod
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        px, py, v, a, beta, yaw, yaw_rate, l, w = x
        course = normalize_angle(yaw + beta)
        course_new = normalize_angle(course + yaw_rate * dt)
        yaw_new = normalize_angle(yaw + yaw_rate * dt)
        beta_new = normalize_angle(course_new - yaw_new)
        v_new = max(0.0, v + a * dt)

        if abs(yaw_rate) < 1e-5:
            dist = v * dt + 0.5 * a * dt**2
            px_new = px + dist * math.cos(course)
            py_new = py + dist * math.sin(course)
        else:
            sin_c, cos_c = math.sin(course), math.cos(course)
            sin_cn, cos_cn = math.sin(course_new), math.cos(course_new)
            px_new = px + ((v * yaw_rate + a * yaw_rate * dt) * sin_cn
                           + a * cos_cn - v * yaw_rate * sin_c
                           - a * cos_c) / yaw_rate**2
            py_new = py + ((-v * yaw_rate - a * yaw_rate * dt) * cos_cn
                           + a * sin_cn + v * yaw_rate * cos_c
                           - a * sin_c) / yaw_rate**2
        return np.array([px_new, py_new, v_new, a, beta_new, yaw_new, yaw_rate, l, w])
    
    """def make_process_noise(self, dt: float) -> np.ndarray:
        q_j = self.process_jerk_std**2
        q_ya = self.process_yaw_accel_std**2
# https://arxiv.org/pdf/1809.06518 A White-Noise-On-Jerk Motion Prior for Continuous-Time Trajectory Estimation on SE(3)
        Q_va = q_j * np.array([[dt**3 / 3.0, dt**2 / 2.0],[dt**2 / 2.0, dt]])
        q_p = (1.0 / 20.0) * q_j * dt**5
        Q_yaw = q_ya * np.array([[dt**3 / 3.0, dt**2 / 2.0],[dt**2 / 2.0, dt]])
        q_l = (self.process_l_std ** 2) * dt
        q_w = (self.process_w_std ** 2) * dt
        Q = np.zeros((8, 8))
        Q[0, 0] = q_p
        Q[1, 1] = q_p
        Q[2:4, 2:4] = Q_va        # v, a
        Q[4:6, 4:6] = Q_yaw       # yaw, yaw_rate
        Q[6, 6] = q_l
        Q[7, 7] = q_w
        return Q"""

    def make_process_noise(self, dt: float) -> np.ndarray:
        x = self.ukf.x
        v = max(0.0, float(x[2]))
        course = normalize_angle(float(x[4] + x[5]))
        c, s = math.cos(course), math.sin(course)
        q = np.zeros((9, 9))
        #Each g vector is the sensitivity of every state to that noise source
        g_j = np.array([c * dt**3 / 6.0, s * dt**3 / 6.0, dt**2 / 2.0,
            dt, 0.0, 0.0, 0.0, 0.0, 0.0])
        g_ya = np.array([-v * s * dt**3 / 6.0, v * c * dt**3 / 6.0,
             0.0, 0.0, 0.0, dt**2 / 2.0, dt, 0.0, 0.0])
        g_beta = np.array([-v * s * dt**2 / 2.0,v * c * dt**2 / 2.0,
                           0.0,0.0,dt,0.0,0.0,0.0,0.0])
        q += (self.process_jerk_std**2) * np.outer(g_j, g_j)
        q += (self.process_yaw_accel_std**2) * np.outer(g_ya, g_ya)
        q += (self.process_beta_rate_std**2) * np.outer(g_beta, g_beta)
        q[7, 7] = (self.process_l_std**2) * dt
        q[8, 8] = (self.process_w_std**2) * dt
        return 0.5 * (q + q.T)

    @staticmethod
    def hx_lidar(x: np.ndarray) -> np.ndarray:
        return np.array([x[0],x[1],x[5],x[7],x[8]])
    
    @staticmethod
    def hx_radar(x: np.ndarray,u_world: np.ndarray) -> np.ndarray:
        _, _, v, _, beta, yaw, _, _, _ = x
        course = normalize_angle(yaw + beta)
        vx = v * math.cos(course)
        vy = v * math.sin(course)
        u = np.asarray(u_world, dtype=float)
        norm = np.linalg.norm(u[:2])
        if norm < 1e-9:
            return np.array([0.0])
        ux, uy = u[0] / norm, u[1] / norm
        return np.array([vx * ux + vy * uy])
    
    def predict(self):
        self.ukf.Q = self.make_process_noise(self.dt)
        self.ukf.predict(dt = self.dt)
        self.ukf.x[4] = normalize_angle(self.ukf.x[4])
        self.ukf.x[5] = normalize_angle(self.ukf.x[5])
        self.stabilize_covariance()

    @staticmethod
    def lidar_z_mean(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        z = np.zeros(5)
        z[0] = np.sum(sigmas[:, 0] * weights)
        z[1] = np.sum(sigmas[:, 1] * weights)
        z[2] = math.atan2(np.sum(np.sin(sigmas[:, 2]) * weights), np.sum(np.cos(sigmas[:, 2]) * weights))
        z[3] = np.sum(sigmas[:, 3] * weights)
        z[4] = np.sum(sigmas[:, 4] * weights)
        return z

    """def update_lidar(self, center_x: float, center_y: float, yaw: float):
        self.ukf.hx = self.hx_lidar
        self.ukf.R = self.lidarR
        self.ukf.dim_z = 3
        z = np.array([center_x, center_y, normalize_angle(yaw)])
        self.ukf.residual_z = self.lidar_residual
        self.ukf.z_mean_fn = self.lidar_z_mean
        self.ukf.update(z,hx=self.hx_lidar,R=self.ukf.R)
        self.ukf.x[4] = normalize_angle(self.ukf.x[4])
        self.stabilize_covariance()

    def update_radar(self, v_r_abs: float, u_world: np.ndarray):
        self.ukf.sigmas_f = self.ukf.points_fn.sigma_points(self.ukf.x, self.ukf.P)
        def hx(x, *args, **kwargs):
            return self.hx_radar(x, u_world)
        z = np.array([v_r_abs])
        self.ukf.dim_z = 1
        self.ukf.residual_z = self.radar_residual
        self.ukf.z_mean_fn = None
        self.ukf.update(z, hx=hx, R=self.radarR)
        self.ukf.x[4] = normalize_angle(self.ukf.x[4])
        self.stabilize_covariance()"""

    def update_combined(self, lidar_meas: tuple, radar_meas_list: list[tuple]):
        z_parts, R_blocks = [], []
        cx, cy, yaw, l, w = lidar_meas
        z_parts.append(np.array([cx, cy, normalize_angle(yaw), l, w]))
        R_blocks.append(self.lidarR)
        for v_r_abs, u_world in radar_meas_list:
            z_parts.append(np.array([v_r_abs]))
            R_blocks.append(self.radarR)

        z = np.concatenate(z_parts)
        R = block_diag(*R_blocks)
        def hx(x):
            out = []
            out.append(np.array([x[0],x[1],x[5],x[7],x[8]]))
            for _, u_world in radar_meas_list:
                out.append(self.hx_radar(x, u_world))
            return np.concatenate(out)

        dim_z = len(z)
        n_sig = self.ukf.sigmas_f.shape[0]
        self.ukf._dim_z = dim_z
        self.ukf.sigmas_h = np.zeros((n_sig, dim_z))
        self.ukf.K = np.zeros((9, dim_z))
        self.ukf.y = np.zeros(dim_z)
        self.ukf.z = np.zeros((dim_z, 1))
        self.ukf.S = np.zeros((dim_z, dim_z))
        self.ukf.SI = np.zeros((dim_z, dim_z))

        self.ukf.residual_z = self.lidar_residual
        self.ukf.z_mean_fn = self.z_mean
        self.ukf.update(z, hx=hx, R=R)
        self.ukf.x[4] = normalize_angle(self.ukf.x[4])
        self.ukf.x[5] = normalize_angle(self.ukf.x[5])
        self.stabilize_covariance()

    @property
    def position(self):
        return self.ukf.x[0], self.ukf.x[1]

    @property
    def speed(self):
        return float(self.ukf.x[2])

    @property
    def acceleration(self):
        return float(self.ukf.x[3])

    @property
    def sideslip(self):
        return float(normalize_angle(self.ukf.x[4]))

    @property
    def yaw(self):
        return float(normalize_angle(self.ukf.x[5]))

    @property
    def yaw_rate(self):
        return float(self.ukf.x[6])

    @property
    def dims(self):
        return self.ukf.x[7], self.ukf.x[8]

    @property
    def vx(self):
        return (self.speed * math.cos(self.yaw))

    @property
    def vy(self):
        return (self.speed * math.sin(self.yaw))

    @property
    def ax(self):
        return self.acceleration * math.cos(self.yaw) - self.speed * self.yaw_rate * math.sin(self.yaw)

    @property
    def ay(self):
        return self.acceleration * math.sin(self.yaw) + self.speed * self.yaw_rate * math.cos(self.yaw)

    @property
    def covariance(self):
        return self.ukf.P

class UKFTrackBB2D:
    def __init__(self, track_id: int, detection : UKFBoundingBox2D,dt: float, max_history: int = 20,
                window_size: int = 10,min_motion_speed: float = 0.5, flip_tolerance_deg: float = 25.0):
        self.id = track_id
        self.dt = dt
        self.ukf = ObjectUKF(detection.center_x, detection.center_y, math.radians(detection.yaw), dt, detection.length, detection.width)
        self.yaw = self.ukf.yaw
        self.bbs = [detection]
        self.len = detection.length
        self.wid = detection.width
        self.px = 0.0
        self.py = 0.0
        self.missed_ticks = 0
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.yaw_rate = 0.0
        self.pos_history: list[tuple[float,float,float]] = []
        self.slip_angle_deg = 0.0
        self.flip_tolerance_deg = flip_tolerance_deg
        self.min_motion_speed = min_motion_speed
        self.window_size = window_size
        self.max_history = max_history
        self.slip_angle_deg = 0.0

    def update_metadata(self,step):
        self.px, self.py = self.ukf.position
        self.len, self.wid = self.ukf.dims
        self.vx = self.ukf.vx
        self.vy = self.ukf.vy
        self.speed = self.ukf.speed
        self.acceleration = self.ukf.acceleration
        self.ax = self.ukf.ax
        self.ay = self.ukf.ay
        self.yaw = self.ukf.yaw
        self.yaw_rate = self.ukf.yaw_rate
        if len(self.pos_history) > self.max_history:
            self.pos_history.pop(0)
        self.pos_history.append((self.px,self.py,step*self.dt))
        self.slip_angle_deg = math.degrees(self.ukf.sideslip)

    def get_all_data(self):
        return trackedbb(px=self.px, py=self.py, moveable = True,
                vx=self.vx, vy=self.vy, speed=self.speed, 
                acceleration=self.acceleration, ax=self.ax, ay=self.ay,
                yaw=self.yaw, yaw_rate=self.yaw_rate, len=self.len, wid=self.wid,
                pos_history=self.pos_history,slip_angle_deg=self.slip_angle_deg)

    """def estimate_slip_angle(self):
        self.slip_angle_deg = 0.0
        if len(self.pos_history) < self.window_size:
            return

        window = self.pos_history[-self.window_size:]
        xs = np.array([p[0] for p in window])
        ys = np.array([p[1] for p in window])
        ts = np.array([p[2] for p in window])

        if ts[-1] - ts[0] < 1e-6:
            return
        vx_fit = np.polyfit(ts, xs, deg=1)[0]
        vy_fit = np.polyfit(ts, ys, deg=1)[0]
        motion_speed = math.hypot(vx_fit, vy_fit)

        if motion_speed < self.min_motion_speed:
            return

        heading_from_motion = math.atan2(vy_fit, vx_fit)
        slip = normalize_angle(heading_from_motion - self.yaw)
        self.ukf.ukf.x[4] = math.degrees(slip)

        if abs(abs(self.slip_angle_deg) - 180.0) < self.flip_tolerance_deg:
            self.ukf.ukf.x[5] = normalize_angle(self.ukf.yaw + math.pi)
            self.ukf.ukf.x[6] = -self.ukf.ukf.x[5]
            self.ukf.ukf.x[4] = math.degrees(normalize_angle(heading_from_motion - self.ukf.yaw))"""

class UKFBoundingBoxTracker:
    def __init__(self, max_match_dist_m: float = 3.0, max_missed_ticks: int = 10, dt: float = 0.05
                 , gate_prob: float = 0.95, static_position_std: float = 0.4):
        self.tracks : list[UKFTrackBB2D] = []
        self._next_id = 0
        self.max_match_dist_m = max_match_dist_m
        self.max_missed_ticks = max_missed_ticks
        self.gate_threshold = chi2.ppf(gate_prob, df=2)
        self.static_S = np.eye(2) * (static_position_std ** 2)
        self.dt = dt

    def _spawn_track(self,detection,step):
        track = UKFTrackBB2D(self._next_id, detection, self.dt)
        self._next_id += 1
        """if np.any(detection.radar_points):
            for radar in detection.radar_points:
                track.ukf.update_radar(v_r_abs=float(radar[3]), u_world=np.asarray(radar[4:7]))"""
        radar_meas_list = [(float(r[3]), np.asarray(r[4:7])) for r in detection.radar_points]
        track.ukf.update_combined((detection.center_x, detection.center_y, math.radians(detection.yaw), detection.length, detection.width),radar_meas_list)
        track.update_metadata(step)
        self.tracks.append(track)

    def associate(self, detections):
        if not self.tracks or not detections:
            return [], set(), set()
        cost = np.full((len(self.tracks),len(detections)), 1e6, dtype=float)
        for i, track in enumerate(self.tracks):
            predicted = track.ukf.ukf.x[0:2]
            P_pos = track.ukf.ukf.P[0:2, 0:2]
            R_pos = track.ukf.lidarR[0:2, 0:2]
            S = P_pos + R_pos

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            for j, det in enumerate(detections):
                measured = np.array([det.center_x, det.center_y])
                d = measured - predicted
                m_dist_sq = float(d @ S_inv @ d)
                if m_dist_sq <= self.gate_threshold:
                    cost[i, j] = m_dist_sq
        rows, cols = linear_sum_assignment(cost)
        matches = []
        matched_tracks = set()
        matched_detections = set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            matches.append((i, j))
            matched_tracks.add(i)
            matched_detections.add(j)
        return matches, matched_tracks, matched_detections

    def update(self, detections, step: float):
        for track in self.tracks:
            track.ukf.predict()
        if not detections:
            self.prune(None)
            return self.tracks
        if not self.tracks:
            for detection in detections:
                self._spawn_track(detection,step)
            return self.tracks
        matches, matched_tracks, matched_detections = self.associate(detections)
        for (track_idx, det_idx) in matches:
            track : UKFTrackBB2D = self.tracks[track_idx]
            detection = detections[det_idx]
            radar_meas_list = [(float(r[3]), np.asarray(r[4:7])) for r in detection.radar_points]
            track.ukf.update_combined((detection.center_x, detection.center_y, math.radians(detection.yaw), detection.length, detection.width),radar_meas_list)
            track.bbs.append(detection)
            if len(track.bbs) > 10:
                track.bbs.pop(0)
            track.missed_ticks = 0
            track.update_metadata(step)
        self.prune(matched_tracks)
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                self._spawn_track(detection,step)
        return self.tracks

    def prune(self, matched_tracks):
        if matched_tracks:
            for i, track in enumerate(self.tracks):
                if i not in matched_tracks:
                    track.missed_ticks += 1
        else:
            for track in self.tracks:
                track.missed_ticks += 1
        self.tracks = [track for track in self.tracks if ( track.missed_ticks <= self.max_missed_ticks)]


def detect_cars_from_lidar(half_length, half_width, state: CalculatedState, master_cloud: np.ndarray, radar_points, step, tracker: UKFBoundingBoxTracker) -> list[trackedbb]:
    obstacle_mask = (master_cloud[:, 0] > -MAX_LOCKBACK) & (master_cloud[:, 0] < MAX_LOOKFORWARD) &\
          (np.abs(master_cloud[:, 1]) < MAX_LOOKSIDEWAYS) & (master_cloud[:, 2] > -GROUND_CLEARANCE) &\
          ((np.abs(master_cloud[:, 0]) > half_length) | (np.abs(master_cloud[:, 1]) > half_width))
    obstacle_pts = master_cloud[obstacle_mask, :3]
    voxel_size = 0.4
    quantized = np.floor(obstacle_pts / voxel_size).astype(np.int64)
    quantized -= quantized.min(axis=0)
    dims = quantized.max(axis=0) + 1
    voxel_keys = (quantized[:, 0] * dims[1] * dims[2]
                + quantized[:, 1] * dims[2]
                + quantized[:, 2])
    _, unique_indices = np.unique(voxel_keys, return_index=True)
    obstacle_pts = obstacle_pts[unique_indices]
    if len(obstacle_pts) < MIN_NUMBER_LIDAR_HITS:
        return tracker.update([], step)
    yaw_rad = state.ego_state.yaw_rad  
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    rot2D = np.array([[cos_yaw, -sin_yaw],[sin_yaw,  cos_yaw]])
    clustering = DBSCAN(eps=MAX_DISTANCE_BETWEEN_LIDAR_POINTS, min_samples=MIN_NUMBER_LIDAR_HITS,algorithm='kd_tree',n_jobs=-1).fit(obstacle_pts)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}
    detections = []
    tracked : list[trackedbb] = []
    for cluster_id in unique_labels:
        cluster = obstacle_pts[labels == cluster_id]
        if cluster.shape[0] < 2:    continue
        min_x, max_x = np.min(cluster[:, 0]), np.max(cluster[:, 0]) 
        min_y, max_y = np.min(cluster[:, 1]), np.max(cluster[:, 1]) 
        min_z, max_z = np.min(cluster[:, 2]), np.max(cluster[:, 2])
        span_z = max_z - min_z
        print("cluster in original coords : min x ", min_x," max x", max_x, "min y", min_y," max y", max_y,"min z", min_z," max z", max_z)
        cov_matrix = np.cov(cluster, rowvar=False)
        if cov_matrix.ndim < 2: continue
        cluster_global_2d = (cluster[:, :2] @ rot2D.T) + np.array([state.ego_state.x, state.ego_state.y])
        evals, evecs = np.linalg.eigh(cov_matrix)
        is_wall = ((evals[1] - evals[0]) / (evals[2] + 1e-6) > 0.85) and (abs(evecs[2, 0]) < 0.3)
        min_x, max_x = np.min(cluster_global_2d[:, 0]), np.max(cluster_global_2d[:, 0]) 
        min_y, max_y = np.min(cluster_global_2d[:, 1]), np.max(cluster_global_2d[:, 1]) 
        min_z, max_z = np.min(cluster[:, 2]), np.max(cluster[:, 2])
        span_z = max_z - min_z
        print("cluster after 2D rotation : min x ", min_x," max x", max_x, "min y", min_y," max y", max_y,"min z", min_z," max z", max_z)
        if is_wall:
            center_x, center_y = np.mean(cluster_global_2d, axis=0)
            centered_2d = cluster_global_2d - [center_x, center_y]
            _, evecs_2d = np.linalg.eigh(np.cov(centered_2d, rowvar=False))
            yaw_deg = np.degrees(np.arctan2(evecs_2d[:, 1][1], evecs_2d[:, 1][0]))            
            projections = centered_2d @ evecs_2d[:, 1]
            length = float(np.max(projections) - np.min(projections))
            width = 0.001
            moveable = False
        else:
            (center_x, center_y), (dim_a, dim_b), raw_angle = cv2.minAreaRect(cluster_global_2d[:, :2].astype(np.float32))
            if dim_a >= dim_b:
                length = dim_a
                width = dim_b
                yaw_deg = raw_angle
            else:
                length = dim_b
                width = dim_a
                yaw_deg = raw_angle + 90.0
            if span_z > 2.5 or span_z < 0.5 or width > 5 or width < 0.5 or length > 7 or length < 0.5:
                moveable = False
            else:
                moveable = True
        margin = 0.2
        radar_mask = (radar_points[:, 0] >= min_x - margin) & (radar_points[:, 0] <= max_x + margin) & \
                (radar_points[:, 1] >= min_y - margin) & (radar_points[:, 1] <= max_y + margin)
        candidate_points = (radar_points[radar_mask] if len(radar_points) > 0 else np.empty((0, 7),dtype=np.float32))
        if len(candidate_points) > 0:
            box = cv2.boxPoints(((center_x, center_y), (width + margin*2, length + margin*2), yaw_deg))
            poly = Path(box)
            inside = poly.contains_points(candidate_points[:, :2])
            matched_radars = candidate_points[inside]
        else:
            matched_radars = np.array([])
        if moveable:
            detections.append(UKFBoundingBox2D(
            center_x=center_x,
            center_y=center_y,
            length=length,
            width=width,
            yaw = yaw_deg,
            step=step,
            radar_points=matched_radars
            ))
        else:
            tracked.append(trackedbb(px=center_x, py=center_y, moveable=False,
                vx=0, vy=0, speed=0, acceleration=0, ax=0, ay=0,
                yaw=yaw_deg * (math.pi/180), yaw_rate=0,
                len=length, wid=width, pos_history=[], slip_angle_deg=0))
    trackes = tracker.update(detections,step)
    for track in trackes:
        data = track.get_all_data()
        #print(data)
        tracked.append(data)
    return tracked

