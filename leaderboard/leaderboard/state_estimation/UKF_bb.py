import math
from typing import Optional
import numpy as np
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from leaderboard.state_estimation.sensor_dataclasses import bev_Det, trackedbb

def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def robust_sqrt(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    try:
        return np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(sym)
        vals = np.maximum(vals, 1e-10)
        return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

DIM_X = 12
YAW_IDX = 6
ACCEL_DECAY_TAU = 1.0
class ObjectUKF:
    """ 
        0     position x
        1     position y
        2     velocity x (world frame)
        3     velocity y (world frame)
        4     acceleration x (world frame)
        5     acceleration y (world frame)
        6     heading (yaw)
        7     heading rate
        8     length
        9     width
        10    center z (elevation of box center, world frame)
        11    height (vertical extent of box)
    """

    def __init__(
        self,
        dt: float,
        x0: float,
        y0: float,
        yaw0: float,
        vx0: float,
        vy0: float,
        l0: float,
        w0: float,
        z0: float,
        h0: float,
        position_std: float = 0.5,
        yaw_std: float = math.radians(10.0),
        accel_std: float = 4.0,
        yaw_rate_std: float = math.radians(20.0),
        l_std: float = 0.125,
        w_std: float = 0.125,
        z_std: float = 0.25,
        h_std: float = 0.25,
        process_jerk_std: float = 5,
        process_jerk_lat_scale: float = 1.0,
        process_yaw_accel_std: float = math.radians(20.0),
        process_l_std: float = 0.05,
        process_w_std: float = 0.05,
        process_z_std: float = 0.05,
        process_h_std: float = 0.05,
        lidar_yaw_std: float = math.radians(5.0),
        lidar_l_std: float = 0.25,
        lidar_w_std: float = 0.25,
        lidar_z_std: float = 0.15,
        lidar_h_std: float = 0.15,
        radar_velocity_std: float = 0.25,
        yaw_align_min_speed: float = 2.0,
        radar_gate_prob: float = 0.95,
        det_velocity_std: float = 1.0
    ):
        self.process_yaw_accel_std = process_yaw_accel_std
        self.process_jerk_std = process_jerk_std
        self.process_jerk_lat_scale = process_jerk_lat_scale
        self.process_l_std = process_l_std
        self.process_w_std = process_w_std
        self.process_z_std = process_z_std
        self.process_h_std = process_h_std
        self.yaw_align_min_speed = yaw_align_min_speed
        self.dt = dt
        self.radar_gate_threshold = chi2.ppf(radar_gate_prob, df=1)
        self._predicted_sigmas_valid = False

        sigma_points = MerweScaledSigmaPoints(n=DIM_X, alpha=0.1, beta=2.0, kappa=0.0, sqrt_method=robust_sqrt)
        self.ukf = UnscentedKalmanFilter(dim_x=DIM_X, dim_z=7, dt=dt, fx=self.fx, hx=self.hx_lidar,
                    points=sigma_points, x_mean_fn=self.state_mean, residual_x=self.state_residual)

        vx_init, vy_init = (vx0, vy0) 
        if math.hypot(vx_init, vy_init) >= max(yaw_align_min_speed, 3.0 * det_velocity_std):
            course0 = math.atan2(vy_init, vx_init)
            if abs(normalize_angle(yaw0 - course0)) > math.pi / 2:
                yaw0 = yaw0 + math.pi

        self.ukf.x = np.zeros(DIM_X)
        self.ukf.x[0] = x0
        self.ukf.x[1] = y0
        self.ukf.x[2] = vx_init
        self.ukf.x[3] = vy_init
        self.ukf.x[6] = normalize_angle(yaw0)
        self.ukf.x[8] = l0
        self.ukf.x[9] = w0
        self.ukf.x[10] = z0
        self.ukf.x[11] = h0

        self.ukf.P = np.diag([position_std**2, position_std**2, det_velocity_std**2, det_velocity_std**2,
            accel_std**2, accel_std**2, yaw_std**2, yaw_rate_std**2, l_std**2, w_std**2, z_std**2, h_std**2])
        self.ukf.Q = self.make_process_noise(dt)
        self.lidarR = np.diag([position_std**2, position_std**2, lidar_yaw_std**2, lidar_l_std**2, lidar_w_std**2, lidar_z_std**2, lidar_h_std**2])
        self.detVelR = np.diag([det_velocity_std**2, det_velocity_std**2])
        self.radarR = np.array([[radar_velocity_std**2]])
        self.ukf.R = self.lidarR

    def stabilize_covariance(self) -> None:
        P = 0.5 * (self.ukf.P + self.ukf.P.T)
        vals, vecs = np.linalg.eigh(P)
        vals = np.maximum(vals, 1e-9)
        self.ukf.P[:] = vecs @ np.diag(vals) @ vecs.T

    @staticmethod
    def state_mean(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        x = np.sum(sigmas * weights[:, None], axis=0)
        x[YAW_IDX] = math.atan2(np.sum(np.sin(sigmas[:, YAW_IDX]) * weights), np.sum(np.cos(sigmas[:, YAW_IDX]) * weights))
        return x

    @staticmethod
    def state_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        y = a - b
        y[YAW_IDX] = normalize_angle(y[YAW_IDX])
        return y

    @staticmethod
    def _z_mean(sigmas: np.ndarray, weights: np.ndarray, angle_idx: Optional[int] = None) -> np.ndarray:
        m = np.sum(sigmas * weights[:, None], axis=0)
        if angle_idx is not None:
            m[angle_idx] = math.atan2(np.sum(np.sin(sigmas[:, angle_idx]) * weights), np.sum(np.cos(sigmas[:, angle_idx]) * weights))
        return m

    @staticmethod
    def _z_residual(a: np.ndarray, b: np.ndarray, angle_idx: Optional[int] = None) -> np.ndarray:
        y = a - b
        if angle_idx is not None:
            y[angle_idx] = normalize_angle(y[angle_idx])
        return y

    @staticmethod
    def z_mean(sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return ObjectUKF._z_mean(sigmas, weights, angle_idx=2)

    @staticmethod
    def lidar_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return ObjectUKF._z_residual(a, b, angle_idx=2)

    @staticmethod
    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        px, py, vx, vy, ax, ay, yaw, yaw_rate, l, w, z, h = x
        psi = yaw_rate * dt

        dx = vx * dt + 0.5 * ax * dt * dt
        dy = vy * dt + 0.5 * ay * dt * dt
        ch, sh = math.cos(0.5 * psi), math.sin(0.5 * psi)
        px_new = px + ch * dx - sh * dy
        py_new = py + sh * dx + ch * dy

        vx1, vy1 = vx + ax * dt, vy + ay * dt
        c, s = math.cos(psi), math.sin(psi)
        decay = math.exp(-dt / ACCEL_DECAY_TAU)
        vx_new = c * vx1 - s * vy1
        vy_new = s * vx1 + c * vy1
        ax_new = decay * (c * ax - s * ay)
        ay_new = decay * (s * ax + c * ay)

        return np.array([px_new, py_new, vx_new, vy_new, ax_new, ay_new, normalize_angle(yaw + psi), yaw_rate, max(0.1, l), max(0.1, w), z, max(0.1, h)])

    def make_process_noise(self, dt: float) -> np.ndarray:
        q = np.zeros((DIM_X, DIM_X))
        yaw = float(self.ukf.x[YAW_IDX])
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s], [s, c]])
        sig_long = self.process_jerk_std**2
        sig_lat = (self.process_jerk_std * self.process_jerk_lat_scale)**2
        jerk_cov = Rz @ np.diag([sig_long, sig_lat]) @ Rz.T
        qva = np.array([
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt]])
        idx = ((0, 2, 4), (1, 3, 5))
        for a in range(2):
            for b in range(2):
                q[np.ix_(idx[a], idx[b])] += jerk_cov[a, b] * qva

        qya = self.process_yaw_accel_std**2
        q[np.ix_((6, 7), (6, 7))] += qya * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        q[8, 8] = self.process_l_std**2 * dt
        q[9, 9] = self.process_w_std**2 * dt
        q[10, 10] = self.process_z_std**2 * dt
        q[11, 11] = self.process_h_std**2 * dt
        return 0.5 * (q + q.T)

    @staticmethod
    def hx_lidar(x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1], x[6], x[8], x[9], x[10], x[11]])

    @staticmethod
    def hx_velocity(x: np.ndarray) -> np.ndarray:
        return np.array([x[2], x[3]])

    @staticmethod
    def hx_radar(x: np.ndarray, u_world: np.ndarray) -> np.ndarray:
        u = np.asarray(u_world, dtype=float)
        norm = np.linalg.norm(u[:2])
        if norm < 1e-9:
            return np.array([0.0])
        return np.array([(x[2] * u[0] + x[3] * u[1]) / norm])

    def predict(self, dt: Optional[float] = None) -> None:
        dt = self.dt if dt is None else dt
        self.ukf.Q = self.make_process_noise(dt)
        self.ukf.predict(dt=dt)
        self.ukf.x[YAW_IDX] = normalize_angle(self.ukf.x[YAW_IDX])
        self.stabilize_covariance()
        self._predicted_sigmas_valid = True

    def _resolve_heading(self, measured_yaw: float) -> float:
        x = self.ukf.x
        measured = normalize_angle(measured_yaw)
        if math.hypot(x[2], x[3]) >= self.yaw_align_min_speed:
            course = math.atan2(x[3], x[2])
            if abs(normalize_angle(x[YAW_IDX] - course)) > math.pi / 2:
                x[YAW_IDX] = normalize_angle(x[YAW_IDX] + math.pi)
                self._predicted_sigmas_valid = False   
            if abs(normalize_angle(measured - course)) > math.pi / 2:
                measured = normalize_angle(measured + math.pi)
            return measured
        if abs(normalize_angle(measured - x[YAW_IDX])) > math.pi / 2:
            measured = normalize_angle(measured + math.pi)
        return measured

    def _sigma_points_for_update(self) -> np.ndarray:
        if self._predicted_sigmas_valid:    return self.ukf.sigmas_f
        return self.ukf.points_fn.sigma_points(self.ukf.x, self.ukf.P)

    def _unscented_update(self, z: np.ndarray, hx, R: np.ndarray, angle_idx: Optional[int] = None) -> None:
        ukf = self.ukf
        sigmas_f = self._sigma_points_for_update()
        Wm, Wc = ukf.Wm, ukf.Wc
        n_sigmas = sigmas_f.shape[0]
        dim_z = len(z)

        sigmas_h = np.array([hx(sigmas_f[i]) for i in range(n_sigmas)])
        zp = self._z_mean(sigmas_h, Wm, angle_idx)

        Pz = R.copy()
        Pxz = np.zeros((DIM_X, dim_z))
        for i in range(n_sigmas):
            z_res = self._z_residual(sigmas_h[i], zp, angle_idx)
            x_res = self.state_residual(sigmas_f[i], ukf.x)
            Pz += Wc[i] * np.outer(z_res, z_res)
            Pxz += Wc[i] * np.outer(x_res, z_res)
        Pz = 0.5 * (Pz + Pz.T)

        try:
            L = np.linalg.cholesky(Pz)
            SI = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(dim_z)))
        except np.linalg.LinAlgError:
            SI = np.linalg.pinv(Pz)

        K = Pxz @ SI
        y = self._z_residual(z, zp, angle_idx)

        ukf.x = ukf.x + K @ y
        ukf.P = ukf.P - K @ Pz @ K.T
        ukf.P = 0.5 * (ukf.P + ukf.P.T)
        ukf.K, ukf.y, ukf.z, ukf.S, ukf.SI, ukf.sigmas_h = K, y, z, Pz, SI, sigmas_h
        self._predicted_sigmas_valid = False

    def _finalize_state(self) -> None:
        self.ukf.x[YAW_IDX] = normalize_angle(self.ukf.x[YAW_IDX])
        self.ukf.x[8] = max(0.1, float(self.ukf.x[8]))
        self.ukf.x[9] = max(0.1, float(self.ukf.x[9]))
        self.ukf.x[11] = max(0.1, float(self.ukf.x[11]))
        self.stabilize_covariance()

    def _radar_innovation_ok(self, v_r_abs: float, u_world: np.ndarray) -> bool:
        sigmas = self._sigma_points_for_update()
        sigmas_h = np.array([self.hx_radar(sigmas[i], u_world) for i in range(sigmas.shape[0])])
        Wm, Wc = self.ukf.Wm, self.ukf.Wc
        zp = float(np.sum(sigmas_h[:, 0] * Wm))
        Pz = float(self.radarR[0, 0] + np.sum(Wc * (sigmas_h[:, 0] - zp) ** 2))
        nis = (v_r_abs - zp) ** 2 / max(Pz, 1e-9)
        return nis <= self.radar_gate_threshold

    def _gate_radar(self, radar_meas_list: list[tuple]) -> list[tuple]:
        valid = []
        for v_r_abs, u_world in radar_meas_list:
            u_world = np.asarray(u_world, dtype=float)
            v_r_abs = float(v_r_abs)
            if np.linalg.norm(u_world[:2]) < 1e-6 or not np.isfinite(v_r_abs):
                continue
            if not self._radar_innovation_ok(v_r_abs, u_world):
                continue
            valid.append((v_r_abs, u_world))
        return valid

    def update_combined(self, lidar_meas: tuple, radar_meas_list: list[tuple],
                        vel_meas: tuple) -> None:
        cx, cy, yaw, l, w, cz, h = lidar_meas
        yaw = self._resolve_heading(yaw)
        l, w = max(0.1, float(l)), max(0.1, float(w))
        h = max(0.1, float(h))
        valid_radar = self._gate_radar(radar_meas_list)
        z_parts = [np.array([cx, cy, yaw, l, w, cz, h])]
        R_blocks = [self.lidarR]
        z_parts.append(np.asarray(vel_meas, dtype=float).reshape(2))
        R_blocks.append(self.detVelR)
        for v_r_abs, _ in valid_radar:
            z_parts.append(np.array([v_r_abs]))
            R_blocks.append(self.radarR)

        z = np.concatenate(z_parts)
        R = block_diag(*R_blocks)

        def hx(x):
            out = [self.hx_lidar(x)]
            out.append(self.hx_velocity(x))
            for _, u_world in valid_radar:
                out.append(self.hx_radar(x, u_world))
            return np.concatenate(out)

        self._unscented_update(z, hx, R, angle_idx=2)
        self._finalize_state()

    def fuse_radar_only(self, radar_meas_list: list[tuple]) -> None:
        valid_radar = self._gate_radar(radar_meas_list)
        if not valid_radar:
            return
        z = np.concatenate([np.array([v]) for v, _ in valid_radar])
        R = block_diag(*[self.radarR for _ in valid_radar])

        def hx(x):
            return np.concatenate([self.hx_radar(x, u) for _, u in valid_radar])

        self._unscented_update(z, hx, R, angle_idx=None)
        self._finalize_state()

    @property
    def position(self) -> tuple:
        return float(self.ukf.x[0]), float(self.ukf.x[1])

    @property
    def vx(self) -> float:
        return float(self.ukf.x[2])

    @property
    def vy(self) -> float:
        return float(self.ukf.x[3])

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def ax(self) -> float:
        return float(self.ukf.x[4])

    @property
    def ay(self) -> float:
        return float(self.ukf.x[5])

    @property
    def yaw(self) -> float:
        return float(normalize_angle(self.ukf.x[YAW_IDX]))

    @property
    def yaw_rate(self) -> float:
        return float(self.ukf.x[7])

    @property
    def acceleration(self) -> float:
        speed = self.speed
        if speed > 1e-3:
            return (self.vx * self.ax + self.vy * self.ay) / speed
        return self.ax * math.cos(self.yaw) + self.ay * math.sin(self.yaw)

    @property
    def sideslip(self) -> float:
        if self.speed < self.yaw_align_min_speed * 0.5:
            return 0.0
        return float(normalize_angle(math.atan2(self.vy, self.vx) - self.yaw))

    @property
    def dims(self) -> tuple:
        return float(self.ukf.x[8]), float(self.ukf.x[9])

    @property
    def z(self) -> float:
        return float(self.ukf.x[10])

    @property
    def height(self) -> float:
        return float(self.ukf.x[11])

    @property
    def covariance(self) -> np.ndarray:
        return self.ukf.P

class UKFTrackBB2D:
    def __init__(self, track_id: int, detection: bev_Det, dt: float, max_history: int = 20,
                 window_size: int = 10, min_motion_speed: float = 0.5, flip_tolerance_deg: float = 25.0,
                 stationary_speed_thresh: float = 0.2, min_hits_to_confirm: int = 1):
        self.id = track_id
        self.dt = dt
        self.ukf = ObjectUKF(dt=dt, x0=detection.center_x, y0=detection.center_y, yaw0=math.radians(detection.yaw), 
                              l0=detection.length, w0=detection.width, z0=detection.center_z, h0=detection.height,
                              vx0=detection.vx, vy0=detection.vy)
        self.yaw = self.ukf.yaw
        self.bbs = [detection]
        self.len = detection.length
        self.wid = detection.width
        self.z = detection.center_z
        self.height = detection.height
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
        self.pos_history: list[tuple[float, float, float]] = []
        self.slip_angle_deg = 0.0
        self.flip_tolerance_deg = flip_tolerance_deg
        self.min_motion_speed = min_motion_speed
        self.stationary_speed_thresh = stationary_speed_thresh
        self.window_size = window_size
        self.max_history = max_history
        self.hits = 1
        self.confirmed = self.hits >= min_hits_to_confirm

    def update_metadata(self, timestamp: float) -> None:
        self.px, self.py = self.ukf.position
        self.len, self.wid = self.ukf.dims
        self.z = self.ukf.z
        self.height = self.ukf.height
        self.yaw = self.ukf.yaw
        self.yaw_rate = self.ukf.yaw_rate
        if self.ukf.speed < self.stationary_speed_thresh:
            self.vx = self.vy = self.speed = 0.0
            self.ax = self.ay = self.acceleration = 0.0
        else:
            self.vx = self.ukf.vx
            self.vy = self.ukf.vy
            self.speed = self.ukf.speed
            self.acceleration = self.ukf.acceleration
            self.ax = self.ukf.ax
            self.ay = self.ukf.ay
        if len(self.pos_history) > self.max_history: self.pos_history.pop(0)
        self.pos_history.append((self.px, self.py, timestamp))
        self.slip_angle_deg = math.degrees(self.ukf.sideslip)

    def get_all_data(self) -> trackedbb:
        return trackedbb(px=self.px, py=self.py, moveable=True,
                vx=self.vx, vy=self.vy, speed=self.speed,
                acceleration=self.acceleration, ax=self.ax, ay=self.ay,
                yaw=self.yaw, yaw_rate=self.yaw_rate, len=self.len, wid=self.wid,
                z=self.z, height=self.height,
                pos_history=self.pos_history, slip_angle_deg=self.slip_angle_deg)

class UKFBoundingBoxTracker:
    def __init__(self, max_match_dist_m: float = 3.0, max_missed_ticks: int = 10, dt: float = 0.05,
                 gate_prob: float = 0.95, yaw_weight: float = 0.15, size_weight: float = 0.1,
                 min_hits_to_confirm: int = 5, max_tentative_misses: int = 3):
        self.tracks: list[UKFTrackBB2D] = []
        self._next_id = 0
        self.max_match_dist_m = max_match_dist_m
        self.max_missed_ticks = max_missed_ticks
        self.gate_threshold = chi2.ppf(gate_prob, df=2)
        self.dt = dt
        self.yaw_weight = yaw_weight
        self.size_weight = size_weight
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_tentative_misses = max_tentative_misses

    @property
    def visible_tracks(self) -> list[UKFTrackBB2D]:
        return [t for t in self.tracks if t.confirmed]

    def _spawn_track(self, detection, step: float, dt: float) -> None:
        track = UKFTrackBB2D(self._next_id, detection, dt, min_hits_to_confirm=self.min_hits_to_confirm)
        self._next_id += 1
        radar_meas_list = [(float(r[3]), np.asarray(r[4:7])) for r in detection.radar_points]
        track.ukf.fuse_radar_only(radar_meas_list)
        track.update_metadata(step * dt)
        self.tracks.append(track)

    def associate(self, detections):
        if not self.tracks or not detections:
            return [], set(), set()
        cost = np.full((len(self.tracks), len(detections)), 1e6, dtype=float)
        for i, track in enumerate(self.tracks):
            predicted = track.ukf.ukf.x[:2]
            S_pos = track.ukf.ukf.P[:2, :2] + track.ukf.lidarR[:2, :2]
            S_pos = 0.5 * (S_pos + S_pos.T)
            try:
                L = np.linalg.cholesky(S_pos)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(S_pos + np.eye(2) * 1e-6)
            track_yaw = track.ukf.yaw
            track_len, track_wid = track.ukf.dims
            for j, det in enumerate(detections):
                d = np.array([det.center_x, det.center_y]) - predicted
                y = np.linalg.solve(L, d)
                m_dist_sq = float(y @ y)
                if m_dist_sq > self.gate_threshold: continue 
                yaw_diff = normalize_angle(math.radians(det.yaw) - track_yaw)
                yaw_diff = min(abs(yaw_diff), abs(normalize_angle(yaw_diff + math.pi)))  # bbox 180 deg flip
                size_diff = ((det.length - track_len) / max(track_len, 0.1)) ** 2 + \
                            ((det.width - track_wid) / max(track_wid, 0.1)) ** 2
                cost[i, j] = m_dist_sq + self.yaw_weight * yaw_diff ** 2 + self.size_weight * size_diff
        rows, cols = linear_sum_assignment(cost)
        matches, matched_tracks, matched_detections = [], set(), set()
        for i, j in zip(rows, cols):
            if cost[i, j] >= 1e6:
                continue
            matches.append((i, j))
            matched_tracks.add(i)
            matched_detections.add(j)
        return matches, matched_tracks, matched_detections

    def update(self, detections, step: float, dt: Optional[float] = None):
        dt = self.dt if dt is None else dt
        timestamp = step * dt
        for track in self.tracks:
            track.ukf.predict(dt=dt)
        if not detections:
            self.prune(None)
            for track in self.tracks:
                track.update_metadata(timestamp)
            return self.visible_tracks
        if not self.tracks:
            for detection in detections:
                self._spawn_track(detection, step, dt)
            return self.visible_tracks
        matches, matched_tracks, matched_detections = self.associate(detections)
        for (track_idx, det_idx) in matches:
            track: UKFTrackBB2D = self.tracks[track_idx]
            detection = detections[det_idx]
            radar_meas_list = [(float(r[3]), np.asarray(r[4:7])) for r in detection.radar_points]
            vel_meas = (detection.vx, detection.vy)
            track.ukf.update_combined((detection.center_x, detection.center_y, math.radians(detection.yaw),
                        detection.length, detection.width, detection.center_z, detection.height),
                        radar_meas_list, vel_meas)
            track.bbs.append(detection)
            if len(track.bbs) > 10:
                track.bbs.pop(0)
            track.missed_ticks = 0
            track.hits += 1
            if not track.confirmed and track.hits >= self.min_hits_to_confirm:
                track.confirmed = True
            track.update_metadata(timestamp)
        self.prune(matched_tracks)
        for track in self.tracks:
            if track.missed_ticks > 0:
                track.update_metadata(timestamp)
        for j, detection in enumerate(detections):
            if j not in matched_detections:
                self._spawn_track(detection, step, dt)
        return self.visible_tracks

    def prune(self, matched_tracks):
        if matched_tracks:
            for i, track in enumerate(self.tracks):
                if i not in matched_tracks:
                    track.missed_ticks += 1
        else:
            for track in self.tracks:
                track.missed_ticks += 1
        self.tracks = [
            track for track in self.tracks
            if (track.missed_ticks <= self.max_missed_ticks if track.confirmed
                else track.missed_ticks < self.max_tentative_misses)]
