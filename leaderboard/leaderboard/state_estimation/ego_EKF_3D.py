from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import carla
import leaderboard.state_estimation.hyperparams as hp
from leaderboard.state_estimation.sensor_dataclasses import Ego3DState
from leaderboard.state_estimation.carla_lida_loc import (CarlaPoseGraphSLAM, AsyncSlamWorker, SlamOutput)

STATE_SIZE = 16
UNSET = object()


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

def small_angle_quat(dtheta: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(dtheta))
    if angle < 1e-10:
        return quat_normalize(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]], dtype=float))
    axis = dtheta / angle
    half = 0.5 * angle
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)

def quat_to_R(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_normalize(q)
    return np.array([
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)],
    ], dtype=float)

def R_to_quat(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z], dtype=float)
    return -q if q[0] < 0 else q

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                  cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy], dtype=float)
    return quat_normalize(q)

def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    qw, qx, qy, qz = quat_normalize(q)
    roll = math.atan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
    pitch = math.asin(max(-1.0, min(1.0, 2.0*(qw*qy - qz*qx))))
    yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return roll, pitch, yaw

def carla_mount_to_flu(pos_carla, rpy_deg_carla) -> Tuple[np.ndarray, np.ndarray]:
    t = np.array([pos_carla[0], -pos_carla[1], pos_carla[2]], dtype=float)
    roll, pitch, yaw = (math.radians(a) for a in rpy_deg_carla)
    return quat_to_R(euler_to_quat(roll, -pitch, -yaw)), t

class SpikeFilter:
    def __init__(self, thresh: float):
        self.thresh = thresh
        self.buf = deque(maxlen=3)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self.buf.append(x.copy())
        if len(self.buf) < 3:
            return x
        med = np.median(np.stack(self.buf), axis=0)
        out = x.copy()
        bad = np.abs(x - med) > self.thresh
        out[bad] = med[bad]
        return out

class EgoErrorStateEKF3D:
    """
    3D quaternion error-state EKF for CARLA ego-state estimation.
    Nominal state : p_w, v_w, q_bw, gyro_bias, accel_bias, wheel_scale
    Error state   : dp, dv, dtheta, dbg, dba, dscale   (16 = 3+3+3+3+3+1)  [dx = truth - estimate]
    IMU           : nominal-state propagation
    Updates       : GNSS position (with lever arm), compass yaw, wheel speed (scale-factor model) +
                    lateral non-holonomic constraint, ZUPT (+ gyro-bias) when stationary,
                    gravity direction when slow, LiDAR-SLAM (relative [default] or absolute).
    """
    def __init__(self, vehicle, world_map, use_gravity_update: bool = True,
                 slam_worker=None, geo_ref=None, seed: Optional[int] = None):
        trans = vehicle.get_transform()
        q0 = euler_to_quat(math.radians(trans.rotation.roll), math.radians(-trans.rotation.pitch),
                           math.radians(-trans.rotation.yaw))
        self.p = np.array([trans.location.x, -trans.location.y, trans.location.z], dtype=float)
        self.v_world = np.zeros(3)
        self.a_body = np.zeros(3)
        self.gyro = np.zeros(3)
        self.q = q0.copy()
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.wheel_scale = 0.0

        gx, gy, gz = hp.GNSS_POS_CARLA
        self.l_gnss = np.array([gx, -gy, gz], dtype=float)
        self.R_lidar, self.t_lidar = carla_mount_to_flu(hp.LIDAR_POS_CARLA, hp.LIDAR_RPY_DEG_CARLA)

        T0 = np.eye(4)
        T0[0:3, 0:3] = quat_to_R(q0)
        T0[0:3, 3] = self.p
        if slam_worker is None:
            engine = CarlaPoseGraphSLAM(T0, extrinsic_R=self.R_lidar, extrinsic_t=self.t_lidar)
            slam_worker = AsyncSlamWorker(engine, synchronous=hp.SLAM_SYNCHRONOUS)
            slam_worker.start()
        self.slam = slam_worker
        self.slam_log = []                 
        self._slam_ref = None              
        self._frame_pose = {}              
        self._frame_time = {}              

        self.P = np.diag([
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            math.radians(5.0)**2, math.radians(5.0)**2, math.radians(10.0)**2,
            math.radians(2.0)**2, math.radians(2.0)**2, math.radians(2.0)**2,
            0.4**2, 0.4**2, 0.4**2, hp.SIGMA_WHEEL_SCALE_P0**2,
        ]).astype(float)

        self.t = 0.0
        self.dt = 0.05
        self.history = deque(maxlen=hp.HISTORY_LEN)   
        self.history_gnss = deque(maxlen=hp.MAX_HISTORY)
        self.nis_history = []
        self.log_innovations = True
        self._rej = {}                                

        self.last_gnss_frame: Optional[int] = None
        self.map = world_map
        self.use_gravity_update = use_gravity_update
        self.prev_state: Optional[Ego3DState] = None
        self.accel_history = []
        self.ACCEL_HISTORY_LEN = 10
        self.GRAVITY_VAR_TOL = 0.05

        self._accel_spike = SpikeFilter(hp.ACCEL_SPIKE_THRESH)
        self._gyro_spike = SpikeFilter(hp.GYRO_SPIKE_THRESH)
        self._imu_win = deque(maxlen=hp.ZUPT_WINDOW)
        self.stationary = False
        self.last_wheel_speed = None
        self._a_world_lpf = None
        self._rng = np.random.default_rng(seed)

        if geo_ref is not None:
            self._geo_ref = geo_ref
        else:
            self._geo_ref = self.map.transform_to_geolocation(carla.Location(0.0, 0.0, 0.0))

    def stabilize_covariance(self) -> None:
        P = 0.5 * (self.P + self.P.T)
        eig_min = float(np.min(np.linalg.eigvalsh(P)))
        if eig_min < 1e-10:
            P += np.eye(STATE_SIZE) * (1e-10 - eig_min)
        self.P = 0.5 * (P + P.T)

    def _discretize_process_noise(self, Fc, Gc, Qc_diag, dt):
        van_loan = np.zeros((2 * STATE_SIZE, 2 * STATE_SIZE), dtype=float)
        van_loan[:STATE_SIZE, :STATE_SIZE] = Fc
        van_loan[:STATE_SIZE, STATE_SIZE:] = Gc @ np.diag(Qc_diag) @ Gc.T
        van_loan[STATE_SIZE:, STATE_SIZE:] = -Fc.T
        van_loan *= dt
        M = expm(van_loan)
        Phi = M[:STATE_SIZE, :STATE_SIZE]
        Qd = M[:STATE_SIZE, STATE_SIZE:] @ Phi.T
        Qd = 0.5 * (Qd + Qd.T)
        eigvals = np.linalg.eigvalsh(Qd)
        if float(np.min(eigvals)) < 0.0:
            Qd += np.eye(STATE_SIZE) * max(0.0, -float(np.min(eigvals)) + 1e-14)
        return Phi, Qd

    def nominal_copy(self) -> dict:
        return {"p": self.p.copy(), "v": self.v_world.copy(), "q": self.q.copy(), "bg": self.bg.copy(),
                "ba": self.ba.copy(), "wheel_scale": float(self.wheel_scale)}

    def inject_error(self, dx: np.ndarray) -> None:
        dx = np.asarray(dx, dtype=float)
        self.p += dx[0:3]
        self.v_world += dx[3:6]
        self.q = quat_normalize(quat_mul(self.q, small_angle_quat(dx[6:9])))
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        self.wheel_scale += float(dx[15])

    @staticmethod
    def carla_imu_to_body(imu_data) -> Tuple[np.ndarray, np.ndarray]:
        gyro_carla = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z], dtype=float)
        accel_carla = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z],
                               dtype=float)
        return np.diag([1.0, -1.0, 1.0]) @ accel_carla, np.diag([-1.0, 1.0, -1.0]) @ gyro_carla

    def propagate(self, dt: float) -> None:
        if dt <= 0.0:
            return
        R = quat_to_R(self.q)
        a_world = R @ self.a_body + np.array([0.0, 0.0, -hp.G], dtype=float)

        self.p += self.v_world * dt + 0.5 * a_world * dt * dt
        self.v_world += a_world * dt
        self.q = quat_normalize(quat_mul(self.q, small_angle_quat(self.gyro * dt)))

        Fc = np.zeros((STATE_SIZE, STATE_SIZE))
        Fc[0:3, 3:6] = np.eye(3)
        Fc[3:6, 6:9] = -R @ skew(self.a_body)
        Fc[3:6, 12:15] = -R
        Fc[6:9, 6:9] = -skew(self.gyro)
        Fc[6:9, 9:12] = -np.eye(3)

        Gc = np.zeros((STATE_SIZE, 13))
        Gc[3:6, 0:3] = R
        Gc[6:9, 3:6] = np.eye(3)
        Gc[9:12, 6:9] = np.eye(3)
        Gc[12:15, 9:12] = np.eye(3)
        Gc[15, 12] = 1.0

        Qc_diag = np.array([
            hp.SIGMA_ACCEL**2, hp.SIGMA_ACCEL**2, hp.SIGMA_ACCEL**2,
            hp.SIGMA_GYRO**2, hp.SIGMA_GYRO**2, hp.SIGMA_GYRO**2,
            hp.SIGMA_GYRO_BIAS_RW**2, hp.SIGMA_GYRO_BIAS_RW**2, hp.SIGMA_GYRO_BIAS_RW**2,
            hp.SIGMA_ACCEL_BIAS_RW**2, hp.SIGMA_ACCEL_BIAS_RW**2, hp.SIGMA_ACCEL_BIAS_RW**2,
            hp.SIGMA_WHEEL_SCALE_RW**2,
        ], dtype=float)
        F, Qd = self._discretize_process_noise(Fc, Gc, Qc_diag, dt)
        self.P = F @ self.P @ F.T + Qd
        self.stabilize_covariance()

    def _apply_error_reset(self, dtheta: np.ndarray) -> None:
        G_reset = np.eye(STATE_SIZE)
        G_reset[6:9, 6:9] = np.eye(3) - 0.5 * skew(dtheta)
        self.P = G_reset @ self.P @ G_reset.T

    def update_error_state(self, z, z_pred, H, R, residual_fn=None, sensor_name: str = "unknown",
                           gate: str = "soft") -> bool:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        innovation = (z - np.atleast_1d(z_pred)) if residual_fn is None else np.atleast_1d(residual_fn(z, z_pred))
        dof = innovation.size
        S = H @ self.P @ H.T + R
        try:
            nis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            nis = float("inf")
        thr = hp.CHI2_999.get(dof, 3.0 * dof + 7.0)

        accepted, scale = True, 1.0
        if not np.isfinite(nis):
            accepted = False
        elif gate == "soft":
            scale = 1.0 if nis <= thr else min(nis / thr, hp.MAX_R_SCALE)
        elif gate == "hard":
            if nis > thr:
                self._rej[sensor_name] = self._rej.get(sensor_name, 0) + 1
                if self._rej[sensor_name] > hp.GATE_MAX_CONSEC_REJECT:
                    scale = min(nis / thr, hp.MAX_R_SCALE)
                    self._rej[sensor_name] = 0
                else:
                    accepted = False
            else:
                self._rej[sensor_name] = 0

        self.nis_history.append({
            "sensor": sensor_name, "t": self.t, "nis": nis, "dof": int(dof), "scale": float(scale),
            "accepted": bool(accepted),
            **({"innov": innovation.tolist()} if self.log_innovations else {})})
        if not accepted:
            return False

        R_eff = R * scale
        PHt = self.P @ H.T
        S = H @ PHt + R_eff
        K = np.linalg.solve(S, PHt.T).T
        dx = K @ innovation

        I_KH = np.eye(STATE_SIZE) - K @ H
        P_new = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (P_new + P_new.T)

        dtheta = dx[6:9].copy()
        self.inject_error(dx)
        self._apply_error_reset(dtheta)
        self.stabilize_covariance()
        return True

    def h_gnss(self) -> np.ndarray:
        return self.p + quat_to_R(self.q) @ self.l_gnss

    def H_gnss(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE))
        H[:, 0:3] = np.eye(3)
        H[:, 6:9] = -quat_to_R(self.q) @ skew(self.l_gnss)
        return H

    def h_compass(self) -> np.ndarray:
        _, _, yaw = quat_to_euler(self.q)
        return np.array([yaw], dtype=float)

    def H_compass(self) -> np.ndarray:
        H = np.zeros((1, STATE_SIZE))
        R = quat_to_R(self.q)
        x, y = R[0, 0], R[1, 0]
        den = max(1e-8, x**2 + y**2)
        dy_dth = np.array([0.0, -R[1, 2], R[1, 1]])
        dx_dth = np.array([0.0, -R[0, 2], R[0, 1]])
        H[0, 6:9] = (x * dy_dth - y * dx_dth) / den
        return H

    def h_wheel(self) -> np.ndarray:
        v_body = quat_to_R(self.q).T @ self.v_world
        return np.array([(1.0 + self.wheel_scale) * v_body[0], v_body[1]], dtype=float)

    def H_wheel(self) -> np.ndarray:
        H = np.zeros((2, STATE_SIZE))
        R_bw = quat_to_R(self.q)
        v_body = R_bw.T @ self.v_world
        H[:, 3:6] = R_bw.T[:2, :]
        H[0:2, 6:9] = skew(v_body)[0:2, :]
        H[0, 3:9] *= (1.0 + self.wheel_scale)
        H[0, 15] = v_body[0]
        return H

    def h_H_gravity_direction(self):
        H = np.zeros((3, STATE_SIZE))
        v = -(quat_to_R(self.q).T @ np.array([0.0, 0.0, -hp.G]))
        norm_v = float(np.linalg.norm(v))
        if norm_v < 1e-12:
            return np.array([0.0, 0.0, 1.0]), H
        h_g = v / norm_v
        H[:, 6:9] = ((np.eye(3) - np.outer(h_g, h_g)) / norm_v) @ skew(v)
        return h_g, H

    def h_vel_world(self) -> np.ndarray:
        return self.v_world.copy()

    def H_vel_world(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE))
        H[:, 3:6] = np.eye(3)
        return H

    def h_slam_pos(self) -> np.ndarray:
        return self.p.copy()

    def H_slam_pos(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE))
        H[:, 0:3] = np.eye(3)
        return H

    def h_slam_att(self) -> np.ndarray:
        return self.q.copy()

    def H_slam_att(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE))
        H[:, 6:9] = np.eye(3)
        return H

    def update_gnss(self, gnss_data) -> None:
        if self.last_gnss_frame is not None and gnss_data.frame == self.last_gnss_frame:
            return

        def _mercator(lat, lon, scale):
            mx = scale * math.radians(lon) * hp.EARTH_RADIUS_EQUA
            my = scale * hp.EARTH_RADIUS_EQUA * math.log(math.tan(math.radians(90.0 + lat) / 2.0))
            return mx, my

        ref = self._geo_ref
        scale = math.cos(math.radians(ref.latitude))
        mx, my = _mercator(gnss_data.latitude, gnss_data.longitude, scale)
        mx0, my0 = _mercator(ref.latitude, ref.longitude, scale)
        x_c, y_c, z_c = mx - mx0, -(my - my0), gnss_data.altitude      # CARLA coordinates
        z_meas = np.array([x_c, -y_c, z_c], dtype=float)               # EKF frame

        self.history_gnss.append((z_meas[0], z_meas[1], z_meas[2], self.t))
        self.update_error_state(z=z_meas, z_pred=self.h_gnss(), H=self.H_gnss(), R=hp.R_GNSS,
                                sensor_name="gnss", gate="hard")
        self.last_gnss_frame = gnss_data.frame
        if hp.USE_GNSS_VELOCITY:
            self.update_gnss_velocity()

    def update_compass(self, imu_data) -> None:
        yaw_measurement = wrap(math.pi / 2.0 - float(imu_data.compass))
        self.update_error_state(
            z=np.array([yaw_measurement]), z_pred=self.h_compass(), H=self.H_compass(), R=hp.R_COMPASS,
            residual_fn=lambda a, b: np.array([wrap(float(a[0] - b[0]))]),
            sensor_name="compass", gate="hard")

    def update_wheel(self, speed_measurement: Optional[float]) -> None:
        if speed_measurement is None:
            return
        z = np.array([float(speed_measurement), 0.0])
        for i in range(2):
            z_pred = self.h_wheel()
            H = self.H_wheel()
            self.update_error_state(
                z=np.array([z[i]]), z_pred=np.array([z_pred[i]]), H=H[i:i + 1, :],
                R=np.array([[hp.R_WHEEL[i, i]]]),
                sensor_name="wheel_longitudinal" if i == 0 else "wheel_lateral", gate="soft")

    def detect_stationary(self, wheel_speed: Optional[float], accel_raw: np.ndarray, gyro_raw: np.ndarray) -> bool:
        self._imu_win.append(np.concatenate([accel_raw, gyro_raw]))
        if len(self._imu_win) < hp.ZUPT_WINDOW:
            return False
        w = np.asarray(self._imu_win)
        if float(np.var(w[:, :3], axis=0).sum()) > hp.ZUPT_ACCEL_VAR:
            return False
        if np.linalg.norm(w[:, 3:].mean(axis=0) - self.bg) > hp.ZUPT_GYRO_MAX:
            return False
        if wheel_speed is not None:
            return abs(wheel_speed) <= hp.ZUPT_WHEEL_MAX
        v_body = quat_to_R(self.q).T @ self.v_world
        return self.stationary and float(np.linalg.norm(v_body)) < hp.ZUPT_FILTER_SPEED_MAX

    def update_zupt(self, gyro_raw: np.ndarray) -> None:
        """Zero velocity + (gyro reads only its bias) while standing still."""
        self.update_error_state(z=np.zeros(3), z_pred=self.h_vel_world(), H=self.H_vel_world(),
                                R=hp.R_ZUPT_VEL, sensor_name="zupt_vel", gate="none")
        H = np.zeros((3, STATE_SIZE))
        H[:, 9:12] = np.eye(3)
        self.update_error_state(z=gyro_raw, z_pred=self.bg.copy(), H=H, R=hp.R_ZUPT_GYRO,
                                sensor_name="zupt_gyro", gate="none")

    def update_gravity(self, accel_body: np.ndarray) -> None:
        if not self.use_gravity_update:
            return
        self.accel_history.append(accel_body.copy())
        accel_norm = float(np.linalg.norm(accel_body))
        if len(self.accel_history) > self.ACCEL_HISTORY_LEN:
            self.accel_history.pop(0)
        if (len(self.accel_history) < self.ACCEL_HISTORY_LEN
                or float(np.var(self.accel_history, axis=0).sum()) > self.GRAVITY_VAR_TOL
                or abs(accel_norm - hp.G) > hp.GRAVITY_NORM_TOL
                or np.linalg.norm(quat_to_R(self.q).T @ self.v_world) > hp.GRAVITY_UPDATE_MAX_SPEED):
            return
        measured_norm = np.linalg.norm(accel_body)
        if measured_norm < 1e-9:
            return
        measured_dir = accel_body / measured_norm
        dynamic_excess = max(0.0, abs(accel_norm - hp.G))
        inflation = min(1.0 + hp.GRAVITY_DYNAMIC_SCALE * dynamic_excess / hp.G, 8.0)
        z_pred, H = self.h_H_gravity_direction()
        self.update_error_state(z=measured_dir, z_pred=z_pred, H=H, R=hp.R_GRAVITY * inflation,
                                sensor_name="gravity", gate="soft")

    def update_gnss_velocity(self) -> None:
        if len(self.history_gnss) < hp.MIN_HISTORY or self.stationary:
            return
        h = np.asarray(self.history_gnss, dtype=float)
        t = h[:, 3] - h[0, 3]
        v = np.array([np.polyfit(t, h[:, k], 1)[0] for k in range(3)])
        self.update_error_state(z=v, z_pred=self.h_vel_world(), H=self.H_vel_world(), R=hp.R_GNSS_VEL,
                                sensor_name="GNSS_vel", gate="soft")

    def _ekf_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[0:3, 0:3] = quat_to_R(self.q)
        T[0:3, 3] = self.p
        return T

    def update_slam(self, lidar_data, step, dynamic_boxes=None) -> None:
        if lidar_data is not None:
            self._frame_pose[step] = self._ekf_pose()
            self._frame_time[step] = self.t
            while len(self._frame_pose) > 600:
                old = next(iter(self._frame_pose))
                self._frame_pose.pop(old)
                self._frame_time.pop(old, None)
            if not self.stationary:                 
                self.slam.push_frame(lidar_data, self._frame_pose[step], step, dynamic_boxes)

        res = self.slam.get_latest_result()
        if res is None: return
        self.slam_log.append((step, res))
        if res.frame_id not in self._frame_pose:
            return
        if hp.SLAM_FUSION_MODE == "absolute":
            self._update_slam_absolute(res)
        else:
            self._update_slam_relative(res)

    def _window_stats(self, t0: float, t1: float):
        sel = [h for h in self.history if t0 + 1e-9 < h[0] <= t1 + 1e-9]
        if len(sel) < 2:
            return None
        dts = np.array([h[4] for h in sel])
        tot = float(dts.sum())
        if tot < 1e-6:
            return None
        v_mean = (np.array([h[2] for h in sel]) * dts[:, None]).sum(axis=0) / tot
        R_rel = np.eye(3)
        for h in sel:
            R_rel = R_rel @ Rotation.from_rotvec(h[3] * h[4]).as_matrix()
        return v_mean, R_rel, tot

    def _update_slam_relative(self, res: SlamOutput) -> None:
        t1 = self._frame_time[res.frame_id]
        T1e = self._frame_pose[res.frame_id]
        ref = self._slam_ref
        new_ref = {"t": t1, "p": res.p.copy(), "q": res.q.copy(), "epoch": res.map_epoch, "T": T1e}
        if (ref is None or ref["epoch"] != res.map_epoch or t1 <= ref["t"]
                or t1 - ref["t"] > hp.SLAM_REL_MAX_DT):
            self._slam_ref = new_ref
            return
        dt_w = t1 - ref["t"]
        if dt_w < hp.SLAM_REL_MIN_DT:   return                                   
        stats = self._window_stats(ref["t"], t1)
        if stats is None:
            self._slam_ref = new_ref
            return
        v_mean, R_rel_ekf, tot = stats

        R0s, R1s = quat_to_R(ref["q"]), quat_to_R(res.q)
        u_slam = R0s.T @ (res.p - ref["p"]) / dt_w
        w_slam = Rotation.from_matrix(R0s.T @ R1s).as_rotvec() / dt_w
        R0e = ref["T"][0:3, 0:3]
        u_ekf = R0e.T @ v_mean
        w_ekf = Rotation.from_matrix(R_rel_ekf).as_rotvec() / tot

        R_now = quat_to_R(self.q)
        Rm = R_now.T @ R0e                           
        y_u = Rm @ (u_slam - u_ekf)
        u_now = Rm @ u_ekf
        y_w = w_slam - w_ekf

        Hu = np.zeros((3, STATE_SIZE))
        Hu[:, 3:6] = R_now.T
        Hu[:, 6:9] = skew(u_now)
        Rm1 = R_now.T @ T1e[0:3, 0:3]
        S_now = Rm1 @ res.trans_shape @ Rm1.T
        sig_u2 = 2.0 * hp.SIGMA_SLAM_REL_POS**2 / dt_w**2 + hp.SLAM_REL_VEL_FLOOR**2
        self.update_error_state(z=y_u, z_pred=np.zeros(3), H=Hu, R=sig_u2 * S_now,
                                sensor_name="slam_rel_vel", gate="hard")

        Hw = np.zeros((3, STATE_SIZE))
        Hw[:, 9:12] = -np.eye(3)
        sig_w2 = 2.0 * hp.SIGMA_SLAM_REL_ROT**2 / dt_w**2 + hp.SLAM_REL_GYRO_FLOOR**2
        self.update_error_state(z=y_w, z_pred=np.zeros(3), H=Hw, R=np.eye(3) * sig_w2,
                                sensor_name="slam_rel_gyro", gate="hard")
        self._slam_ref = new_ref
        if hp.SLAM_ABS_TILT_Z:
            self._update_slam_tilt_z(res)

    def _update_slam_tilt_z(self, res: SlamOutput) -> None:
        T_then = self._frame_pose[res.frame_id]
        T_slam = np.eye(4)
        T_slam[0:3, 0:3] = quat_to_R(res.q)
        T_slam[0:3, 3] = res.p
        T = self._ekf_pose() @ np.linalg.inv(T_then) @ T_slam
        Ht = np.zeros((2, STATE_SIZE))
        Ht[0, 6] = Ht[1, 7] = 1.0
        self.update_error_state(z=R_to_quat(T[0:3, 0:3]), z_pred=self.h_slam_att(), H=Ht,
                                R=np.eye(2) * hp.SIGMA_SLAM_TILT**2,
                                residual_fn=lambda zm, zp: self._att_residual(zm, zp)[0:2],
                                sensor_name="slam_tilt", gate="hard")
        Hz = np.zeros((1, STATE_SIZE))
        Hz[0, 2] = 1.0
        self.update_error_state(z=np.array([T[2, 3]]), z_pred=np.array([self.p[2]]), H=Hz,
                                R=np.array([[hp.SIGMA_SLAM_Z**2]]), sensor_name="slam_z", gate="hard")

    @staticmethod
    def _att_residual(z_meas, z_pred):
        q_pred_inv = np.array([z_pred[0], -z_pred[1], -z_pred[2], -z_pred[3]], dtype=float)
        dq = quat_mul(q_pred_inv, z_meas)
        if dq[0] < 0.0:
            dq = -dq
        return 2.0 * dq[1:4]

    def _update_slam_absolute(self, res: SlamOutput) -> None:
        T_then = self._frame_pose[res.frame_id]
        T_slam = np.eye(4)
        T_slam[0:3, 0:3] = quat_to_R(res.q)
        T_slam[0:3, 3] = res.p
        T = self._ekf_pose() @ np.linalg.inv(T_then) @ T_slam
        p_meas, q_meas = T[0:3, 3], R_to_quat(T[0:3, 0:3])
        Rw = T_then[0:3, 0:3]                                   # weak ICP directions -> larger sigma
        R_pos = hp.SIGMA_SLAM_POS**2 * (Rw @ res.trans_shape @ Rw.T)
        self.update_error_state(z=p_meas, z_pred=self.h_slam_pos(), H=self.H_slam_pos(),
                                R=R_pos, sensor_name="slam_position", gate="hard")
        self.update_error_state(z=q_meas, z_pred=self.h_slam_att(), H=self.H_slam_att(), R=hp.R_SLAM_ATT,
                                residual_fn=self._att_residual, sensor_name="slam_attitude", gate="hard")

    @staticmethod
    def get_simulated_speedometer(vehicle, noise_std: float = None, quantization: float = None,
                                  scale_err: float = None, dropout_prob: float = None,
                                  rng: Optional[np.random.Generator] = None) -> Optional[float]:
        noise_std = hp.SPEEDO_NOISE_STD if noise_std is None else noise_std
        quantization = hp.SPEEDO_QUANT if quantization is None else quantization
        scale_err = hp.SPEEDO_SCALE_ERR if scale_err is None else scale_err
        dropout_prob = hp.SPEEDO_DROPOUT if dropout_prob is None else dropout_prob
        rng = rng or np.random.default_rng()
        if dropout_prob > 0.0 and rng.random() < dropout_prob:
            return None
        vel = vehicle.get_velocity()
        fwd = vehicle.get_transform().get_forward_vector()
        true_speed = float(np.dot([vel.x, vel.y, vel.z], [fwd.x, fwd.y, fwd.z]))
        if abs(true_speed) < 0.02:
            return 0.0
        meas = true_speed * (1.0 + scale_err)
        if noise_std > 0.0:
            meas += rng.normal(0.0, noise_std)
        if quantization > 0.0:
            meas = round(meas / quantization) * quantization
        return float(meas)

    def extract_state_from_filter(self) -> Ego3DState:
        roll, pitch, yaw = quat_to_euler(self.q)
        R_bw = quat_to_R(self.q)
        v_body = R_bw.T @ self.v_world
        speed = float(np.linalg.norm(v_body))
        a_world = R_bw @ self.a_body + np.array([0.0, 0.0, -hp.G])
        alpha = self.dt / (hp.ACCEL_OUT_LPF_TAU + self.dt) if hp.ACCEL_OUT_LPF_TAU > 0 else 1.0
        self._a_world_lpf = a_world.copy() if self._a_world_lpf is None else \
            (1.0 - alpha) * self._a_world_lpf + alpha * a_world
        a_w = self._a_world_lpf
        a_b = R_bw.T @ a_w                                   # body-frame kinematic acceleration
        zs = hp.OUT_Z_SIGN

        state = Ego3DState(
            x=float(self.p[0]), y=-float(self.p[1]), z=float(self.p[2]),
            roll_rad=float(roll), pitch_rad=-float(pitch), yaw_rad=-float(yaw),
            roll=math.degrees(float(roll)), pitch=math.degrees(-float(pitch)), yaw=math.degrees(-float(yaw)),
            roll_rate=float(self.gyro[0]), pitch_rate=-float(self.gyro[1]),
            yaw_rate=float(hp.OUT_YAW_RATE_SIGN * self.gyro[2]),
            vx_body=float(v_body[0]), vy_body=float(-v_body[1]), vz_body=float(v_body[2]),
            ax_body=float(a_b[0]), ay_body=float(-a_b[1]), az_body=float(zs * a_b[2]),
            speed=speed, bg_yaw=self.bg[2], bg_pitch=self.bg[1], bg_roll=self.bg[0],
            long_accel=float(a_b[0]), lat_accel=float(-a_b[1]), vert_accel=float(zs * a_w[2]),
            vx_world=float(self.v_world[0]), vy_world=float(-self.v_world[1]), vz_world=float(self.v_world[2]),
        )
        self.prev_state = state
        return state

    def compute(self, dt: float, vehicle, imu_data, gnss_data, lidar_data, step,
                wheel_speed=UNSET, dynamic_boxes=None) -> Ego3DState:
        self.t += dt
        self.dt = dt
        accel_raw, gyro_raw = self.carla_imu_to_body(imu_data)
        accel_body = np.clip(self._accel_spike(accel_raw), -hp.ACCEL_CLIP, hp.ACCEL_CLIP)
        gyro_body = np.clip(self._gyro_spike(gyro_raw), -hp.GYRO_CLIP_RAD, hp.GYRO_CLIP_RAD)
        if self.last_gnss_frame is None:
            gyro_body = np.zeros(3)

        self.gyro = gyro_body - self.bg
        self.a_body = accel_body - self.ba
        self.propagate(dt)

        if gnss_data is not None:
            self.update_gnss(gnss_data)
        if imu_data is not None:
            self.update_compass(imu_data)

        ss = (self.get_simulated_speedometer(vehicle, rng=self._rng) if wheel_speed is UNSET else wheel_speed)
        self.last_wheel_speed = ss
        self.update_wheel(ss)

        self.stationary = bool(hp.USE_ZUPT and self.last_gnss_frame is not None
                               and self.detect_stationary(ss, accel_body, gyro_body))
        if self.stationary:
            self.update_zupt(gyro_body)
        self.update_gravity(accel_body)

        self.history.append((self.t, quat_to_R(self.q), self.v_world.copy(), self.gyro.copy(), dt))
        self.update_slam(lidar_data, step, dynamic_boxes)
        return self.extract_state_from_filter()

    def nis_summary(self) -> dict:
        """Per-sensor consistency. mean_nis_per_dof should be ~1: >>1 -> R too optimistic (or bias),
        <<1 -> R too conservative. accept_rate shows how often the gate fires."""
        out = {}
        for r in self.nis_history:
            s = out.setdefault(r["sensor"], {"n": 0, "acc": 0, "sum": 0.0, "fin": 0, "dof": r["dof"]})
            s["n"] += 1
            s["acc"] += int(r["accepted"])
            if np.isfinite(r["nis"]):
                s["sum"] += r["nis"]
                s["fin"] += 1
        return {k: {"n": v["n"], "accept_rate": v["acc"] / v["n"],
                    "mean_nis_per_dof": (v["sum"] / (v["fin"] * v["dof"])) if v["fin"] else float("nan")}
                for k, v in out.items()}

    def shutdown(self) -> None:
        if hasattr(self.slam, "stop"):
            self.slam.stop()