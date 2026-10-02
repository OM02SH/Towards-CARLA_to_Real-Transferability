import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import carla
from scipy.linalg import expm
import leaderboard.state_estimation.hyperparams as hp
from leaderboard.state_estimation.sensor_dataclasses import Ego3DState
from leaderboard.state_estimation.carla_lida_loc import CarlaPoseGraphSLAM, AsyncSlamWorker, SlamUpdateResult

STATE_SIZE = 16

def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0],], dtype=float)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:   return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
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
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s,], dtype=float)

def quat_to_R(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_normalize(q)
    return np.array([
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw),],
        [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
    ], dtype=float)

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

class EgoErrorStateEKF3D:
    """
    3D quaternion error-state EKF for CARLA ego-state estimation.
    Nominal state: p_w, v_w, q_bw, gyro_bias, accel_bias, wheel_bias
    Error state: dp, dv, dtheta, dbg, dba, dbv
    IMU: Used for nominal-state propagation.
    Updates:
        GNSS position
        Compass yaw
        Wheel forward speed + lateral nonholonomic constraint
        Optional gravity-direction attitude correction when nearly static
    There is NO hard vz=0 constraint.
    """

    def __init__(self, vehicle: carla.Vehicle, world_map: carla.Map,use_gravity_update: bool = True):
        trans = vehicle.get_transform()
        q0 = euler_to_quat(math.radians(trans.rotation.roll), math.radians(-trans.rotation.pitch), math.radians(-trans.rotation.yaw))
        self.p = np.array([trans.location.x, -trans.location.y, trans.location.z],dtype=float)
        self.v_world = np.zeros(3, dtype=float)
        self.a_body = np.zeros(3, dtype=float)
        self.q = q0.copy()
        self.bg = np.zeros(3, dtype=float)
        self.ba = np.zeros(3, dtype=float)
        self.wheel_bias = 0.0
        self.history_gnss = []
        initial_transform = np.eye(4)
        initial_transform[0:3, 0:3] = quat_to_R(q0)
        initial_transform[0:3, 3] = self.p
        self.slam = AsyncSlamWorker(CarlaPoseGraphSLAM(initial_transform))
        self.slam.start()
        self.last_slam= np.zeros(3), np.zeros(4)

        self.P = np.diag([
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            math.radians(5.0)**2, math.radians(5.0)**2, math.radians(10.0)**2,
            math.radians(2.0)**2, math.radians(2.0)**2, math.radians(2.0)**2,
            0.4**2, 0.4**2, 0.4**2, 0.8**2,
        ]).astype(float)

        # Innovation statistics for tuning / thesis evaluation.
        self.nis_history = []

        self.last_gnss_frame: Optional[int] = None
        self.map = world_map
        self.use_gravity_update = use_gravity_update
        self.last_imu_timestamp: Optional[float] = None
        self.prev_state: Optional[Ego3DState] = None
        self.accel_history = []
        self.ACCEL_HISTORY_LEN = 10
        self.GRAVITY_VAR_TOL = 0.05
        self._geo_ref = self.map.transform_to_geolocation(carla.Location(0.0, 0.0, 0.0))
        
    def stabilize_covariance(self) -> None:
        """Keep P symmetric positive definite after numerical operations."""
        P = 0.5 * (self.P + self.P.T)
        eig_min = float(np.min(np.linalg.eigvalsh(P)))
        if eig_min < 1e-10:
            P += np.eye(STATE_SIZE) * (1e-10 - eig_min)
        self.P = 0.5 * (P + P.T)

    def _discretize_process_noise(self, Fc: np.ndarray, Gc: np.ndarray, Qc_diag: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
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
        return {"p": self.p.copy(), "v": self.v_world.copy(),
            "q": self.q.copy(),"bg": self.bg.copy(),
            "ba": self.ba.copy(),"wheel_bias": float(self.wheel_bias)}

    def inject_error(self, dx: np.ndarray) -> None:
        dx = np.asarray(dx, dtype=float)
        self.p += dx[0:3]
        self.v_world += dx[3:6]
        dtheta = dx[6:9]
        self.q = quat_normalize(quat_mul(self.q, small_angle_quat(dtheta)))
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        self.wheel_bias += float(dx[15])

    @staticmethod
    def carla_imu_to_body(imu_data: carla.IMUMeasurement,) -> Tuple[np.ndarray, np.ndarray]:
        gyro_carla = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z,], dtype=float)
        accel_carla = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z,], dtype=float)
        return np.diag([1.0, -1.0, 1.0]) @ accel_carla, np.diag([-1.0, 1.0, -1.0])@ gyro_carla

    def propagate(self, dt: float) -> None:
        if dt <= 0.0:   return
        R = quat_to_R(self.q)
        a_world = R @ self.a_body + np.array([0.0, 0.0, -hp.G], dtype=float)

        self.p += self.v_world * dt + 0.5 * a_world * dt * dt
        self.v_world += a_world * dt
        self.q = quat_normalize(quat_mul(self.q, small_angle_quat(self.gyro * dt)))

        Fc = np.zeros((STATE_SIZE, STATE_SIZE), dtype=float)
        Fc[0:3, 3:6] = np.eye(3)
        Fc[3:6, 6:9] = -R @ skew(self.a_body)
        Fc[3:6, 12:15] = -R
        Fc[6:9, 6:9] = -skew(self.gyro)
        Fc[6:9, 9:12] = -np.eye(3)

        Gc = np.zeros((STATE_SIZE, 13), dtype=float)
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
            hp.SIGMA_WHEEL_BIAS_RW**2,
        ], dtype=float)
        F, Qd = self._discretize_process_noise(Fc, Gc, Qc_diag, dt)
        self.P = F @ self.P @ F.T + Qd
        self.stabilize_covariance()

    def robust_scale(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
        """Return (NIS, covariance inflation factor) for a measurement."""
        S = H @ self.P @ H.T + R
        try:    d2 = float(innovation.T @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return float("inf"), hp.MAX_R_SCALE
        dof = innovation.size
        threshold = hp.CHI2_999.get(dof, 3.0 * dof + 7.0)
        scale = 1.0 if d2 <= threshold else min(d2 / threshold, hp.MAX_R_SCALE)
        return d2, scale

    def _apply_error_reset(self, dtheta: np.ndarray) -> None:
        """Apply the covariance reset associated with quaternion error injection."""
        G_reset = np.eye(STATE_SIZE, dtype=float)
        G_reset[6:9, 6:9] = np.eye(3) - 0.5 * skew(dtheta)
        self.P = G_reset @ self.P @ G_reset.T

    def update_error_state(self, z: np.ndarray, z_pred: np.ndarray,
        H: np.ndarray, R: np.ndarray, residual_fn=None,sensor_name: str = "unknown") -> bool:
        """EKF measurement update using Joseph-form covariance update."""
        if residual_fn is None: innovation = z - z_pred
        else:   innovation = np.atleast_1d(residual_fn(z, z_pred)) # for compass

        nis, scale = self.robust_scale(innovation, H, R)
        R_eff = R * scale
        PHt = self.P @ H.T
        S = H @ PHt + R_eff
        K = np.linalg.solve(S, PHt.T).T
        dx = K @ innovation

        I_KH = np.eye(STATE_SIZE, dtype=float) - K @ H
        P_new = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (P_new + P_new.T)

        # Inject the error into the nominal state.
        dtheta = dx[6:9].copy()
        self.inject_error(dx)

        # Injection changes the local attitude-error coordinates, so reset
        # the corresponding covariance block.
        self._apply_error_reset(dtheta)
        self.stabilize_covariance()

        self.nis_history.append({
            "sensor": sensor_name,
            "nis": float(nis),
            "dof": int(z.size),
            "scale": float(scale),
            "accepted": bool(np.isfinite(nis)),
        })
        return True

    def h_gnss(self) -> np.ndarray:
        return self.p.copy()

    def H_gnss(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE), dtype=float)
        H[:, 0:3] = np.eye(3)
        return H

    def h_compass(self) -> np.ndarray:
        _, _, yaw = quat_to_euler(self.q)
        return np.array([yaw], dtype=float)

    def H_compass(self) -> np.ndarray:
        H = np.zeros((1, STATE_SIZE), dtype=float)
        R = quat_to_R(self.q)
        x, y = R[0, 0], R[1, 0] # yaw = atan2(y,x)
        den = max(1e-8,x**2 + y**2) 
        dy_dth = np.array([0.0, -R[1, 2], R[1, 1]])
        dx_dth = np.array([0.0, -R[0, 2], R[0, 1]])
        H[0, 6:9] = (x * dy_dth - y * dx_dth) / den
        return H

    def h_wheel(self) -> np.ndarray:
        v_body = quat_to_R(self.q).T @ self.v_world
        return np.array([v_body[0] + self.wheel_bias, v_body[1]], dtype=float)

    def H_wheel(self) -> np.ndarray:
        H = np.zeros((2, STATE_SIZE), dtype=float)
        R_bw = quat_to_R(self.q)
        v_body = R_bw.T @ self.v_world
        H[:, 3:6] = R_bw.T[:2, :]
        H[0:2, 6:9] = skew(v_body)[0:2, :]
        H[0, 15] = 1.0 
        return H

    def h_H_gravity_direction(self) -> tuple[np.ndarray, np.ndarray]:
        H = np.zeros((3, STATE_SIZE), dtype=float)
        v = -(quat_to_R(self.q).T @ np.array([0.0, 0.0, -G], dtype=float))
        norm_v = float(np.linalg.norm(v)) # should be always 9.81
        if norm_v < 1e-12: return (np.array([0.0, 0.0, 1.0], dtype=float), H)
        h_g = v / norm_v
        H[:, 6:9] = ((np.eye(3) - np.outer(h_g, h_g)) / norm_v) @ (skew(v))
        return (h_g , H)

    def h_gnss_vel_world(self) -> np.ndarray:
        return self.v_world.copy()

    def H_gnss_vel_world(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE), dtype=float)
        H[:, 3:6] = np.eye(3)
        return H

    def h_slam_pos(self) -> np.ndarray:
        return self.p.copy()

    def H_slam_pos(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE), dtype=float)
        H[:, 0:3] = np.eye(3)
        return H

    def h_slam_att(self) -> np.ndarray:
        return self.q.copy()

    def H_slam_att(self) -> np.ndarray:
        H = np.zeros((3, STATE_SIZE), dtype=float)
        H[:, 6:9] = np.eye(3)
        return H

    def update_gnss(self, gnss_data: carla.GnssMeasurement, step) -> None:
        if self.last_gnss_frame is not None and gnss_data.frame == self.last_gnss_frame: return
        def _mercator(lat, lon, scale):
            mx = scale * math.radians(lon) * hp.EARTH_RADIUS_EQUA
            my = scale * hp.EARTH_RADIUS_EQUA * math.log(math.tan(math.radians(90.0 + lat) / 2.0))
            return mx, my

        def geolocation_to_location(lat, lon, alt):
            ref = self._geo_ref
            scale = math.cos(math.radians(ref.latitude))
            mx, my = _mercator(lat, lon, scale)
            mx0, my0 = _mercator(ref.latitude, ref.longitude, scale)
            return carla.Location(x=mx - mx0, y=-(my - my0), z=alt)

        loc = geolocation_to_location(self.map, gnss_data.latitude, gnss_data.longitude, gnss_data.altitude)
        self.history_gnss.append((loc.x, -loc.y, loc.z,step))
        self.update_error_state(
            z=np.array([loc.x, -loc.y, loc.z], dtype=float),
            z_pred=self.h_gnss(),
            H=self.H_gnss(),
            R=hp.R_GNSS,
            sensor_name="gnss",
        )
        self.last_gnss_frame = gnss_data.frame

    def update_compass(self, imu_data: carla.IMUMeasurement) -> None:
        yaw_measurement = wrap(math.pi / 2.0 - float(imu_data.compass))
        self.update_error_state(
            z=np.array([yaw_measurement], dtype=float),
            z_pred=self.h_compass(),
            H=self.H_compass(),
            R=hp.R_COMPASS,
            residual_fn=lambda a, b: np.array([wrap(float(a[0] - b[0]))]),
            sensor_name="compass",
        )

    def update_wheel(self, speed_measurement: Optional[float]) -> None:
        if speed_measurement is None:
            return
        z = np.array([float(speed_measurement), 0.0], dtype=float)
        for i in range(2):
            z_pred = self.h_wheel()
            H = self.H_wheel()
            self.update_error_state(
                z=np.array([z[i]], dtype=float),
                z_pred=np.array([z_pred[i]], dtype=float),
                H=H[i:i + 1, :],
                R=np.array([[hp.R_WHEEL[i, i]]], dtype=float),
                sensor_name="wheel_longitudinal" if i == 0 else "wheel_lateral",
            )

    def update_gravity(self, accel_body: np.ndarray) -> None:
        if not self.use_gravity_update: return
        self.accel_history.append(accel_body.copy())
        accel_norm = float(np.linalg.norm(accel_body))
        if len(self.accel_history) > self.ACCEL_HISTORY_LEN:
            self.accel_history.pop(0)
        if len(self.accel_history) < self.ACCEL_HISTORY_LEN or float(np.var(self.accel_history, axis=0).sum()) > self.GRAVITY_VAR_TOL \
            or abs(accel_norm - hp.G) > hp.GRAVITY_NORM_TOL or np.linalg.norm(quat_to_R(self.q).T @ self.v_world) > hp.GRAVITY_UPDATE_MAX_SPEED:
            return

        measured_dir = accel_body.copy()
        measured_norm = np.linalg.norm(measured_dir)
        if measured_norm < 1e-9:    return
        measured_dir /= measured_norm
        dynamic_excess = max(0.0, abs(accel_norm - hp.G))
        inflation = min(1.0 + hp.GRAVITY_DYNAMIC_SCALE * dynamic_excess / hp.G, 8.0)
        z_pred, H = self.h_H_gravity_direction()
        self.update_error_state(z=measured_dir, z_pred=z_pred, H=H, R=hp.R_GRAVITY * inflation,sensor_name="gravity")

    def update_gnss_velocity(self, dt:float = 0.05) -> None:
        if len(self.history_gnss) < hp.MIN_HISTORY:  return
        history = np.asarray(self.history_gnss, dtype=float)
        px, py, pz = history[:, 0], history[:, 1], history[:, 2]
        steps = history[:, 3]
        t = (steps - steps[0]) * dt
        A = np.column_stack((t, np.ones_like(t)))
        beta_x, _, _, _ = np.linalg.lstsq(A, px, rcond=None)
        beta_y, _, _, _ = np.linalg.lstsq(A, py, rcond=None)
        beta_z, _, _, _ = np.linalg.lstsq(A, pz, rcond=None)
        v_world = np.array([beta_x[0], beta_y[0], beta_z[0]], dtype=float)
        self.update_error_state(
            z=v_world, 
            z_pred=self.h_gnss_vel_world(),
            H=self.H_gnss_vel_world(), 
            R=hp.R_GNSS_VEL,
            sensor_name="GNSS_vel"
        )        
        if len(self.history_gnss) == hp.MAX_HISTORY:
            self.history_gnss.pop(0)

    def update_slam(self, lidar_data: carla.LidarMeasurement, step) -> None:
        if step%2 == 1 and lidar_data is not None:
            pred = np.eye(4)
            pred[0:3, 0:3] = quat_to_R(self.q)
            pred[0:3, 3] = self.p
            self.slam.push_frame(lidar_data, pred)
        new_slam = self.slam.get_latest_result()
        if new_slam is None:    return
        new_pos, new_ori = new_slam
        last_pos, last_ori = self.last_slam
        if (last_pos == new_pos).all() and (last_ori == new_ori).all(): return
        self.last_slam = p_slam, q_slam = new_slam
        if p_slam is not None:
            self.update_error_state(
                z=np.asarray(p_slam, dtype=float),
                z_pred=self.h_slam_pos(),
                H=self.H_slam_pos(),
                R=hp.R_SLAM_POS,
                sensor_name="slam_position"
            )
        if q_slam is not None:
            def att_residual(z_meas, z_pred):
                #q = q_pred^(-1) (*) q_meas
                q_pred_inv = np.array([z_pred[0], -z_pred[1], -z_pred[2], -z_pred[3]], dtype=float)
                dq = quat_mul(q_pred_inv, z_meas)
                if dq[0] < 0.0: dq = -dq
                return 2.0 * dq[1:4]
    
            self.update_error_state(
                z=np.asarray(q_slam, dtype=float),
                z_pred=self.h_slam_att(),
                H=self.H_slam_att(),
                R=hp.R_SLAM_ATT,
                residual_fn=att_residual,
                sensor_name="slam_attitude"
            )

    @staticmethod
    def get_simulated_speedometer(vehicle, noise_std: float = 0.5, 
            quantization: float = 0.1, bias: float = 0.8, dropout_prob: float = 0.08,) -> Optional[float]:
        if dropout_prob > 0.0 and np.random.rand() < dropout_prob:
            return None
        vel = vehicle.get_velocity()
        velocity_world = np.array([vel.x, vel.y, vel.z],dtype=float)
        forward = vehicle.get_transform().get_forward_vector()
        forward_world = np.array([forward.x, forward.y, forward.z],dtype=float)
        true_forward_speed = float(np.dot(velocity_world, forward_world))
        measurement = true_forward_speed + bias
        if noise_std > 0.0: measurement += np.random.normal(0.0, noise_std)
        if quantization > 0.0:measurement = (round(measurement / quantization)* quantization)
        return float(measurement)

    def extract_state_from_filter(self) -> Ego3DState:
        roll, pitch, yaw = quat_to_euler(self.q)
        R_bw = quat_to_R(self.q)
        v_body = R_bw.T @ self.v_world
        a_world = quat_to_R(self.q) @ self.a_body + np.array([0.0, 0.0, -hp.G], dtype=float)
        speed = float(np.linalg.norm(v_body))

        state = Ego3DState(
            x=float(self.p[0]), y=-float(self.p[1]), z=float(self.p[2]),
            roll_rad=float(roll), pitch_rad=-float(pitch), yaw_rad=-float(yaw),
            roll=math.degrees(float(roll)), pitch=math.degrees(-float(pitch)), yaw=math.degrees(-float(yaw)),
            roll_rate=float(self.gyro[0]), pitch_rate=-float(self.gyro[1]), yaw_rate=float(self.gyro[2]),
            vx_body=float(v_body[0]), vy_body=float(-v_body[1]), vz_body=float(v_body[2]),
            ax_body=float(self.a_body[0]),ay_body=float(-self.a_body[1]),az_body=-float(self.a_body[2]),
            speed=speed,bg_yaw = self.bg[2], bg_pitch = self.bg[1], bg_roll = self.bg[0],
            long_accel=float(a_world[0]), lat_accel=float(-a_world[1]), vert_accel=-float(a_world[2]),
            vx_world=float(self.v_world[0]), vy_world=float(-self.v_world[1]), vz_world=float(self.v_world[2]),
        )
        self.prev_state = state
        return state

    def compute(
        self, dt: float,
        vehicle: carla.Vehicle,
        imu_data: carla.IMUMeasurement,
        gnss_data: carla.GnssMeasurement,
        lidar_data: carla.LidarMeasurement,
        step
    ) -> Ego3DState:
        accel_body, gyro_body = self.carla_imu_to_body(imu_data) #from LH sys to RH sys
        accel_body = np.clip(accel_body, -10.0, 10.0)
        gyro_body = np.clip(gyro_body, -math.radians(100.0), math.radians(100.0))
        if self.last_gnss_frame == None:
            gyro_body = np.zeros(3, dtype=float)
        self.gyro = gyro_body.copy() - self.bg
        self.a_body = accel_body.copy() - self.ba
        self.propagate(dt=dt)
        if gnss_data is not None:   self.update_gnss(gnss_data,step)
        if imu_data is not None:    self.update_compass(imu_data)
        ss = self.get_simulated_speedometer(vehicle, noise_std=0.05, quantization=0.03, bias=0.2,dropout_prob=0.01)
        self.update_wheel(ss)
        self.update_gravity(accel_body)
        self.update_gnss_velocity()
        self.update_slam(lidar_data, step)
        return self.extract_state_from_filter()
