from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import carla
from leaderboard.state_estimation.sensor_dataclasses import Ego3DState
import leaderboard.state_estimation.hyperparams as hp


def wrap_angle_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _vec(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)

def _unit(name: str) -> str:
    if name in hp.ANGLE_FIELDS:
        return "deg"
    if name in hp.POSITION_FIELDS:
        return "m"
    if name in hp.RATE_FIELDS:
        return "rad/s"
    if name in hp.VELOCITY_FIELDS:
        return "m/s"
    return "m/s^2"

def _stats(err) -> dict:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"n": 0, **{k: float("nan") for k in hp.STAT_KEYS}}
    a = np.abs(err)
    return {"n": int(err.size),
            "bias": float(err.mean()), "std": float(err.std()),
            "mae": float(a.mean()), "median_abs": float(np.median(a)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "p95_abs": float(np.percentile(a, 95)), "max_abs": float(a.max())}

def ego_state_from_vehicle(vehicle: carla.Vehicle) -> Ego3DState:
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

def evaluate_frames(gt_frames: Dict[str, Ego3DState], pred_frames: Dict[str, Ego3DState],
                    rpe_delta_frames: int = hp.DEFAULT_RPE_DELTA_FRAMES,
                    moving_threshold: float = hp.MOVING_SPEED_THRESHOLD,
                    stationary_threshold: float = hp.STATIONARY_SPEED_THRESHOLD) -> dict:
    ids = [f for f in gt_frames if f in pred_frames]

    def col(frames, name):
        return np.array([getattr(frames[f], name) for f in ids], dtype=float)

    def mat(frames, names):
        return np.stack([col(frames, n) for n in names], axis=1)

    def diff_norm(names):
        return np.linalg.norm(mat(pred_frames, names) - mat(gt_frames, names), axis=1)

    fields = {}
    for name in hp.LINEAR_FIELDS + hp.ANGLE_FIELDS:
        err = col(pred_frames, name) - col(gt_frames, name)
        if name in hp.ANGLE_FIELDS:
            err = np.degrees(wrap_angle_rad(err))
        fields[name] = {"unit": _unit(name), **_stats(err)}

    agg: Dict[str, dict] = {}

    gp, ep = mat(gt_frames, hp.POSITION_FIELDS), mat(pred_frames, hp.POSITION_FIELDS)
    gyaw, eyaw = col(gt_frames, "yaw_rad"), col(pred_frames, "yaw_rad")
    dp = ep - gp
    agg["position_2d"] = _stats(np.linalg.norm(dp[:, :2], axis=1))
    agg["position_3d"] = _stats(np.linalg.norm(dp, axis=1))
    c, s = np.cos(gyaw), np.sin(gyaw)                                # error expressed in the GT heading frame
    agg["position_longitudinal"] = _stats(c * dp[:, 0] + s * dp[:, 1])
    agg["position_lateral"] = _stats(-s * dp[:, 0] + c * dp[:, 1])   # positive = right (CARLA)

    ang = np.stack([wrap_angle_rad(col(pred_frames, n) - col(gt_frames, n)) for n in hp.ANGLE_FIELDS], axis=1)
    agg["attitude_3d"] = _stats(np.degrees(np.linalg.norm(ang, axis=1)))
    agg["tilt"] = _stats(np.degrees(np.linalg.norm(ang[:, :2], axis=1)))

    agg["velocity_world_vec"] = _stats(diff_norm(("vx_world", "vy_world", "vz_world")))
    agg["velocity_body_vec"] = _stats(diff_norm(("vx_body", "vy_body", "vz_body")))
    agg["angular_rate_vec"] = _stats(diff_norm(hp.RATE_FIELDS))
    agg["accel_body_vec"] = _stats(diff_norm(("ax_body", "ay_body", "az_body")))

    gs, es = col(gt_frames, "speed"), col(pred_frames, "speed")
    moving = gs > moving_threshold
    agg["speed_relative_moving"] = _stats(100.0 * (es[moving] - gs[moving]) / gs[moving])
    agg["speed_stationary"] = _stats((es - gs)[gs < stationary_threshold])

    gvx, gvy = col(gt_frames, "vx_world"), col(gt_frames, "vy_world")
    evx, evy = col(pred_frames, "vx_world"), col(pred_frames, "vy_world")
    both_moving = (gs > moving_threshold) & (es > moving_threshold)
    course = wrap_angle_rad(np.arctan2(evy, evx) - np.arctan2(gvy, gvx))
    agg["course_angle"] = _stats(np.degrees(course[both_moving]))

    ok = (np.all(np.isfinite(gp), axis=1) & np.all(np.isfinite(ep), axis=1)
          & np.isfinite(gyaw) & np.isfinite(eyaw))
    gp_, ep_, gy_, ey_ = gp[ok], ep[ok], gyaw[ok], eyaw[ok]
    nan = float("nan")

    trajectory = {"path_length_m": nan, "final_pos_err_2d_m": nan, "drift_percent": nan, "final_yaw_err_deg": nan}
    if len(gp_) >= 2:
        path = float(np.linalg.norm(np.diff(gp_[:, :2], axis=0), axis=1).sum())
        final = float(np.linalg.norm(ep_[-1, :2] - gp_[-1, :2]))
        trajectory = {"path_length_m": path, "final_pos_err_2d_m": final,
                      "drift_percent": 100.0 * final / path if path > 1e-6 else nan,
                      "final_yaw_err_deg": float(math.degrees(wrap_angle_rad(ey_[-1] - gy_[-1])))}

    k = rpe_delta_frames
    if k > 0 and len(gp_) > k:
        dg, de = gp_[k:] - gp_[:-k], ep_[k:] - ep_[:-k]
        trans_err = np.linalg.norm(de - dg, axis=1)
        seg_len = np.linalg.norm(dg, axis=1)
        long_enough = seg_len > hp.MIN_RPE_SEGMENT
        yaw_rel_err = wrap_angle_rad(wrap_angle_rad(ey_[k:] - ey_[:-k]) - wrap_angle_rad(gy_[k:] - gy_[:-k]))
        agg["rpe_translation"] = _stats(trans_err)
        agg["rpe_translation_percent"] = _stats(100.0 * trans_err[long_enough] / seg_len[long_enough])
        agg["rpe_yaw"] = _stats(np.degrees(yaw_rel_err))
    else:
        for key in ("rpe_translation", "rpe_translation_percent", "rpe_yaw"):
            agg[key] = _stats([])

    return {"num_frames": len(ids), "fields": fields, "aggregate": agg, "trajectory": trajectory,
            "config": {"rpe_delta_frames": rpe_delta_frames, "moving_threshold": moving_threshold,
                       "stationary_threshold": stationary_threshold}}

def _row(label: str, unit: str, m: dict) -> str:
    return (f"{label:<24}{unit:>6}{m['n']:>7}{m['bias']:>10.3f}{m['std']:>10.3f}{m['mae']:>10.3f}"
            f"{m['median_abs']:>10.3f}{m['rmse']:>10.3f}{m['p95_abs']:>10.3f}{m['max_abs']:>10.3f}")

def print_results(metrics: dict) -> None:
    """This function was written by an LLM"""
    cfg = metrics["config"]
    header = (f"{'':<24}{'unit':>6}{'n':>7}{'bias':>10}{'std':>10}{'MAE':>10}"
              f"{'median':>10}{'RMSE':>10}{'p95':>10}{'max':>10}")
    width = len(header)

    print()
    print("=" * width)
    print(f"EGO STATE EVALUATION ({metrics['num_frames']} frames, error = estimate - GT)")
    print("=" * width)

    print("\nPER-FIELD ERRORS")
    print(header)
    for name, m in metrics["fields"].items():
        print(_row(name, m["unit"], m))

    print(f"\nAGGREGATE ERRORS  (RPE window = {cfg['rpe_delta_frames']} frames, "
          f"moving > {cfg['moving_threshold']} m/s, stationary < {cfg['stationary_threshold']} m/s)")
    print(header)
    for key, label, unit in hp.AGGREGATE_ROWS:
        print(_row(label, unit, metrics["aggregate"][key]))

    t = metrics["trajectory"]
    print("\nTRAJECTORY")
    print(f"  GT path length     : {t['path_length_m']:.2f} m")
    print(f"  Final 2D pos error : {t['final_pos_err_2d_m']:.3f} m")
    print(f"  Drift              : {t['drift_percent']:.3f} % of distance travelled")
    print(f"  Final yaw error    : {t['final_yaw_err_deg']:.3f} deg")
    print("=" * width)

def save_metrics(metrics: dict, path: Path) -> None:
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(sanitize(metrics), indent=2))

class EgoStateOnlineEvaluator:
    def __init__(self, world: carla.World, ego_actor_id: int,
                 rpe_delta_frames: int = hp.DEFAULT_RPE_DELTA_FRAMES,
                 moving_threshold: float = hp.MOVING_SPEED_THRESHOLD,
                 stationary_threshold: float = hp.STATIONARY_SPEED_THRESHOLD):
        self.vehicle = world.get_actor(ego_actor_id)
        self.rpe_delta_frames = rpe_delta_frames
        self.moving_threshold = moving_threshold
        self.stationary_threshold = stationary_threshold
        self._gt_frames: Dict[str, Ego3DState] = {}
        self._pred_frames: Dict[str, Ego3DState] = {}
        self.dt = 0.05
        self._alpha = 0.05 / (hp.ACCEL_OUT_LPF_TAU + 0.05) if hp.ACCEL_OUT_LPF_TAU > 0 else 1.0
        self._prev_v_w = None
        self._a_lpf = None

    def _with_derived_accel(self, gt):
        """Replace get_acceleration() by d(GT velocity)/dt, filtered like the estimator output."""
        v_w = np.array([gt.vx_world, gt.vy_world, gt.vz_world], dtype=float)
        a_w = np.zeros(3) if self._prev_v_w is None else (v_w - self._prev_v_w) / self.dt
        self._prev_v_w = v_w
        self._a_lpf = a_w if self._a_lpf is None else (1 - self._alpha) * self._a_lpf + self._alpha * a_w
        a_w = self._a_lpf

        tf = carla.Transform(carla.Location(),
                             carla.Rotation(pitch=gt.pitch, yaw=gt.yaw, roll=gt.roll)) 
        R_wb = np.asarray(tf.get_inverse_matrix(), dtype=float)[:3, :3]
        a_b = R_wb @ a_w
        c, s = math.cos(gt.yaw_rad), math.sin(gt.yaw_rad)
        gt.ax_body, gt.ay_body, gt.az_body = a_b
        gt.long_accel = c * a_w[0] + s * a_w[1]
        gt.lat_accel = -s * a_w[0] + c * a_w[1]
        gt.vert_accel = a_w[2]
        return gt

    def add_frame(self, frame_id: str, ego_state: Ego3DState):
        self._gt_frames[frame_id] = self._with_derived_accel(ego_state_from_vehicle(self.vehicle))
        self._pred_frames[frame_id] = ego_state

    def reset(self) -> None:
        self._gt_frames.clear()
        self._pred_frames.clear()

    @property
    def num_frames(self) -> int:
        return len(set(self._gt_frames) & set(self._pred_frames))

    def evaluate(self) -> dict:
        return evaluate_frames(self._gt_frames, self._pred_frames, self.rpe_delta_frames,
                               self.moving_threshold, self.stationary_threshold)

    def print_report(self) -> None:
        print_results(self.evaluate())

    def save_metrics_json(self, filename: "str | Path") -> None:
        print("filename: ", filename)
        save_metrics(self.evaluate(), Path(filename))