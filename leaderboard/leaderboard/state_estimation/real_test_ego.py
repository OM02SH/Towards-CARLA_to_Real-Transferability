from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np
import carla
from leaderboard.leaderboard.state_estimation.sensor_dataclasses import Ego3DState
from hyperparams import LINEAR_FIELDS, ANGLE_FIELDS

def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def _vec(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=float)

def ego_state_from_vehicle(vehicle: carla.Vehicle) -> Ego3DState:
    tf = vehicle.get_transform()
    R_wb = np.asarray(tf.get_inverse_matrix(), dtype=float)[:3, :3]   

    v_w = _vec(vehicle.get_velocity())
    a_w = _vec(vehicle.get_acceleration())
    w_b = R_wb @ np.radians(_vec(vehicle.get_angular_velocity()))
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

def evaluate_frames(gt_frames: Dict[str, Ego3DState], pred_frames: Dict[str, Ego3DState]) -> dict:
    frames = [f for f in gt_frames if f in pred_frames]
    result = {}
    for name in LINEAR_FIELDS + ANGLE_FIELDS:
        errs = []
        for f in frames:
            e = getattr(pred_frames[f], name) - getattr(gt_frames[f], name)
            if name in ANGLE_FIELDS:
                e = math.degrees(wrap_angle_rad(e))
            if math.isfinite(e):
                errs.append(e)
        a = np.array(errs)
        result[name] = {
            "unit": "deg" if name in ANGLE_FIELDS else "",
            "n": len(errs),
            "bias": float(a.mean()) if len(a) else float("nan"),
            "mae": float(np.abs(a).mean()) if len(a) else float("nan"),
            "rmse": float(np.sqrt((a ** 2).mean())) if len(a) else float("nan"),
            "max": float(np.abs(a).max()) if len(a) else float("nan"),
        }
    pos = np.array([[getattr(pred_frames[f], k) - getattr(gt_frames[f], k) for k in "xyz"] for f in frames])
    return {"num_frames": len(frames), "fields": result,
            "pos_err_2d_mean": float(np.nanmean(np.linalg.norm(pos[:, :2], axis=1))) if len(pos) else float("nan"),
            "pos_err_3d_mean": float(np.nanmean(np.linalg.norm(pos, axis=1))) if len(pos) else float("nan")}

def print_results(metrics: dict) -> None:
    print()
    print("=" * 78)
    print(f"EGO STATE EVALUATION ({metrics['num_frames']} frames, error = estimate - GT)")
    print("=" * 78)
    print(f"{'field':<12}{'unit':>5}{'bias':>10}{'MAE':>10}{'RMSE':>10}{'max|e|':>10}")
    for name, m in metrics["fields"].items():
        print(f"{name:<12}{m['unit']:>5}{m['bias']:>10.3f}{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['max']:>10.3f}")
    print(f"\nMean position error  2D: {metrics['pos_err_2d_mean']:.3f} m   3D: {metrics['pos_err_3d_mean']:.3f} m")
    print("=" * 78)

def save_metrics(metrics: dict, path: Path) -> None:
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(sanitize(metrics), indent=2))

class EgoStateOnlineEvaluator:
    def __init__(self, world: carla.World, ego_actor_id: int):
        self.vehicle = world.get_actor(ego_actor_id)
        self._gt_frames: Dict[str, Ego3DState] = {}
        self._pred_frames: Dict[str, Ego3DState] = {}

    def add_frame(self, frame_id: str, ego_state: Ego3DState):
        self._gt_frames[frame_id] = ego_state_from_vehicle(self.vehicle)
        self._pred_frames[frame_id] = ego_state

    def reset(self) -> None:
        self._gt_frames.clear()
        self._pred_frames.clear()

    @property
    def num_frames(self) -> int:
        return len(set(self._gt_frames) & set(self._pred_frames))

    def evaluate(self) -> dict:
        return evaluate_frames(self._gt_frames, self._pred_frames)

    def print_report(self) -> None:
        print_results(self.evaluate())

    def save_metrics_json(self, filename: "str | Path") -> None:
        save_metrics(self.evaluate(), Path(filename))
