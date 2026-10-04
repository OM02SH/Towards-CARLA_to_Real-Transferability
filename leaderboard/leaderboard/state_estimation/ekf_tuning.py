"""
Record once, replay offline, tune on many routes.

LIVE (inside your agent loop, once per route):
    rec = SensorRecorder(geo_ref)                                   # geo_ref = map.transform_to_geolocation(carla.Location(0,0,0))
    ...
    wheel = EgoErrorStateEKF3D.get_simulated_speedometer(vehicle, rng=rng)   # simulate OUTSIDE so it can be recorded
    state = ekf.compute(dt, vehicle, imu, gnss, lidar, step, wheel_speed=wheel)
    rec.add(step, dt, imu, gnss, lidar is not None, wheel, vehicle)
    ...
    rec.save("route_042.npz", ekf)                                  # also stores the SLAM outputs (cached for replay)

OFFLINE:
    python ekf_tuning.py report route_042.npz
    python ekf_tuning.py tune --train r1.npz r2.npz r3.npz --val r4.npz r5.npz --trials 200

Replay re-uses the recorded SLAM outputs (ICP is the slow part and barely depends on the EKF), so each trial
only runs the filter. SLAM-internal parameters (ICP, keyframes, masking) need a live/raw-lidar run instead.
"""
import argparse
import math
import sys
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

for _m in ("carla", "open3d"):          # offline replay does not need them
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = mock.MagicMock()

import leaderboard.state_estimation.hyperparams as hp                    # noqa: E402
import leaderboard.state_estimation.ego_EKF_3D as E                         # noqa: E402
from leaderboard.state_estimation.carla_lida_loc import SlamOutput       # noqa: E402


# ============================================================================ recording
class SensorRecorder:
    def __init__(self, geo_ref):
        self.geo = (geo_ref.latitude, geo_ref.longitude, geo_ref.altitude)
        self.rows = []

    def add(self, step, dt, imu, gnss, has_lidar, wheel_speed, vehicle):
        tf, vel = vehicle.get_transform(), vehicle.get_velocity()
        self.rows.append(dict(
            step=step, dt=dt,
            accel=[imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z],
            gyro=[imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z], compass=float(imu.compass),
            gnss=[gnss.latitude, gnss.longitude, gnss.altitude, gnss.frame] if gnss is not None else [np.nan] * 4,
            wheel=np.nan if wheel_speed is None else float(wheel_speed), lidar=bool(has_lidar),
            gt_loc=[tf.location.x, tf.location.y, tf.location.z],
            gt_rot=[tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw],
            gt_vel=[vel.x, vel.y, vel.z]))

    def save(self, path, ekf):
        r = self.rows
        arr = lambda k: np.asarray([x[k] for x in r], dtype=float)
        slam = ekf.slam_log
        np.savez_compressed(
            path, geo=np.asarray(self.geo), steps=arr("step"), dt=arr("dt"), accel=arr("accel"), gyro=arr("gyro"),
            compass=arr("compass"), gnss=arr("gnss"), wheel=arr("wheel"), lidar=arr("lidar").astype(bool),
            gt_loc=arr("gt_loc"), gt_rot=arr("gt_rot"), gt_vel=arr("gt_vel"),
            slam_arrival=np.asarray([a for a, _ in slam], dtype=float),
            slam_frame=np.asarray([o.frame_id for _, o in slam], dtype=float),
            slam_p=np.asarray([o.p for _, o in slam], dtype=float).reshape(-1, 3),
            slam_q=np.asarray([o.q for _, o in slam], dtype=float).reshape(-1, 4),
            slam_shape=np.asarray([np.asarray(o.trans_shape).ravel() for _, o in slam], dtype=float).reshape(-1, 9),
            slam_epoch=np.asarray([o.map_epoch for _, o in slam], dtype=float))


# ============================================================================ replay
class ReplaySlam:
    def __init__(self, d):
        self.arrival = d["slam_arrival"]
        self.items = [SlamOutput(frame_id=int(f), p=p, q=q, trans_shape=s.reshape(3, 3), map_epoch=int(e))
                      for f, p, q, s, e in zip(d["slam_frame"], d["slam_p"], d["slam_q"], d["slam_shape"], d["slam_epoch"])]
        self.i, self.current_step = 0, -1

    def push_frame(self, *a, **k): pass

    def get_latest_result(self):
        out = None
        while self.i < len(self.arrival) and self.arrival[self.i] <= self.current_step:
            out, self.i = self.items[self.i], self.i + 1
        return out

    def stop(self): pass


def _gt_ekf_frame(d, i):
    x, y, z = d["gt_loc"][i]
    roll, pitch, yaw = np.radians(d["gt_rot"][i])
    R = E.quat_to_R(E.euler_to_quat(roll, -pitch, -yaw))
    vx, vy, vz = d["gt_vel"][i]
    return np.array([x, -y, z]), np.array([vx, -vy, vz]), R


def run_replay(d, collect=True):
    n = len(d["dt"])
    tf0 = NS(location=NS(x=d["gt_loc"][0][0], y=d["gt_loc"][0][1], z=d["gt_loc"][0][2]),
             rotation=NS(roll=d["gt_rot"][0][0], pitch=d["gt_rot"][0][1], yaw=d["gt_rot"][0][2]))
    geo = NS(latitude=d["geo"][0], longitude=d["geo"][1], altitude=d["geo"][2])
    slam = ReplaySlam(d)
    ekf = E.EgoErrorStateEKF3D(NS(get_transform=lambda: tf0), None, slam_worker=slam, geo_ref=geo, seed=0)
    out = {k: [] for k in ("pos_err", "vel_err", "att_err", "nees", "stationary_gt", "yaw_err_deg", "tilt_err_deg")}
    for i in range(n):
        a, g = d["accel"][i], d["gyro"][i]
        imu = NS(accelerometer=NS(x=a[0], y=a[1], z=a[2]), gyroscope=NS(x=g[0], y=g[1], z=g[2]), compass=d["compass"][i])
        gn = d["gnss"][i]
        gnss = None if np.isnan(gn[0]) else NS(latitude=gn[0], longitude=gn[1], altitude=gn[2], frame=int(gn[3]))
        w = None if np.isnan(d["wheel"][i]) else float(d["wheel"][i])
        step = int(d["steps"][i])
        slam.current_step = step
        ekf.compute(float(d["dt"][i]), None, imu, gnss, object() if d["lidar"][i] else None, step, wheel_speed=w)
        p_gt, v_gt, R_gt = _gt_ekf_frame(d, i)
        R = E.quat_to_R(ekf.q)
        dth = Rotation.from_matrix(R.T @ R_gt).as_rotvec()
        dx = np.concatenate([p_gt - ekf.p, v_gt - ekf.v_world, dth])
        try:
            nees = float(dx @ np.linalg.solve(ekf.P[0:9, 0:9], dx))
        except np.linalg.LinAlgError:
            nees = np.nan
        out["pos_err"].append(p_gt - ekf.p)
        out["vel_err"].append(v_gt - ekf.v_world)
        out["nees"].append(nees)
        out["att_err"].append(dth)
        out["stationary_gt"].append(np.linalg.norm(v_gt) < 0.1)
        rpy_est, rpy_gt = Rotation.from_matrix(R).as_euler("xyz"), Rotation.from_matrix(R_gt).as_euler("xyz")
        out["yaw_err_deg"].append(math.degrees(E.wrap(rpy_est[2] - rpy_gt[2])))
        out["tilt_err_deg"].append(math.degrees(np.linalg.norm(rpy_est[:2] - rpy_gt[:2])))
    res = {k: np.asarray(v) for k, v in out.items()}
    res["nis"] = ekf.nis_summary()
    return res


def metrics(r):
    pe, ve, st = r["pos_err"], r["vel_err"], r["stationary_gt"]
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")
    return dict(pos2d_rmse=rms(np.linalg.norm(pe[:, :2], axis=1)), pos3d_rmse=rms(np.linalg.norm(pe, axis=1)),
                z_bias=float(-pe[:, 2].mean()), final_pos2d=float(np.linalg.norm(pe[-1, :2])),
                yaw_rmse_deg=rms(r["yaw_err_deg"]), tilt_rmse_deg=rms(r["tilt_err_deg"]),
                vel_rmse=rms(np.linalg.norm(ve, axis=1)), vel_rmse_stationary=rms(np.linalg.norm(ve[st], axis=1)),
                nees_mean=float(np.nanmean(r["nees"])))        # ideal ~9 (dof of p,v,theta)


def objective_value(m, w=(1.0, 0.5, 0.05, 0.05)):
    v = w[0] * m["pos2d_rmse"] + w[1] * m["vel_rmse"] + w[2] * m["yaw_rmse_deg"] + w[3] * m["tilt_rmse_deg"]
    return v if np.isfinite(v) else 1e6


# ============================================================================ tuning
SEARCH = {   # (low, high) log-uniform
    "SIGMA_GYRO": (math.radians(0.05), math.radians(2.0)),
    "SIGMA_ACCEL": (0.02, 1.0),
    "SIGMA_GYRO_BIAS_RW": (1e-6, 1e-2),
    "SIGMA_ACCEL_BIAS_RW": (1e-4, 0.1),
    "SIGMA_WHEEL_SCALE_RW": (1e-5, 1e-2),
    "SIGMA_GNSS_XY": (0.1, 3.0),
    "SIGMA_GNSS_Z": (0.1, 3.0),
    "SIGMA_COMPASS": (math.radians(0.5), math.radians(10.0)),
    "SIGMA_WHEEL_LONG": (0.02, 0.5),
    "SIGMA_WHEEL_LAT": (0.02, 0.5),
    "SIGMA_SLAM_REL_POS": (0.002, 0.1),
    "SIGMA_SLAM_REL_ROT": (math.radians(0.02), math.radians(1.0)),
    "SLAM_REL_VEL_FLOOR": (0.01, 0.3),
    "SIGMA_SLAM_TILT": (math.radians(0.1), math.radians(3.0)),
    "SIGMA_SLAM_Z": (0.03, 1.0),
    "SIGMA_ZUPT_VEL": (0.005, 0.1),
}


def apply(params):
    for k, v in params.items():
        setattr(hp, k, v)
    hp.rebuild()


def tune(train, val, trials, fix=(), seed=0):
    import optuna
    names = [k for k in SEARCH if k not in set(fix)]
    default = {k: getattr(hp, k) for k in SEARCH}
    train_d = [dict(np.load(p)) for p in train]

    def obj(trial):
        apply({k: trial.suggest_float(k, *SEARCH[k], log=True) for k in names})
        try:
            return float(np.mean([objective_value(metrics(run_replay(d))) for d in train_d]))
        except Exception as e:                       # numerically broken parameter set
            print("trial failed:", e)
            return 1e6

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: min(max(default[k], SEARCH[k][0]), SEARCH[k][1]) for k in names})   # current values first
    study.optimize(obj, n_trials=trials)
    print("\nBest train objective:", study.best_value)
    print("Best parameters (paste into hyperparams.py):")
    for k, v in study.best_params.items():
        print(f"{k} = {v:.6g}")
    if val:
        for label, p in (("DEFAULT", default), ("TUNED", study.best_params)):
            apply(p)
            vs = [metrics(run_replay(dict(np.load(f)))) for f in val]
            print(f"\nValidation with {label} params:")
            for f, m in zip(val, vs):
                print(" ", f, {k: round(x, 3) for k, x in m.items()})
    apply(default)


def report(path):
    r = run_replay(dict(np.load(path)))
    print("Metrics:", {k: round(v, 3) for k, v in metrics(r).items()})
    print("\nPer-sensor consistency (mean NIS/dof should be ~1; <<1 = R too conservative, >>1 = too optimistic):")
    for k, v in sorted(r["nis"].items()):
        print(f"  {k:20s} n={v['n']:6d} accept={v['accept_rate']:.2f} NIS/dof={v['mean_nis_per_dof']:.2f}")
    print("\nNEES(p,v,theta) mean should be ~9. Mean innovation per sensor can be inspected in ekf.nis_history[*]['innov'].")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("report"); a.add_argument("recording")
    b = sub.add_parser("tune")
    b.add_argument("--train", nargs="+", required=True)
    b.add_argument("--val", nargs="*", default=[])
    b.add_argument("--trials", type=int, default=150)
    b.add_argument("--fix", nargs="*", default=[], help="parameter names to keep fixed (e.g. ones equal to the CARLA sensor noise)")
    args = ap.parse_args()
    if args.cmd == "report":
        report(args.recording)
    else:
        tune(args.train, args.val, args.trials, args.fix)