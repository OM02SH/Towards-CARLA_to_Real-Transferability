# Sensor-based state estimation for Bench2Drive (`state_estimation/`)

Perception and state-estimation stack from the bachelor's thesis
*Towards CARLA-to-Real Transferability* (Omar Shaat, TUM, 2026). It runs inside
the CARLA Leaderboard / Bench2Drive harness and **only observes** the driving
agent: it reconstructs ego state, surrounding traffic, collision risk, and
traffic-light state from sensors, and compares them with live CARLA ground truth.

Tested with CARLA 0.9.15.

## Repository layout

```
final/
├── Dockerfile, requirements.txt, run_full.sh   # container + full-run script
├── leaderboard/                                # Bench2Drive leaderboard fork
│   ├── run_leaderboard.sh, results.json
│   └── leaderboard/state_estimation/           # <- this stack (see below)
├── scenario_runner/                            # CARLA scenario runner
├── pcdet/, tools/                              # OpenPCDet (detector training / inference)
├── prepare_carlaocc.py                         # CarlaOcc -> OpenPCDet infos (LiDAR)
├── prepare_carlaocc_bevfusion.py               # adds camera calibration for BEVFusion
├── add_parked_car_boxes.py                     # recovers parked-vehicle boxes (Thesis Sec. 5.1.5)
├── aggregate_routes.py, agg/                   # pools per-route results (CIs, tables)
└── out/, carla_out/                            # run outputs   TODO: confirm
```

## `state_estimation/` files

| File | Purpose | Thesis |
|---|---|---|
| `ego_EKF_3D.py` | Error-state EKF: IMU propagation, gated GNSS / compass / wheel-speed / gravity / ZUPT / SLAM updates | 4.1 |
| `carla_lida_loc.py` | LiDAR pose-graph SLAM (submap ICP, loop closure), asynchronous | 4.1.10 |
| `bbs_KF.py`, `UKF_bb.py` | UKF multi-object tracker (association, tentative/confirmed tracks) | 4.2 |
| `bevfusion_carlaocc.yaml`, `bevfusion.pth` | BEVFusion config and fine-tuned weights | 5.1 |
| `swin_tiny_patch4_window7_224.pth` | ImageNet Swin-T backbone weights | 5.1.3 |
| `carlaocc_dataset.yaml` | CarlaOcc dataset config for OpenPCDet | 5.1.1 |
| `TwoDimTTC.py` | 2D time-to-collision | 4.3 |
| `traffic_light_worker.py`, `best.pt` | YOLOv8 traffic-light worker and weights | 4.4, 5.3 |
| `state_from_sensors.py` | Assembles the per-frame state (ego, tracks, TTC, light, map context) | 4.5 |
| `sensor_dataclasses.py` | Shared data containers for sensor and state objects | n/a |
| `hyperparams.py` | Central hyperparameters (thresholds, noise, lifecycle) | n/a |
| `metrics_sensors.py` | Sensor suite that matches the Carlaocc  | 5.2 |
| `bbs_metrics_export.py` | Report / JSON export of the metrics | 6.1.4 |
| `inspect_ghosts.py` | Spatial aggregation of ghost detections | 6.3.6 |
| `atomic_criteria_sensors.py`, `shadow_criteria.py` | Bench2Drive criteria recomputed from the estimated state, run next to the originals | 6.5.2 |
| `real_test_ego.py`, `real_test_BBs.py` | Live ground-truth extraction and online ego/detection evaluators  | 6.1 |

## Run with Docker

```bash
docker build -t sensors-bench2drive .

docker run -d --name test_run --gpus all --net=host \
  -v "$PWD":/app \
  -v "$PWD/CARLA_PythonAPI":/opt/carla/PythonAPI:ro \
  sensors-bench2drive
```

- `--gpus all` is needed for BEVFusion and YOLOv8.
- `--net=host` lets the container reach the CARLA server on the host.
- `$PWD` is mounted at `/app`, so code changes take effect without rebuilding.
- `CARLA_PythonAPI` is mounted read-only; it must match the CARLA server version.
- Follow progress with `docker logs -f test_run`; stop with `docker stop test_run`.

You must start carla and make it run on host 200 first.

## Outputs

- Per-route metrics (JSON) and the Leaderboard `results.json`  
- `aggregate_routes.py` pools routes (route-level bootstrap CIs) into the tables of thesis Chapter 6.

## Weights

`bevfusion.pth`, `best.pt`, and the Swin-T checkpoint are large binaries.

## Scope

All results are obtained in CARLA against CARLA ground truth; no claim is made
about real-world performance. Map-relative quantities (lane, speed limit,
junction distance) still come from the CARLA map.
