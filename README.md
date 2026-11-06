# Sensor-based State estimation and Perception in Carla

Perception and state-estimation stack from the bachelor's thesis
*Towards CARLA-to-Real Transferability* (Omar Shaat, TUM, 2026). It runs inside
the CARLA Leaderboard / Bench2Drive harness and **only observes** the driving
agent: it reconstructs ego state, surrounding traffic, collision risk, and
traffic-light state from sensors, and compares them with live CARLA ground truth.

Tested with CARLA 0.9.15.

In the thesis experiments (Chapter 6), the stack was run on 37 Bench2Drive routes
while the vehicle was driven by an independent agent (TransFuser v6, LEAD),
executed through PCLA. The stack has no influence on the agent's actions or on the
route outcomes.

## Repository layout

```
/.
├── Dockerfile, requirements.txt, run_full.sh   # container + full-run script
├── leaderboard/                                # Bench2Drive leaderboard fork
│   ├── run_leaderboard.sh, results.json
│   └── leaderboard/state_estimation/           # <- this stack (see below)
├── scenario_runner/                            # CARLA scenario runner
├── pcdet/, tools/                              # OpenPCDet (detector training / inference)
├── prepare_dataset/
│   ├── prepare_carlaocc.py                     # CarlaOcc -> OpenPCDet infos (LiDAR)
│   ├── prepare_carlaocc_bevfusion.py           # adds camera calibration for BEVFusion
│   ├── add_parked_car_boxes.py                 # recovers parked-vehicle boxes (Thesis Sec. 5.1.5)
│   └── carlaocc/                               # CarlaOcc dataset in OpenPCDet format
├── aggregate_routes.py, agg/                   # pools per-route results (CIs, tables)
└── out/                                        # run outputs
    └── inspect_ghosts.py                       # ghost-detection analysis (Thesis Sec. 6.3.6)
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
| `TwoDimTTC.py` | 2D time-to-collision (third-party algorithm, see Acknowledgements) | 4.3 |
| `traffic_light_worker.py`, `best.pt` | YOLOv8 traffic-light worker and weights | 4.4, 5.3 |
| `state_from_sensors.py` | Assembles the per-frame state (ego, tracks, TTC, light, map context) | 4.5 |
| `sensor_dataclasses.py` | Shared data containers for sensor and state objects | supporting |
| `hyperparams.py` | Central hyperparameters (thresholds, noise, lifecycle) | supporting |
| `metrics_sensors.py` | Deployment sensor suite spawned for closed-loop runs (not identical to CarlaOcc's recording rig) | 5.2 |
| `bbs_metrics_export.py` | Report / JSON export of the metrics | 6.1.4 |
| `inspect_ghosts.py` | Spatial aggregation of ghost detections (see below) | 6.3.6 |
| `atomic_criteria_sensors.py`, `shadow_criteria.py` | Bench2Drive criteria recomputed from the estimated state, run next to the originals | 6.5.2 |
| `real_test_ego.py`, `real_test_BBs.py` | Live ground-truth extraction and online ego/detection evaluators | 6.1 |

## Run with Docker

Start the CARLA 0.9.15 server on the host first, listening on RPC port 2000
(the default). Then:

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

## Outputs

- Per-route metrics (JSON) and the Leaderboard `results.json`, written to `out/`.

### Aggregating routes

`aggregate_routes.py` pools the per-route results (route-level bootstrap
confidence intervals) into the tables of thesis Chapter 6:

```bash
python aggregate_routes.py --dir out --out agg/
```

### Ghost-detection analysis (Thesis Sec. 6.3.6)

`out/inspect_ghosts.py` takes a CSV of ghost detections (columns `x_world`,
`y_world`, `length`, `width`, `speed`, `score`, `frame`), groups them into 2 m
world-position cells, and for the busiest cells queries the running CARLA
server for environment objects and actors within a given radius. This tests
whether ghost detections correspond to real objects (for example parked
vehicles) that are missing from the live ground truth.

```bash
# with a running CARLA server (default localhost:2000)
python out/inspect_ghosts.py <ghosts.csv> --top 10 --radius 4.0

# only print the per-cell statistics, without querying the simulator
python out/inspect_ghosts.py <ghosts.csv> --no-carla
```

## Weights

`bevfusion.pth`, `best.pt`, and the Swin-T checkpoint are large binaries.

## Acknowledgements and third-party code

- `TwoDimTTC.py` implements the two-dimensional time-to-collision algorithm of
  Jiao et al. (2022); it is not an original contribution of this thesis.
- `pcdet/` and `tools/` are OpenPCDet, used for detector training and inference.
- `leaderboard/` and `scenario_runner/` are the CARLA Leaderboard and
  scenario runner as used by Bench2Drive.
- The CarlaOcc dataset, TransFusion-L, BEVFusion, and YOLOv8 are cited in the
  thesis (Chapter 3).
- The agent used in the experiments (TransFuser v6, LEAD) and PCLA are
  third-party and are cited in the thesis (Section 6.2).

## Scope

All results are obtained in CARLA against CARLA ground truth; no claim is made
about real-world performance. Map-relative quantities (lane, speed limit,
junction distance) still come from the CARLA map.
