#!/usr/bin/env python3
import argparse
import pickle
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import yaml
from plyfile import PlyData

# Optional fast loader for YAML
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

CLASS_NAMES = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian',
    'rider', 'train', 'building', 'pole', 'traffic_sign', 'traffic_light',
    'wall', 'fence', 'guard_rail', 'bridge', 'other'
]

SEMANTIC_ID_TO_NAME = {
    3:  'building', 4:  'wall', 5:  'fence', 6:  'pole',
    7:  'traffic_light', 8:  'traffic_sign', 12: 'pedestrian', 13: 'rider',
    14: 'car', 15: 'truck', 16: 'bus', 17: 'train',
    18: 'motorcycle', 19: 'bicycle', 22: 'other', 26: 'bridge',
    28: 'guard_rail',
}

STATIC_OCC_SEMANTIC_IDS = {
    sid for sid, name in SEMANTIC_ID_TO_NAME.items()
    if name in {'building', 'pole', 'traffic_sign', 'traffic_light',
                'wall', 'fence', 'guard_rail', 'bridge', 'other'}
}


def load_matrix_file(path):
    poses = {}
    if not path.exists():
        return poses
    with path.open() as f:
        for line in f:
            p = line.split()
            if len(p) != 17:
                continue
            try:
                poses[int(p[0])] = np.asarray(p[1:], dtype=np.float64).reshape(4, 4)
            except ValueError:
                continue
    return poses


def load_ply(path):
    ply = PlyData.read(str(path))
    v = ply['vertex']
    names = set(v.data.dtype.names or [])
    xyz = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    intensity_name = 'I' if 'I' in names else 'intensity' if 'intensity' in names else None
    if intensity_name:
        intensity = np.asarray(v[intensity_name], dtype=np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((len(xyz), 1), dtype=np.float32)
    return np.hstack((xyz, intensity))


def to_array(x):
    if isinstance(x, dict):
        if all(k in x for k in ('x', 'y', 'z')):
            return np.array([x['x'], x['y'], x['z']], dtype=np.float64)
        if 'matrix' in x:
            return np.asarray(x['matrix'], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def matrix_from_actor_transform(x):
    if isinstance(x, dict):
        if 'matrix' in x:
            return np.asarray(x['matrix'], dtype=np.float64).reshape(4, 4)
        if 'location' in x:
            loc = x['location']
            rot = x.get('rotation', {})
            tx = float(loc.get('x', 0.0)); ty = float(loc.get('y', 0.0)); tz = float(loc.get('z', 0.0))
            roll = np.deg2rad(float(rot.get('roll', 0.0)))
            pitch = np.deg2rad(float(rot.get('pitch', 0.0)))
            yaw = np.deg2rad(float(rot.get('yaw', 0.0)))
            cr, sr = np.cos(roll), np.sin(roll); cp, sp = np.cos(pitch), np.sin(pitch); cy, sy = np.cos(yaw), np.sin(yaw)
            R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                        [-sp, cp*sr, cp*cr]], dtype=np.float64)
            T = np.eye(4, dtype=np.float64); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
            return T
    a = np.asarray(x, dtype=np.float64)
    if a.size == 16:
        return a.reshape(4, 4)
    raise ValueError(f'Unsupported transform format: {type(x)} {np.shape(x)}')


def actor_class(type_id):
    t = str(type_id).lower()
    if 'rider' in t:
        return 'rider'
    if t.startswith('walker.') or t.startswith('pedestrian.'):
        return 'pedestrian'
    if 'train' in t:
        return 'train'
    if 'motorcycle' in t:
        return 'motorcycle'
    if any(k in t for k in ('crossbike', 'bicycle', 'bike')):
        return 'bicycle'
    if 'bus' in t:
        return 'bus'
    if any(k in t for k in ('truck', 'carlacola', 'european_hgv', 'hgv', 'trailer')):
        return 'truck'
    if t.startswith('vehicle.'):
        return 'car'
    return None


def make_actor_box(actor, is_pedestrian, T_world_lidar):
    T_actor_world = matrix_from_actor_transform(actor['transform'])
    if is_pedestrian:
        extent = np.abs(to_array(actor['bbox_extent']).reshape(3))
        T_box_world = T_actor_world
    else:
        bbox = actor['bbox']
        extent = np.abs(to_array(bbox['extent']).reshape(3))
        T_box_actor = matrix_from_actor_transform(bbox['transform'])
        T_box_world = T_actor_world @ T_box_actor

    T_lidar_world = np.linalg.inv(T_world_lidar)
    center = (T_lidar_world @ np.r_[T_box_world[:3, 3], 1.0])[:3]
    R = T_lidar_world[:3, :3] @ T_box_world[:3, :3]
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    dx, dy, dz = 2.0 * extent
    return np.array([center[0], center[1], center[2], dx, dy, dz, yaw], dtype=np.float32)


def actor_velocity(actor, T_world_lidar):
    if 'velocity' not in actor:
        return np.zeros(2, dtype=np.float32)
    v = to_array(actor['velocity']).reshape(3)
    v_lidar = T_world_lidar[:3, :3].T @ v
    return v_lidar[:2].astype(np.float32)


def occupancy_boxes(path, min_voxels=30, max_voxels=2_000_000):
    with np.load(path, allow_pickle=False) as data:
        occ = np.asarray(data['occupancy'])
        voxel_size = np.asarray(data['voxel_size'], dtype=np.float64).reshape(-1)
        if voxel_size.size == 1:
            voxel_size = np.repeat(voxel_size, 3)
        elif voxel_size.size != 3:
            raise ValueError(f"Unexpected voxel_size shape {voxel_size.shape} in {path}")

        origin = np.asarray(data['voxel_origin'], dtype=np.float64).reshape(-1)
        if origin.size != 3:
            raise ValueError(f"Unexpected voxel_origin shape {origin.shape} in {path}")

    if occ.ndim != 2 or occ.shape[1] != 4:
        raise ValueError(f'Unexpected occupancy shape {occ.shape} in {path}')

    labels = occ[:, 3].astype(np.uint32)
    sem = ((labels >> 11) & 0x1F).astype(np.int64)
    inst = (labels & 0x7FF).astype(np.int64)

    keep = (inst > 0) & np.isin(sem, list(STATIC_OCC_SEMANTIC_IDS))
    occ = occ[keep]
    sem = sem[keep]
    inst = inst[keep]

    boxes, names, ids = [], [], []
    if len(occ) == 0:
        return boxes, names, ids

    xyz = origin + (occ[:, :3].astype(np.float64) + 0.5) * voxel_size
    for semantic_id, instance_id in np.unique(np.column_stack([sem, inst]), axis=0):
        name = SEMANTIC_ID_TO_NAME.get(int(semantic_id))
        if name is None:
            continue
        mask = (sem == semantic_id) & (inst == instance_id)
        pts = xyz[mask]
        if pts.shape[0] < min_voxels or pts.shape[0] > max_voxels:
            continue
        center_xy = pts[:, :2].mean(axis=0)
        xy = pts[:, :2] - center_xy
        cov = np.cov(xy.T) if len(pts) > 2 else np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        yaw = float(np.arctan2(axis[1], axis[0]))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, s], [-s, c]], dtype=np.float64)
        local_xy = xy @ R.T
        mn = local_xy.min(axis=0)
        mx = local_xy.max(axis=0)
        dx, dy = mx - mn
        zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
        center_local = 0.5 * (mn + mx)
        center_world = np.array([
            center_xy[0] + c * center_local[0] - s * center_local[1],
            center_xy[1] + s * center_local[0] + c * center_local[1],
            0.5 * (zmin + zmax),
        ])
        dz = zmax - zmin
        if min(dx, dy, dz) <= 0:
            continue
        boxes.append(np.array([*center_world, dx, dy, dz, yaw], dtype=np.float32))
        names.append(name)
        ids.append((int(semantic_id) << 11) | int(instance_id))
    return boxes, names, ids


def make_sweeps(lidar_dir, frame, lidar_poses, max_sweeps, frame_dt, seq_name):
    sweeps = []
    if max_sweeps <= 1:
        return sweeps
    T_cur_world = np.linalg.inv(lidar_poses[frame])
    for prev in range(frame - 1, max(-1, frame - max_sweeps), -1):
        if prev not in lidar_poses:
            continue
        ply = lidar_dir / f'{prev:04d}.ply'
        if not ply.exists():
            continue
        sweeps.append({
            'lidar_path': (Path('converted_lidar') / seq_name / f'{prev:04d}.bin').as_posix(),
            'transform_matrix': (T_cur_world @ lidar_poses[prev]).astype(np.float32),
            'time_lag': float((prev - frame) * frame_dt),
        })
    return sweeps


def convert_frame(out, seq_name, lidar_dir, traffic_dir, occ_dir, frame,
                  frame_dt, max_sweeps, include_static, static_min_voxels, pose):
    """Note: `pose` matrix passed directly from sequence-level cache."""
    if frame not in pose:
        return None
    traffic_path = traffic_dir / f'{frame:04d}.yaml'
    if not traffic_path.exists():
        return None
    with traffic_path.open() as f:
        traffic = yaml.load(f, Loader=SafeLoader) or {}

    in_ply = lidar_dir / f'{frame:04d}.ply'
    out_bin = out / 'converted_lidar' / seq_name / f'{frame:04d}.bin'
    if not out_bin.exists():
        out_bin.parent.mkdir(parents=True, exist_ok=True)
        load_ply(in_ply).astype(np.float32).tofile(out_bin)

    T_world_lidar = pose[frame]
    ego = traffic.get('ego_vehicles')
    ego_id = int(ego.get('id', -1)) if isinstance(ego, dict) else -1

    boxes, names, velocities, ids = [], [], [], []
    actors = [(a, False) for a in traffic.get('vehicles', [])]
    actors += [(a, True) for a in traffic.get('pedestrians', [])]

    for actor, is_ped in actors:
        aid = int(actor.get('id', -1))
        if aid == ego_id:
            continue
        name = actor_class(actor.get('type', ''))
        if name is None or name not in CLASS_NAMES:
            continue
        try:
            boxes.append(make_actor_box(actor, is_ped, T_world_lidar))
            names.append(name)
            velocities.append(actor_velocity(actor, T_world_lidar))
            ids.append(aid)
        except Exception as exc:
            print(f'WARNING {seq_name} frame {frame} actor {aid}: {exc}')

    if include_static:
        occ_path = occ_dir / f'{frame:04d}.npz'
        if occ_path.exists():
            try:
                sboxes, snames, sids = occupancy_boxes(occ_path, min_voxels=static_min_voxels)
                for b, n, iid in zip(sboxes, snames, sids):
                    if n in CLASS_NAMES:
                        boxes.append(b)
                        names.append(n)
                        velocities.append(np.zeros(2, dtype=np.float32))
                        ids.append(iid)
            except Exception as exc:
                print(f'WARNING {seq_name} frame {frame} occupancy {occ_path}: {exc}')

    if boxes:
        gt_boxes = np.stack(boxes).astype(np.float32)
        gt_names = np.asarray(names)
        gt_velocity = np.stack(velocities).astype(np.float32)
        instance_ids = np.asarray(ids, dtype=np.int64)
    else:
        gt_boxes = np.zeros((0, 7), dtype=np.float32)
        gt_names = np.asarray([], dtype='<U32')
        gt_velocity = np.zeros((0, 2), dtype=np.float32)
        instance_ids = np.zeros((0,), dtype=np.int64)

    return {
        'frame_id': f'{seq_name}_{frame:04d}',
        'token': f'{seq_name}_{frame:04d}',
        'timestamp': float(frame * frame_dt),
        'lidar_path': (Path('converted_lidar') / seq_name / f'{frame:04d}.bin').as_posix(),
        'sweeps': make_sweeps(lidar_dir, frame, pose, max_sweeps, frame_dt, seq_name),
        'gt_boxes': gt_boxes,
        'gt_names': gt_names,
        'gt_velocity': gt_velocity,
        'instance_ids': instance_ids,
    }


def process_sequence(args_tuple):
    """Processes a single sequence entirely."""
    seq_name, lidar_root, poses_root, traffic_root, occ_root, out, args = args_tuple
    
    lidar_dir = lidar_root / seq_name / 'lidar'
    poses_dir = poses_root / seq_name / 'poses'
    traffic_dir = traffic_root / seq_name / 'traffic_info'
    occ_dir = occ_root / seq_name / 'occupancy' / args.occ_variant

    if not (lidar_dir.exists() and poses_dir.exists() and traffic_dir.exists()):
        print(f'WARNING: sequence {seq_name} missing required paths')
        return seq_name, []

    # Read poses once for the entire sequence
    poses = load_matrix_file(poses_dir / 'lidar.txt')
    
    frames = sorted(int(p.stem) for p in lidar_dir.glob('*.ply') if p.stem.isdigit())
    frames = [f for f in frames if f in poses and (traffic_dir / f'{f:04d}.yaml').exists()]
    
    if args.limit:
        frames = frames[:args.limit]

    infos = []
    for frame in frames:
        info = convert_frame(
            out, seq_name, lidar_dir, traffic_dir, occ_dir, frame,
            args.frame_dt, args.max_sweeps, args.include_static, args.static_min_voxels, poses
        )
        if info is not None:
            infos.append(info)

    return seq_name, infos


def discover_sequences(lidar_root):
    return sorted(p.name for p in lidar_root.iterdir() if p.is_dir() and '_Opt_Seq' in p.name)


def auto_split(seq_names, train_frac=0.8, val_frac=0.1, seed=42):
    seqs = list(seq_names)
    random.Random(seed).shuffle(seqs)
    n = len(seqs)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {
        'train': sorted(seqs[:n_train]),
        'val': sorted(seqs[n_train:n_train + n_val]),
        'test': sorted(seqs[n_train + n_val:]),
    }


def get_splits(splits_dir, seq_names, out_dir):
    if splits_dir is not None and all((splits_dir / f'{s}.txt').exists() for s in ('train', 'val', 'test')):
        return {
            s: [Path(x.strip()).name for x in (splits_dir / f'{s}.txt').read_text().splitlines() if x.strip()]
            for s in ('train', 'val', 'test')
        }
    splits = auto_split(seq_names)
    for s, names in splits.items():
        (out_dir / f'auto_split_{s}.txt').write_text('\n'.join(names) + '\n')
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--lidar-dirname', default='all_lidar')
    ap.add_argument('--occupancy-dirname', default='all_occupancy')
    ap.add_argument('--poses-dirname', default='all_poses')
    ap.add_argument('--traffic-dirname', default='all_traffix_info')
    ap.add_argument('--splits-dirname', default='splits')
    ap.add_argument('--max-sweeps', type=int, default=1)
    ap.add_argument('--frame-dt', type=float, default=0.05)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--include-static', action='store_true')
    ap.add_argument('--static-min-voxels', type=int, default=30)
    ap.add_argument('--occ-variant', default='vs_0_1')
    ap.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel processes')
    args = ap.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    lidar_root = root / args.lidar_dirname
    occ_root = root / args.occupancy_dirname
    poses_root = root / args.poses_dirname
    traffic_root = root / args.traffic_dirname
    splits_dir = root / args.splits_dirname
    if not splits_dir.exists():
        splits_dir = None

    seq_names = discover_sequences(lidar_root)
    with (out / 'classes.txt').open('w') as f:
        f.write('\n'.join(CLASS_NAMES) + '\n')

    splits = get_splits(splits_dir, seq_names, out)
    summary = {}

    print(f"Starting processing with {args.workers} worker threads...")

    for split in ('train', 'val', 'test'):
        wanted = splits[split]
        task_args = [
            (seq_name, lidar_root, poses_root, traffic_root, occ_root, out, args)
            for seq_name in wanted
        ]

        infos = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_sequence, t) for t in task_args]
            for idx, future in enumerate(as_completed(futures), 1):
                seq_name, seq_infos = future.result()
                infos.extend(seq_infos)
                print(f'[{idx}/{len(wanted)}] Done sequence {seq_name} ({len(seq_infos)} frames)')

        with (out / f'carlaocc_infos_{split}.pkl').open('wb') as f:
            pickle.dump(infos, f, protocol=pickle.HIGHEST_PROTOCOL)
        summary[split] = len(infos)

    with (out / 'conversion_summary.yaml').open('w') as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f'Done: {summary}')


if __name__ == '__main__':
    main()
