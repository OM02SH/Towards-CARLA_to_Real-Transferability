"""
    Default run
    python add_parked_car_boxes.py build \
        --scene-meshes-dir SceneMeshes \
        --poses-dir all_poses \
        --pkl carlaocc_infos_train.pkl carlaocc_infos_val.pkl carlaocc_infos_test.pkl

    Visual boxes
    python add_parked_car_boxes.py visualize \
        --pkl carlaocc_infos_train.pkl --token Town05_Opt_Seq02_0622 \
        --dataset-root . --out debug_Town05_0622.png
"""
import argparse
import json
import os
import pickle
import re
import sys
from collections import Counter
import numpy as np
import cv2

CLASS_KEYWORD_MAP = [
    ("truck", "truck"),
    ("bus", "truck"),      
    ("car", "car"),        
    ("vehicle", "car"),    
]
UE_UNITS_ARE_CENTIMETERS = False  
CM2M = 0.01 if UE_UNITS_ARE_CENTIMETERS else 1.0
ROTATION_IN_DEGREES = True
BOX_Z_CONVENTION = "center"
PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 5.0)
DEDUP_IOU_THRESH = 0.1
DEDUP_DIST_THRESH = 6.0  
SAME_ACTOR_OVERLAP_THRESH = 0.5
PLAUSIBLE_VEHICLE_LENGTH_M_TIGHT = (2.5, 9.0)
PLAUSIBLE_VEHICLE_WIDTH_M_TIGHT = (1.0, 3.0)
PLAUSIBLE_VEHICLE_LENGTH_M_LOOSE = (1.5, 14.0)
PLAUSIBLE_VEHICLE_WIDTH_M_LOOSE = (1.5, 14.0)
LIDAR_DIM = 4
STATIC_ID_BASE = 900_000 

TOKEN_RE = re.compile(r"^(?P<town>Town\w+?_Opt)_Seq(?P<seq>\d+)_(?P<frame>\d+)$")

def _get(d, *keys, default=None):
    for key in keys:
        cur = d
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default

def get_actor_class(entry):
    return _get(entry, "semantic_label", default="")

def get_actor_location_m(entry):
    loc = _get(entry, "center", "location", "position", "pos", "translation", "world_location")
    if isinstance(loc, dict):
        xyz = (loc["x"], loc["y"], loc["z"])
    elif isinstance(loc, (list, tuple)) and len(loc) >= 3:
        xyz = tuple(loc[:3])
    else:
        xyz = (entry["x"], entry["y"], entry["z"])
    return tuple(float(v) * CM2M for v in xyz)

def get_actor_yaw_rad(entry):
    rot = _get(entry, "rotation", "rot", "orientation")
    yaw = None
    if isinstance(rot, dict):
        yaw = _get(rot, "yaw", "z")
    elif isinstance(rot, (list, tuple)) and len(rot) >= 3:
        yaw = rot[2]
    if yaw is None:
        yaw = entry["yaw"]
    if yaw is None:
        return None
    yaw = float(yaw)
    return np.deg2rad(yaw) if ROTATION_IN_DEGREES else yaw

def get_actor_extent_m(entry):
    ext = _get(entry, "dimensions", "extent", "bbox_extent", "bounding_box.extent", "size")
    if isinstance(ext, dict):
        xyz = (_get(ext, "x", default=4.5), _get(ext, "y", default=1.8), _get(ext, "z", default=1.5))
    elif isinstance(ext, (list, tuple)) and len(ext) >= 3:
        xyz = tuple(ext[:3])
    else:
        xyz = (4.5, 1.8, 1.5)  # generic sedan fallback, better than crashing
    return tuple(float(v) * CM2M for v in xyz)

def yaw_from_aspect_ratio(length, width):
    return 0.0 if length >= width else np.pi / 2.0

def load_obj_vertices_xy(obj_path):
    xs, ys = [], []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    xs.append(float(parts[1]))
                    ys.append(float(parts[2]))
    return np.column_stack([xs, ys]).astype(np.float32)

def obb_yaw_and_extent_from_mesh(obj_path):
    pts = load_obj_vertices_xy(obj_path)
    if len(pts) < 3:
        return None, None, None, "too_few_vertices"
    (_cx, _cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    if w <= 0 or h <= 0:
        return None, None, None, "degenerate_rect"
    length, width = (w, h) if w >= h else (h, w)
    yaw_deg = angle_deg if w >= h else angle_deg + 90.0
    return float(np.deg2rad(yaw_deg)), float(length), float(width), "ok"

def normalize_vehicle_class(raw):
    s = str(raw).lower()
    for kw, mapped in CLASS_KEYWORD_MAP:
        if kw in s:
            return mapped
    return None

def _aabb_xy_overlap_fraction(min_a, max_a, min_b, max_b):
    ix0, iy0 = max(min_a[0], min_b[0]), max(min_a[1], min_b[1])
    ix1, iy1 = min(max_a[0], max_b[0]), min(max_a[1], max_b[1])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, max_a[0] - min_a[0]) * max(0.0, max_a[1] - min_a[1])
    area_b = max(0.0, max_b[0] - min_b[0]) * max(0.0, max_b[1] - min_b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0

def dedupe_same_actor_parts(vehicles, overlap_thresh=SAME_ACTOR_OVERLAP_THRESH):
    n = len(vehicles)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            frac = _aabb_xy_overlap_fraction(
                vehicles[i]["bbox_min_xy"], vehicles[i]["bbox_max_xy"],
                vehicles[j]["bbox_min_xy"], vehicles[j]["bbox_max_xy"])
            if frac >= overlap_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    def _raw_area(k):
        mn, mx = vehicles[k]["bbox_min_xy"], vehicles[k]["bbox_max_xy"]
        return max(0.0, mx[0] - mn[0]) * max(0.0, mx[1] - mn[1])

    return [vehicles[max(idxs, key=_raw_area)] for idxs in clusters.values()]

def world_to_lidar_transform(lidar_to_world_4x4):
    R_lw = lidar_to_world_4x4[:3, :3]
    t_lw = lidar_to_world_4x4[:3, 3]
    R_wl = R_lw.T
    t_wl = -R_wl @ t_lw
    return R_wl, t_wl

def transform_box_to_lidar(loc_world, yaw_world, R_wl, t_wl):
    p_lidar = R_wl @ np.asarray(loc_world) + t_wl
    fwd_world = np.array([np.cos(yaw_world), np.sin(yaw_world), 0.0])
    fwd_lidar = R_wl @ fwd_world
    yaw_lidar = float(np.arctan2(fwd_lidar[1], fwd_lidar[0]))
    return p_lidar, yaw_lidar

def box_corners_bev(box):
    x, y, _, l, w, _, yaw = box
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return corners @ R.T + np.array([x, y])

def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def clip_polygon(subject, clip):
    """Sutherland-Hodgman convex polygon clipping."""
    output = subject.tolist()
    cn = len(clip)
    for i in range(cn):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % cn]
        edge = b - a

        def inside(p):
            d = np.asarray(p) - a
            return (edge[0] * d[1] - edge[1] * d[0]) >= 0 

        inp = output
        output = []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    output.append(_seg_intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_seg_intersect(prev, cur, a, b))
    return np.array(output) if output else np.zeros((0, 2))

def _seg_intersect(p1, p2, p3, p4):
    p1, p2, p3, p4 = map(np.asarray, (p1, p2, p3, p4))
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]  
    if abs(denom) < 1e-9:
        return p2
    d3 = p3 - p1
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / denom
    return p1 + t * d1

def bev_iou(box_a, box_b):
    ca, cb = np.array(box_a[:2]), np.array(box_b[:2])
    if np.linalg.norm(ca - cb) > DEDUP_DIST_THRESH:
        return 0.0
    pa, pb = box_corners_bev(box_a), box_corners_bev(box_b)
    inter = clip_polygon(pa, pb)
    if len(inter) < 3:
        return 0.0
    area_i = polygon_area(inter)
    area_u = polygon_area(pa) + polygon_area(pb) - area_i
    return area_i / area_u if area_u > 0 else 0.0

def in_pc_range(xyz):
    x, y, z = xyz
    x0, y0, z0, x1, y1, z1 = PC_RANGE
    return x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1

def _guess_lidar_dim(path):
    size = os.path.getsize(path)
    for d in (4, 5, 6):
        if size % (d * 4) == 0 and size // (d * 4) > 50:
            return d
    return LIDAR_DIM

def count_points_in_box(points_xyz, box):
    x, y, z, l, w, h, yaw = box
    p = points_xyz - np.array([x, y, z])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p = p @ R.T
    m = ((np.abs(p[:, 0]) <= l / 2) & (np.abs(p[:, 1]) <= w / 2) & (np.abs(p[:, 2]) <= h / 2))
    return int(m.sum())

def load_town_seq_poses(poses_dir, seq_name):
    candidates = [
        os.path.join(poses_dir, seq_name, "poses", "lidar.txt"),
        os.path.join(poses_dir, seq_name, "lidar.txt"),
        os.path.join(poses_dir, seq_name, "lidar_poses.txt"),
        os.path.join(poses_dir, seq_name, "lidar_poses.npy"),
        os.path.join(poses_dir, f"{seq_name}.txt"),
        os.path.join(poses_dir, f"{seq_name}.npy"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        if path.endswith(".npy"):
            arr = np.load(path)
            if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                return {i: arr[i] for i in range(len(arr))}
            if arr.ndim == 2 and arr.shape[1] == 7:  # tx ty tz qx qy qz qw
                return {i: _pose7_to_matrix(arr[i]) for i in range(len(arr))}
        else:
            rows = np.loadtxt(path)
            if rows.ndim == 1:
                rows = rows[None, :]
            ncol = rows.shape[1]
            if ncol == 17:
                return {int(round(rows[i, 0])): rows[i, 1:].reshape(4, 4) for i in range(len(rows))}
            if ncol == 13:
                return {int(round(rows[i, 0])): np.vstack([rows[i, 1:].reshape(3, 4), [0, 0, 0, 1]])
                        for i in range(len(rows))}
            if ncol == 8:  # frame_idx + quat7
                return {int(round(rows[i, 0])): _pose7_to_matrix(rows[i, 1:]) for i in range(len(rows))}
            if ncol == 7:
                return {i: _pose7_to_matrix(rows[i]) for i in range(len(rows))}
            if ncol in (12, 16):
                return {i: rows[i].reshape(4, 4) if ncol == 16
                         else np.vstack([rows[i].reshape(3, 4), [0, 0, 0, 1]])
                         for i in range(len(rows))}
    return None

def _pose7_to_matrix(row):
    tx, ty, tz, qx, qy, qz, qw = row
    def _quat_to_rot(x, y, z, w):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    R = _quat_to_rot(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def load_town_static_vehicles(scene_meshes_dir, town, mesh_fit_stats=None):
    if mesh_fit_stats is None:
        mesh_fit_stats = Counter()
    for fname in ("actor_info.json", "exported_scene_info.json"):
        path = os.path.join(scene_meshes_dir, town, fname)
        if not os.path.exists(path):
            continue
        mesh_dir = os.path.dirname(path)
        with open(path) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else None
        if entries is None and isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    entries = v
                    break
        if entries is None:
            continue

        # Step 1: collect raw candidates
        raw = []
        for e in entries:
            cls = normalize_vehicle_class(get_actor_class(e))
            if cls is None:
                continue
            bbox_min = _get(e, "bbox_min", default=None)
            bbox_max = _get(e, "bbox_max", default=None)
            loc = get_actor_location_m(e)
            if bbox_min is None or bbox_max is None:
                bbox_min_xy = (loc[0], loc[1])
                bbox_max_xy = (loc[0], loc[1])
            else:
                bbox_min_xy = (float(bbox_min[0]) * CM2M, float(bbox_min[1]) * CM2M)
                bbox_max_xy = (float(bbox_max[0]) * CM2M, float(bbox_max[1]) * CM2M)
            raw.append({
                "cls": cls,
                "loc_world": loc,
                "extent": get_actor_extent_m(e),
                "explicit_yaw": get_actor_yaw_rad(e),
                "instance_id": e.get("instance_id"),
                "exported_file": e.get("exported_file", ""),
                "bbox_min_xy": bbox_min_xy,
                "bbox_max_xy": bbox_max_xy,
            })

        # Step 2: collapse same-actor sub-parts (glass, trim) into their
        # parent body BEFORE spending a mesh fit on parts we'll discard.
        raw = dedupe_same_actor_parts(raw)

        # Step 3: fit yaw/extent for each surviving candidate, preferring
        # the exact mesh-OBB fit over the AABB aspect-ratio heuristic.
        vehicles = []
        for v in raw:
            yaw, extent, source = v["explicit_yaw"], v["extent"], "explicit"
            if yaw is None:
                obj_path = os.path.join(mesh_dir, "bg_actors", v["exported_file"])
                fit_yaw, fit_l, fit_w, reason = obb_yaw_and_extent_from_mesh(obj_path)
                mesh_fit_stats[reason] += 1
                if reason == "ok":
                    yaw, extent, source = fit_yaw, (fit_l, fit_w, extent[2]), "mesh-OBB"
                else:
                    yaw, source = yaw_from_aspect_ratio(extent[0], extent[1]), "AABB-heuristic"
                    if extent[0] < extent[1]:
                        extent = (extent[1], extent[0], extent[2])
            vehicles.append({
                "cls": v["cls"], "loc_world": v["loc_world"], "yaw_world": yaw,
                "extent": extent, "instance_id": v["instance_id"], "yaw_source": source,
            })

        return vehicles
    return []

def build(scene_meshes_dir, poses_dir, pkl_paths, write, lidar_root):
    town_vehicles_cache = {}
    poses_cache = {}
    stats = Counter()
    mesh_fit_stats = Counter()

    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        infos = data["infos"] if isinstance(data, dict) and "infos" in data else \
            data["data_list"] if isinstance(data, dict) and "data_list" in data else data

        for rec in infos:
            token = rec.get("token", rec.get("frame_id"))
            m = TOKEN_RE.match(token)
            parsed = m.group("town"), m.group("seq"), int(m.group("frame"))
            if parsed is None:
                stats["unparseable_token"] += 1
                continue
            town, seq, frame_idx = parsed
            seq_name = f"{town}_Seq{seq}"

            if town not in town_vehicles_cache:
                town_vehicles_cache[town] = load_town_static_vehicles(scene_meshes_dir, town, mesh_fit_stats)
            vehicles = town_vehicles_cache[town]
            if not vehicles:
                stats["towns_with_no_static_vehicles_found"] += 1
                continue

            if seq_name not in poses_cache:
                poses_cache[seq_name] = load_town_seq_poses(poses_dir, seq_name)
            poses = poses_cache[seq_name]
            if not poses or frame_idx not in poses:
                stats["frames_with_no_pose"] += 1
                continue
            # Step 4
            # Transform center and heading into each frame’s LiDAR frame with that frame’s pose
            R_wl, t_wl = world_to_lidar_transform(poses[frame_idx])

            existing_boxes = list(rec["gt_boxes"])
            existing_names = list(rec["gt_names"])
            new_boxes, new_names, new_instance_ids = [], [], []

            for i, v in enumerate(vehicles):
                p_lidar, yaw_lidar = transform_box_to_lidar(v["loc_world"], v["yaw_world"], R_wl, t_wl)
                if not in_pc_range(p_lidar):
                    continue
                l, w, h = v["extent"]
                z = p_lidar[2] if BOX_Z_CONVENTION == "bottom" else p_lidar[2]
                candidate = np.array([p_lidar[0], p_lidar[1], z, l, w, h, yaw_lidar], dtype=np.float32)

                dup = False
                for eb, en in zip(existing_boxes, existing_names):
                    if en not in ("car", "truck"):
                        continue
                    #   Step 5
                    #   Discard candidates whose bird’s-eye-view IoU with an existing car or truck box exceeds 0.1
                    if bev_iou(candidate, eb) > DEDUP_IOU_THRESH:
                        dup = True
                        break
                if dup:
                    stats["deduped_against_existing"] += 1
                    continue

                new_boxes.append(candidate)
                new_names.append(v["cls"])
                new_instance_ids.append(v["instance_id"] if v["instance_id"] is not None else i)
                stats[f"added_{v['cls']}"] += 1

            if not new_boxes:
                continue
            #   Step 6
            #   Append the survivors 
            n_new = len(new_boxes)
            rec["gt_boxes"] = np.concatenate([rec["gt_boxes"], np.stack(new_boxes)], axis=0)
            rec["gt_names"] = np.concatenate([rec["gt_names"], np.array(new_names, dtype=rec["gt_names"].dtype)])
            if "gt_velocity" in rec:
                rec["gt_velocity"] = np.concatenate(
                    [rec["gt_velocity"], np.zeros((n_new, 2), dtype=np.float32)], axis=0)
            if "instance_ids" in rec:
                new_ids = [STATIC_ID_BASE + iid for iid in new_instance_ids]
                rec["instance_ids"] = np.concatenate([rec["instance_ids"], np.array(new_ids)])
            if "num_lidar_pts" in rec:
                lidar_path = os.path.join(lidar_root, rec["lidar_path"]) if lidar_root else rec["lidar_path"]
                if os.path.exists(lidar_path):
                    dim = _guess_lidar_dim(lidar_path)
                    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, dim)[:, :3]
                    counts = [count_points_in_box(pts, b) for b in new_boxes]
                else:
                    counts = [0] * n_new
                rec["num_lidar_pts"] = np.concatenate([rec["num_lidar_pts"], np.array(counts)])
            if "valid_flag" in rec:
                rec["valid_flag"] = np.concatenate(
                    [rec["valid_flag"], np.ones(n_new, dtype=rec["valid_flag"].dtype)])

        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

def cmd_visualize(pkl_path, token, dataset_root, out_path):
    """
        This function was written by a LLM
    """
    from PIL import Image, ImageDraw
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"] if isinstance(data, dict) and "infos" in data else \
        data["data_list"] if isinstance(data, dict) and "data_list" in data else data
    rec = next((r for r in infos if r.get("token", r.get("frame_id")) == token), None)
    if rec is None:
        print(f"token {token} not found in {pkl_path}")
        return

    cam = rec["cams"]["cam_00"]
    img_path = os.path.join(dataset_root, cam["data_path"]) if dataset_root else cam["data_path"]
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    R = np.array(cam["sensor2lidar_rotation"])
    t = np.array(cam["sensor2lidar_translation"])
    K = np.array(cam["camera_intrinsics"])

    def draw_box(box, color):
        x, y, z, l, w, h, yaw = box
        dx, dy, dz = l / 2, w / 2, h / 2
        local = np.array([[i, j, k] for i in (-dx, dx) for j in (-dy, dy) for k in (-dz, dz)])
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners_lidar = (local @ Rz.T) + np.array([x, y, z])
        corners_cam = (corners_lidar - t) @ R  # inverse of sensor2lidar (R orthonormal)
        pts2d = []
        for p in corners_cam:
            if p[2] <= 0.1:
                pts2d.append(None)
                continue
            uv = K @ p
            pts2d.append((uv[0] / uv[2], uv[1] / uv[2]))
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                 (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for a, b in edges:
            if pts2d[a] and pts2d[b]:
                draw.line([pts2d[a], pts2d[b]], fill=color, width=2)

    for box in rec["gt_boxes"]:
        draw_box(box, "lime")
    print(f"drew {len(rec['gt_boxes'])} existing boxes in green; "
          f"re-run after `build` and diff to see the new ones, or pass "
          f"--extra-boxes if you saved them separately.")
    img.save(out_path)
    print(f"saved {out_path}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build")
    p.add_argument("--scene-meshes-dir", required=True)
    p.add_argument("--poses-dir", required=True)
    p.add_argument("--pkl", nargs="+", required=True)
    p.add_argument("--lidar-root", default="")
    p.add_argument("--write", action="store_true")

    p = sub.add_parser("visualize")
    p.add_argument("--pkl", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--dataset-root", default="")
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "build":
        build(args.scene_meshes_dir, args.poses_dir, args.pkl, args.write, args.lidar_root)
    elif args.cmd == "visualize":
        cmd_visualize(args.pkl, args.token, args.dataset_root, args.out)

if __name__ == "__main__":
    sys.exit(main())
