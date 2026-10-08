#!/usr/bin/env python3
"""
add_parked_car_boxes.py
========================
Recover missing 3D boxes for STATIC / PARKED vehicles in a CarlaOcc ->
TransFusion-L / BEVFusion conversion.

BACKGROUND (why the boxes are missing)
---------------------------------------
CarlaOcc splits every town into two separate systems:

  * fg_actors / traffic_info  -> vehicles+walkers actually driven by CARLA's
    traffic manager. This is what your existing gt_boxes come from.
  * SceneMeshes/<Town>/bg_actors (+ bg_actor_occ) -> static, non-actor scene
    dressing: buildings, landscape chunks, and -- in towns that use them --
    decorative parked vehicles placed once by the map designer. These have
    occupancy baked in and are visible in RGB/LiDAR, but were never a CARLA
    actor, so they never reach traffic_info / your gt_boxes.

This script reads SceneMeshes/<Town>/actor_info.json (small, per-town,
independent of the heavy occupancy/mesh downloads), keeps the vehicle-class
entries, transforms them from world space into each frame's LiDAR frame
using your all_poses data, de-duplicates against boxes you already have,
and appends the rest as new zero-velocity boxes.

KNOWN DATA QUIRK (found by inspecting real output, 2024 pass)
---------------------------------------------------------------
Some vehicle models are exported as MULTIPLE separate StaticMeshActor
entries -- e.g. a Cybertruck's window glass panels each get their own
actor_info.json entry, all tagged semantic_label="Car" just like the body,
all clustered at ~the same world location, with tiny/oddly-shaped extents.
Treated naively, one physical vehicle can produce 10+ spurious boxes.
`dedupe_same_actor_parts()` + `plausible_vehicle_size()` below collapse
and filter these before they ever reach the pkl. VERIFY this against every
town you use -- the Cybertruck pattern may not be the only multi-mesh model.

WHAT I COULD NOT VERIFY WITHOUT YOUR FILES
-------------------------------------------
1. The exact keys inside actor_info.json / exported_scene_info.json.
2. The exact file format under all_poses/.
3. Whether gt_boxes' z is box-center or box-bottom convention.
4. converted_lidar .bin point dimensionality (assumed 4: x,y,z,intensity).

Everything that depends on these has a `_get(...)` fallback-key lookup or a
CONFIG flag, and is flagged "VERIFY" in a comment. Run the `inspect-*`
subcommands first (100% read-only, safe to run right now) and skim the
output; fix the CONFIG block to match what you actually see, or paste me
the output and I'll do it.

USAGE
-----
    # 1. Safe, read-only reconnaissance -- do this first.
    python add_parked_car_boxes.py inspect-pkl carlaocc_infos_train.pkl
    python add_parked_car_boxes.py inspect-actors SceneMeshes/Town05_Opt/actor_info.json
    python add_parked_car_boxes.py inspect-poses all_poses/Town05_Opt_Seq02

    # 2. Dry run (default) -- reports counts, writes nothing.
    python add_parked_car_boxes.py build \
        --scene-meshes-dir SceneMeshes \
        --poses-dir all_poses \
        --pkl carlaocc_infos_train.pkl carlaocc_infos_val.pkl carlaocc_infos_test.pkl

    # 3. Visual QA on one frame before trusting it dataset-wide.
    python add_parked_car_boxes.py visualize \
        --pkl carlaocc_infos_train.pkl --token Town05_Opt_Seq02_0622 \
        --dataset-root . --out debug_Town05_0622.png

    # 4. Once the dry run counts look sane, commit for real.
    python add_parked_car_boxes.py build ... --write

NOTE on yaw: parked-vehicle orientation is fit directly from each vehicle's
own exported mesh (minimum-area oriented rectangle / rotating calipers via
cv2.minAreaRect on the vertex footprint) -- see `obb_yaw_and_extent_from_mesh`.
This is exact regardless of whether the car is axis-aligned, diagonally
parked, or on a curve, and needs no road/lane data. It cannot resolve the
180-degree front/back ambiguity (nothing in a silhouette says which end is
the bumper) -- that is a fundamental limit of geometry-only fitting, not
something a road heading would fix either, since a road only gives street
direction, not which way into a (possibly perpendicular) stall the car sits.
If a mesh file is missing/unreadable, this falls back to the old
axis-aligned-AABB aspect-ratio heuristic (0 or 90 degrees), flagged in the
per-vehicle debug print.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np

# ============================================================================
# CONFIG -- edit these after running the inspect-* subcommands.
# ============================================================================

# Vehicle-ish class strings (case-insensitive substring match) -> the class
# name your pkl's gt_names already uses (must match classes.txt exactly).
CLASS_KEYWORD_MAP = [
    ("truck", "truck"),
    ("bus", "truck"),      # VERIFY: point this at whatever your classes.txt
    ("car", "car"),        # calls buses, if it distinguishes them at all.
    ("vehicle", "car"),    # generic "vehicle.*" blueprint id fallback.
]

# UE5 scene-metadata JSON is normally exported in centimeters, CARLA's own
# Python API and traffic_info are in meters. VERIFY against inspect-actors
# output: if 'location' values are in the thousands, this is correct.
UE_UNITS_ARE_CENTIMETERS = False  # CONFIRMED: actor_info.json + all_poses are both in meters
CM2M = 0.01 if UE_UNITS_ARE_CENTIMETERS else 1.0

# UE rotations are usually degrees. VERIFY the same way.
ROTATION_IN_DEGREES = True

# gt_boxes z convention: "center" (geometric center) or "bottom".
# VERIFY: for an existing car box, bottom = z - h/2 if "center", or z if
# "bottom" -- whichever lands consistently near your ground height across
# many boxes is the right one. Un-verified default below is "center", which
# is the more common mmdet3d LiDARInstance3DBoxes convention.
BOX_Z_CONVENTION = "center"

# Point-cloud / detection range filter applied to newly added boxes, in
# LiDAR frame, meters: (x_min, y_min, z_min, x_max, y_max, z_max).
# VERIFY against conversion_summary.yaml / your model configs. CarlaOcc's
# published occupancy volume is roughly 76.8 x 51.2 x 13 m, ego-relative --
# adjust to match whatever your BEVFusion/TransFusion-L config actually uses.
PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 5.0)

# BEV IoU threshold above which a candidate parked-car box is considered a
# duplicate of something you already annotated (and is dropped).
DEDUP_IOU_THRESH = 0.1
DEDUP_DIST_THRESH = 6.0  # meters, coarse center-distance prefilter

# Fraction of the SMALLER of two candidates' world-XY AABB area that must
# be covered by their intersection for them to be treated as sub-parts of
# the SAME physical vehicle (e.g. a Cybertruck's body + its individually
# -exported glass panels), rather than two distinct parked cars.
# REPLACES an earlier pure center-distance rule (SAME_ACTOR_DIST_THRESH),
# which was DISPROVEN by real data: two genuinely distinct Nissan Patrols
# in Town05 sit only 2.72m apart (adjacent parking bays) and were wrongly
# merged by a 3.0m distance cutoff, while their AABBs have zero overlap.
# A sub-mesh (glass, trim) is always geometrically nested inside its
# parent's AABB -> overlap fraction ~1.0. Two side-by-side distinct
# vehicles, even when close, do not nest -> overlap fraction ~0.
SAME_ACTOR_OVERLAP_THRESH = 0.5

# Sanity bounds (meters) on plausible vehicle footprint. Two tiers:
# TIGHT applies when the extent came from an exact mesh-OBB fit (reliable).
# LOOSE applies when only the AABB-heuristic fallback extent is available --
# proven UNRELIABLE for anything not axis-aligned (a real, diagonally-
# oriented Cybertruck showed an AABB width of 4.83m -- pure rotation
# artifact, not a real dimension) so we only screen out obvious junk there,
# not legitimate-but-rotated vehicles. Fix the mesh-OBB pipeline (see the
# per-run failure-reason summary) rather than tightening LOOSE bounds.
PLAUSIBLE_VEHICLE_LENGTH_M_TIGHT = (2.5, 9.0)
PLAUSIBLE_VEHICLE_WIDTH_M_TIGHT = (1.0, 3.0)
PLAUSIBLE_VEHICLE_LENGTH_M_LOOSE = (1.5, 14.0)
PLAUSIBLE_VEHICLE_WIDTH_M_LOOSE = (1.5, 14.0)

# assumed LiDAR .bin column count for num_lidar_pts recomputation; auto-probed
# against file size if this is wrong (see `_guess_lidar_dim`).
LIDAR_DIM = 4

STATIC_ID_BASE = 900_000  # new instance_ids start here, offset per town

TOKEN_RE = re.compile(r"^(?P<town>Town\w+?_Opt)_Seq(?P<seq>\d+)_(?P<frame>\d+)$")


# ============================================================================
# Small dict/JSON fallback-key helpers
# ============================================================================

def _get(d, *keys, default=None):
    """Try several possible (possibly dotted) key paths on a dict."""
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
    """Returns (x, y, z) in meters, world frame."""
    loc = _get(entry, "center", "location", "position", "pos", "translation", "world_location")
    if isinstance(loc, dict):
        xyz = (_get(loc, "x", default=0.0), _get(loc, "y", default=0.0), _get(loc, "z", default=0.0))
    elif isinstance(loc, (list, tuple)) and len(loc) >= 3:
        xyz = tuple(loc[:3])
    else:
        # flat x/y/z on the entry itself
        xyz = (_get(entry, "x", default=0.0), _get(entry, "y", default=0.0), _get(entry, "z", default=0.0))
    return tuple(float(v) * CM2M for v in xyz)


def get_actor_yaw_rad(entry):
    """Returns None if there is no explicit rotation field (CONFIRMED: this
    schema -- actor_info.json with idx/instance_id/center/bbox_min/bbox_max/
    dimensions -- has none). Caller falls back to the mesh-OBB fit, then to
    an aspect-ratio heuristic if the mesh isn't available."""
    rot = _get(entry, "rotation", "rot", "orientation")
    yaw = None
    if isinstance(rot, dict):
        yaw = _get(rot, "yaw", "z")
    elif isinstance(rot, (list, tuple)) and len(rot) >= 3:
        yaw = rot[2]
    if yaw is None:
        yaw = _get(entry, "yaw")
    if yaw is None:
        return None
    yaw = float(yaw)
    return np.deg2rad(yaw) if ROTATION_IN_DEGREES else yaw


def get_actor_extent_m(entry):
    """Returns (length, width, height) in meters -- full extents.
    CONFIRMED for this schema: 'dimensions' == bbox_max - bbox_min already,
    i.e. a full size, not a half-extent -- no doubling here."""
    ext = _get(entry, "dimensions", "extent", "bbox_extent", "bounding_box.extent", "size")
    if isinstance(ext, dict):
        xyz = (_get(ext, "x", default=4.5), _get(ext, "y", default=1.8), _get(ext, "z", default=1.5))
    elif isinstance(ext, (list, tuple)) and len(ext) >= 3:
        xyz = tuple(ext[:3])
    else:
        xyz = (4.5, 1.8, 1.5)  # generic sedan fallback, better than crashing
    return tuple(float(v) * CM2M for v in xyz)


def yaw_from_aspect_ratio(length, width):
    """LAST-RESORT fallback only, used when no explicit rotation field AND
    no readable mesh file are available. World-space AABB dimensions are
    then the only geometric cue we have, so this assumes the object's long
    axis is aligned with a world axis (0 or 90 degrees) -- correct only for
    vehicles parked parallel to a rectilinear grid; wrong for diagonal
    parking / curved roads. Prefer `obb_yaw_and_extent_from_mesh` whenever
    the actor's exported .obj is available."""
    return 0.0 if length >= width else np.pi / 2.0


def load_obj_vertices_xy(obj_path):
    """Minimal OBJ parser -- just pulls vertex (x, y) positions, ignoring
    faces/normals/materials, since that's all an oriented-footprint fit
    needs."""
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
    """Fit the minimum-area ORIENTED rectangle (rotating calipers over the
    2D convex hull, via cv2.minAreaRect) to the vehicle's own mesh
    footprint. This gives an exact world-frame yaw for the vehicle
    regardless of whether it happens to be axis-aligned, diagonally
    parked, or sitting on a curved road -- no road/lane data needed at all.

    Returns (yaw_rad, length_m, width_m, "ok") on success, or
    (None, None, None, reason) on failure, where reason is one of:
      "cv2_not_installed"  -- `pip install opencv-python-headless`
      "mesh_file_not_found" -- exported_file isn't at the expected path;
                                check whether you actually have the mesh
                                payload downloaded, not just actor_info.json
      "too_few_vertices"    -- malformed/empty obj
      "degenerate_rect"     -- minAreaRect returned a zero-area box

    ALWAYS check the aggregated reason counts a run prints -- if every
    entry fails with the same reason, the fit isn't running at all and
    every yaw/extent you're seeing is the (unreliable) AABB fallback,
    not a real fit.

    KNOWN LIMITATION even when this succeeds: like any silhouette-only
    fit, it cannot tell front from back -- the returned yaw is correct up
    to a 180-degree flip. For static, zero-velocity boxes this is a
    well-known, generally acceptable limitation.
    """
    try:
        import cv2
    except ImportError:
        return None, None, None, "cv2_not_installed"
    if not os.path.exists(obj_path):
        return None, None, None, "mesh_file_not_found"
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
    """Fraction of the SMALLER box's XY area covered by the intersection.
    ~1.0 when one box is nested inside the other (sub-mesh inside parent
    body); ~0.0 for two side-by-side boxes with little/no overlap, even
    when their centers are close together."""
    ix0, iy0 = max(min_a[0], min_b[0]), max(min_a[1], min_b[1])
    ix1, iy1 = min(max_a[0], max_b[0]), min(max_a[1], max_b[1])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, max_a[0] - min_a[0]) * max(0.0, max_a[1] - min_a[1])
    area_b = max(0.0, max_b[0] - min_b[0]) * max(0.0, max_b[1] - min_b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def dedupe_same_actor_parts(vehicles, overlap_thresh=SAME_ACTOR_OVERLAP_THRESH):
    """Multiple StaticMeshActor sub-parts (glass, trim, panels) of one
    physical vehicle can all carry the same vehicle-ish semantic_label --
    e.g. a Cybertruck's ~10 individually-exported window panes, all tagged
    "Car" just like the body, all geometrically nested inside the body's
    AABB. Collapse any cluster of candidates whose raw world-XY AABBs
    overlap by at least `overlap_thresh` (as a fraction of the smaller
    box) into a single entry: keep the one with the largest raw AABB
    footprint, which is the actual body mesh in every case inspected so
    far.

    Uses raw AABB (bbox_min_xy/bbox_max_xy), NOT a center-distance cutoff:
    real, distinct, closely-spaced parked cars (adjacent bays can be just
    ~2.7m apart center-to-center) do not nest inside each other even when
    close, so this correctly leaves them separate.

    Uses union-find over the pairwise overlap graph (TRANSITIVE grouping),
    not a single greedy pass: two sub-parts that each overlap the body but
    not each other directly (e.g. two window panes on opposite sides of a
    vehicle) must still end up in the same cluster via the body as the
    common hub. A naive greedy pass can strand the second such part as an
    orphaned singleton once the body has already been claimed by the
    first -- caught in testing precisely because a leftover orphaned
    window pane is small enough to (usually) fail the plausibility filter
    downstream, but that's luck, not a guarantee."""
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


def plausible_vehicle_size(extent, source):
    """Sanity check on vehicle footprint, calibrated to how reliable the
    extent's SOURCE is: a mesh-OBB fit gives true dimensions (tight
    bounds); the AABB-heuristic fallback can inflate a rotated vehicle's
    apparent width well beyond its real size (a diagonally-oriented
    Cybertruck showed an AABB width of 4.83m), so that source only gets a
    loose sanity check meant to catch obvious junk, not to validate real
    dimensions."""
    l, w = sorted(extent[:2], reverse=True)
    if source == "mesh-OBB":
        lo_l, hi_l = PLAUSIBLE_VEHICLE_LENGTH_M_TIGHT
        lo_w, hi_w = PLAUSIBLE_VEHICLE_WIDTH_M_TIGHT
    else:
        lo_l, hi_l = PLAUSIBLE_VEHICLE_LENGTH_M_LOOSE
        lo_w, hi_w = PLAUSIBLE_VEHICLE_WIDTH_M_LOOSE
    return lo_l <= l <= hi_l and lo_w <= w <= hi_w


# ============================================================================
# inspect-* subcommands (read-only, safe to run immediately)
# ============================================================================

def cmd_inspect_pkl(path):
    print(f"\n{'=' * 70}\nPKL: {path}\n{'=' * 70}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"top-level type: {type(data)}")
    if isinstance(data, dict):
        print(f"top-level keys: {list(data.keys())}")
        infos = data.get("infos", data.get("data_list"))
        meta = {k: v for k, v in data.items() if k not in ("infos", "data_list")}
        if meta:
            print(f"other top-level entries: {meta}")
    elif isinstance(data, list):
        infos = data
    else:
        print("Unrecognized top-level structure:")
        print(repr(data)[:2000])
        return
    if infos is None:
        print("No 'infos'/'data_list' key found -- inspect the dict manually.")
        return
    print(f"num records: {len(infos)}")
    rec = infos[0]
    print("\nfirst record keys / types:")
    for k, v in rec.items():
        if hasattr(v, "shape"):
            print(f"  {k}: array shape={v.shape} dtype={v.dtype}")
        else:
            s = repr(v)
            print(f"  {k}: {type(v).__name__} = {s[:150]}")
    names = Counter()
    for r in infos:
        for n in r.get("gt_names", []):
            names[n] += 1
    print(f"\nclass counts across split: {dict(names)}")
    print(f"sample tokens: {[r.get('token', r.get('frame_id', '?')) for r in infos[:5]]}")


def cmd_inspect_actors(path):
    print(f"\n{'=' * 70}\nACTOR JSON: {path}\n{'=' * 70}")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"top-level type: {type(data)}")
    entries = data if isinstance(data, list) else None
    if entries is None and isinstance(data, dict):
        print(f"top-level keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"-> treating '{k}' as the actor list ({len(v)} entries)")
                entries = v
                break
    if entries is None:
        print("Could not find a list of actor dicts. Raw head:")
        print(json.dumps(data, indent=2)[:2000])
        return
    print(f"num entries: {len(entries)}")
    print(f"\nfirst entry:\n{json.dumps(entries[0], indent=2)[:1500]}")
    key_union = set()
    for e in entries:
        if isinstance(e, dict):
            key_union |= set(e.keys())
    print(f"\nkey union across all entries: {sorted(key_union)}")
    for cand in ("class", "category", "type", "tag", "actor_type", "semantic",
                 "semantic_label", "label", "blueprint", "blueprint_id"):
        vals = Counter(str(e.get(cand)) for e in entries if isinstance(e, dict) and cand in e)
        if vals:
            print(f"\nvalues for key '{cand}': {dict(vals.most_common(30))}")
    n_vehicle_like = sum(1 for e in entries if normalize_vehicle_class(get_actor_class(e)))
    print(f"\nentries that normalize_vehicle_class() currently recognizes as a vehicle: {n_vehicle_like}")

    town = os.path.basename(os.path.dirname(path)) or "."
    scene_meshes_dir = os.path.dirname(os.path.dirname(path)) or "."
    mesh_fit_stats = Counter()
    vehicles = load_town_static_vehicles(scene_meshes_dir, town, mesh_fit_stats)
    print("final len veh", len(vehicles))
    print(f"\n{'=' * 70}\nMESH-OBB FIT DIAGNOSTIC -- READ THIS FIRST\n{'=' * 70}")
    if not mesh_fit_stats:
        print("  every vehicle entry had an explicit rotation field -- no mesh fit needed.")
    else:
        for reason, count in mesh_fit_stats.most_common():
            print(f"  {reason}: {count}")
        if mesh_fit_stats.get("ok", 0) == 0 and sum(mesh_fit_stats.values()) > 0:
            print("  *** mesh-OBB fit engaged ZERO times -- every yaw/extent below is the")
            print("      unreliable AABB-heuristic fallback. Fix the reason above (install")
            print("      opencv-python-headless, and/or confirm the .obj files referenced by")
            print("      'exported_file' actually exist under this town's mesh directory)")
            print("      before trusting anything downstream.")

    print(f"\n{'=' * 70}\nFULL DATA FOR ALL {n_vehicle_like} RECOGNIZED VEHICLE ENTRIES "
          f"(before same-actor dedup)\n{'=' * 70}")
    for e in entries:
        if not normalize_vehicle_class(get_actor_class(e)):
            continue
        print(json.dumps(e, indent=2))
        print("-" * 70)

    print(f"\n{'=' * 70}\nAFTER dedupe_same_actor_parts (bbox-overlap) + plausible_vehicle_size\n{'=' * 70}")
    for v in vehicles:
        print(f"  cls={v['cls']:5s} instance_id={v['instance_id']:>4} "
              f"loc_m={tuple(round(c, 2) for c in v['loc_world'])} "
              f"extent_m={tuple(round(c, 2) for c in v['extent'])} "
              f"yaw_rad={v['yaw_world']:.3f} (source={v['yaw_source']})")
    print(f"\n  raw vehicle-tagged entries : {n_vehicle_like}")
    print(f"  final kept vehicles        : {len(vehicles)}  <- this is what `build` will actually emit")


def cmd_inspect_poses(path):
    print(f"\n{'=' * 70}\nPOSES PATH: {path}\n{'=' * 70}")
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            depth = root[len(path):].count(os.sep)
            if depth >= 2:
                dirs[:] = []
                continue
            for fn in sorted(files)[:10]:
                fp = os.path.join(root, fn)
                print(f"  {fp}  ({os.path.getsize(fp)} bytes)")
    files_to_peek = [path] if os.path.isfile(path) else glob.glob(os.path.join(path, "*"))[:2]
    for fp in files_to_peek:
        if fp.endswith(".npy"):
            arr = np.load(fp)
            print(f"\n{fp}: npy shape={arr.shape} dtype={arr.dtype}\nfirst row: {arr.reshape(len(arr), -1)[0]}")
        elif os.path.isfile(fp):
            with open(fp) as f:
                head = [next(f) for _ in range(3)]
            print(f"\n{fp}: text, first lines:\n{''.join(head)}")


# ============================================================================
# Geometry
# ============================================================================

def yaw_from_rotation_matrix(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def world_to_lidar_transform(lidar_to_world_4x4):
    """lidar_to_world: 4x4, lidar frame -> world frame. Returns (R_wl, t_wl)
    such that p_lidar = R_wl @ p_world + t_wl."""
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
    """box = [x, y, z, l, w, h, yaw] (z-center convention). Returns 4x2 BEV corners."""
    x, y, _, l, w, _, yaw = box
    dx, dy = l / 2.0, w / 2.0
    # counter-clockwise winding -- required by the Sutherland-Hodgman clip below
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
            return (edge[0] * d[1] - edge[1] * d[0]) >= 0  # 2D cross product

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
    denom = d1[0] * d2[1] - d1[1] * d2[0]  # 2D cross product
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


# ============================================================================
# LiDAR point counting (for num_lidar_pts, only if your pkl schema has it)
# ============================================================================

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


# ============================================================================
# Pose loading -- pluggable, VERIFY against your real all_poses/ layout.
# ============================================================================

def load_town_seq_poses(poses_dir, seq_name):
    """Returns dict {frame_idx:int -> 4x4 lidar_to_world np.ndarray (meters)}
    or None if nothing recognizable was found. Tries a few common layouts;
    edit/extend this once `inspect-poses` shows you the real one."""
    candidates = [
        # CONFIRMED layout: all_poses/<seq>/poses/lidar.txt
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
            # CONFIRMED format: col 0 is the frame index, remaining 16 are a
            # row-major 4x4 lidar-to-world matrix (all_poses/<seq>/poses/lidar.txt)
            if ncol == 17:
                return {int(round(rows[i, 0])): rows[i, 1:].reshape(4, 4) for i in range(len(rows))}
            if ncol == 13:
                return {int(round(rows[i, 0])): np.vstack([rows[i, 1:].reshape(3, 4), [0, 0, 0, 1]])
                        for i in range(len(rows))}
            if ncol == 8:  # frame_idx + quat7
                return {int(round(rows[i, 0])): _pose7_to_matrix(rows[i, 1:]) for i in range(len(rows))}
            # no leading index column
            if ncol == 7:
                return {i: _pose7_to_matrix(rows[i]) for i in range(len(rows))}
            if ncol in (12, 16):
                return {i: rows[i].reshape(4, 4) if ncol == 16
                         else np.vstack([rows[i].reshape(3, 4), [0, 0, 0, 1]])
                         for i in range(len(rows))}
    return None


def _pose7_to_matrix(row):
    tx, ty, tz, qx, qy, qz, qw = row
    R = _quat_to_rot(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ============================================================================
# Build
# ============================================================================

def parse_token(token):
    m = TOKEN_RE.match(token)
    if not m:
        return None
    return m.group("town"), m.group("seq"), int(m.group("frame"))


def load_town_static_vehicles(scene_meshes_dir, town, mesh_fit_stats=None):
    """mesh_fit_stats: optional Counter to accumulate obb_yaw_and_extent_from_mesh
    failure/success reasons into, so a `build` run can report at the end
    whether the mesh fit is actually engaging at all (see the module
    docstring's NOTE on yaw)."""
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

        # Pass 1: collect raw candidates, keeping the ORIGINAL AABB
        # (bbox_min/bbox_max) untouched -- needed for the overlap-based
        # dedup below, which must run BEFORE any yaw/extent fitting so it
        # isn't confused by an already-rotated extent.
        raw = []
        for e in entries:
            cls = normalize_vehicle_class(get_actor_class(e))
            if cls is None:
                continue
            bbox_min = _get(e, "bbox_min", default=None)
            bbox_max = _get(e, "bbox_max", default=None)
            loc = get_actor_location_m(e)
            if bbox_min is None or bbox_max is None:
                # No raw AABB available (unexpected schema) -- synthesize
                # a degenerate point box so dedup still runs, just never
                # merges this entry with anything.
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

        # Pass 2: collapse same-actor sub-parts (glass, trim) into their
        # parent body BEFORE spending a mesh fit on parts we'll discard.
        raw = dedupe_same_actor_parts(raw)
        print(len(raw))
        # Pass 3: fit yaw/extent for each surviving candidate, preferring
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
        print("final veh", len(vehicles))
        return vehicles
        # Pass 4: final sanity filter, calibrated per extent source (see
        # plausible_vehicle_size docstring -- AABB-fallback extents get a
        # much looser check since they can be legitimately inflated).
        # return [v for v in vehicles if plausible_vehicle_size(v["extent"], v["yaw_source"])]
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
            parsed = parse_token(token)
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

            n_new = len(new_boxes)
            rec["gt_boxes"] = np.concatenate([rec["gt_boxes"], np.stack(new_boxes)], axis=0)
            rec["gt_names"] = np.concatenate([rec["gt_names"], np.array(new_names, dtype=rec["gt_names"].dtype)])
            if "gt_velocity" in rec:
                rec["gt_velocity"] = np.concatenate(
                    [rec["gt_velocity"], np.zeros((n_new, 2), dtype=np.float32)], axis=0)
            if "instance_ids" in rec:
                # actor_info.json instance_id is already unique per town; just offset it
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

        print(f"\n{pkl_path}: {sum(1 for r in infos)} records processed")
        if write:
            bak = pkl_path + ".bak"
            if not os.path.exists(bak):
                os.rename(pkl_path, bak)
                with open(pkl_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                with open(pkl_path, "wb") as f:
                    pickle.dump(data, f)
            print(f"  written (backup at {bak})")
        else:
            print("  DRY RUN -- nothing written (pass --write to commit)")

    print(f"\n{'=' * 70}\nMESH-OBB FIT DIAGNOSTIC (across all towns touched by this run)\n{'=' * 70}")
    if not mesh_fit_stats:
        print("  every vehicle entry had an explicit rotation field -- no mesh fit needed.")
    else:
        for reason, count in mesh_fit_stats.most_common():
            print(f"  {reason}: {count}")
        if mesh_fit_stats.get("ok", 0) == 0 and sum(mesh_fit_stats.values()) > 0:
            print("  *** mesh-OBB fit engaged ZERO times across every town -- every yaw/extent")
            print("      added below is the unreliable AABB-heuristic fallback. Do NOT --write")
            print("      until this says at least 'ok: <nonzero>' for the towns you care about.")

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


# ============================================================================
# Visual QA
# ============================================================================

def cmd_visualize(pkl_path, token, dataset_root, out_path):
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


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect-pkl")
    p.add_argument("path")

    p = sub.add_parser("inspect-actors")
    p.add_argument("path")

    p = sub.add_parser("inspect-poses")
    p.add_argument("path")

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
    if args.cmd == "inspect-pkl":
        cmd_inspect_pkl(args.path)
    elif args.cmd == "inspect-actors":
        cmd_inspect_actors(args.path)
    elif args.cmd == "inspect-poses":
        cmd_inspect_poses(args.path)
    elif args.cmd == "build":
        build(args.scene_meshes_dir, args.poses_dir, args.pkl, args.write, args.lidar_root)
    elif args.cmd == "visualize":
        cmd_visualize(args.pkl, args.token, args.dataset_root, args.out)


if __name__ == "__main__":
    sys.exit(main())
