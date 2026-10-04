from __future__ import annotations

import collections
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from leaderboard.state_estimation.sensor_dataclasses import trackedbb, Box, StaticVehicleRef, Ego3DState
import numpy as np
import carla
import leaderboard.state_estimation.hyperparams as hp


def wrap_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

def center_distance(a: Box, b: Box) -> float:
    return float(np.linalg.norm(a.translation[:2] - b.translation[:2]))

def scale_iou(a: Box, b: Box) -> float:
    min_size = np.minimum(a.size, b.size)
    intersection = float(np.prod(min_size))
    union = float(np.prod(a.size)) + float(np.prod(b.size)) - intersection
    return intersection / union if union > 0 else 0.0

def yaw_diff(a: Box, b: Box) -> float:
    diff = (a.yaw - b.yaw + math.pi) % (2 * math.pi) - math.pi
    return abs(float(diff))

def velocity_l2(a: Box, b: Box) -> float:
    if not np.all(np.isfinite(a.velocity)) or not np.all(np.isfinite(b.velocity)):
        return float("nan")
    return float(np.linalg.norm(a.velocity - b.velocity))

def greedy_match(gt_boxes: Sequence[Box], pred_boxes: Sequence[Box],
                 threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            d = center_distance(gt, pred)
            if d <= threshold:
                candidates.append((d, gi, pi))
    candidates.sort(key=lambda c: c[0])

    matched_gt, matched_pred = set(), set()
    pairs = []
    for d, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        pairs.append((gi, pi))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return pairs, unmatched_gt, unmatched_pred

def _filter_range(boxes: Sequence[Box], max_range: float) -> List[Box]:
    return [b for b in boxes if float(np.linalg.norm(b.translation[:2])) <= max_range]

def _is_visible(box: Box) -> bool:
    return bool(getattr(box, 'visible', True))

def evaluate_frames(gt_frames: Dict[str, List[Box]], pred_frames: Dict[str, List[Box]],
                    thresholds: Sequence[float] = hp.DIST_THRESHOLDS, tp_threshold: float = hp.TP_DISTANCE_THRESHOLD,
                    max_range: Optional[float] = None, by_tag: bool = True) -> dict:
    """
    Visibility-aware scoring. Every GT box is either lidar-visible or not.
      - A visible GT that is not matched is a MISS: it lowers recall.
      - An unseen GT (too few lidar points) that is not matched costs nothing.
      - An unseen GT that IS matched is a bonus: a true positive for precision, and it enters the
        recall numerator and denominator (so it can only raise recall, never lower it).
      - A prediction that matches no GT at all is a false positive: it lowers precision.
    recall    = matched / (visible GT + matched unseen GT)
    precision = matched / predictions
    """
    # tp_threshold has to be one of the evaluated thresholds, otherwise the TP errors are all NaN
    thresholds = sorted(set(float(t) for t in thresholds) | {float(tp_threshold)})
    all_frames = sorted(set(gt_frames) | set(pred_frames))
    per_threshold = {}
    tp_errors_by_tag: Dict[str, Dict[str, List[float]]] = {}
    overall_tp_errors: Dict[str, List[float]] = {"trans_err": [], "scale_err": [], "orient_err": [],
                                                 "orient_fold_err": [], "vel_err": []}

    for thresh in thresholds:
        n_gt_visible = n_pred = n_matched = n_matched_visible = n_matched_unseen = 0
        for frame_id in all_frames:
            gt = gt_frames.get(frame_id, [])
            pred = pred_frames.get(frame_id, [])
            if max_range is not None:
                gt = _filter_range(gt, max_range)
                pred = _filter_range(pred, max_range)

            pairs, _, _ = greedy_match(gt, pred, thresh)

            vis = [_is_visible(g) for g in gt]
            hits_visible = sum(1 for gi, _ in pairs if vis[gi])
            n_gt_visible += sum(vis)
            n_pred += len(pred)
            n_matched += len(pairs)
            n_matched_visible += hits_visible
            n_matched_unseen += len(pairs) - hits_visible

            if abs(thresh - tp_threshold) < 1e-9:
                for gi, pi in pairs:
                    g, p = gt[gi], pred[pi]
                    overall_tp_errors["trans_err"].append(center_distance(g, p))
                    overall_tp_errors["scale_err"].append(1.0 - scale_iou(g, p))
                    overall_tp_errors["orient_err"].append(yaw_diff(g, p))
                    overall_tp_errors["orient_fold_err"].append(min(yaw_diff(g, p), math.pi - yaw_diff(g, p)))
                    overall_tp_errors["vel_err"].append(velocity_l2(g, p))
                    if by_tag and g.tag:
                        d = tp_errors_by_tag.setdefault(g.tag, {"trans_err": [], "scale_err": [],
                                                                "orient_err": [], "vel_err": []})
                        d["trans_err"].append(center_distance(g, p))
                        d["scale_err"].append(1.0 - scale_iou(g, p))
                        d["orient_err"].append(yaw_diff(g, p))
                        d["vel_err"].append(velocity_l2(g, p))

        n_gt_eval = n_gt_visible + n_matched_unseen
        recall = n_matched / n_gt_eval if n_gt_eval else float("nan")
        precision = n_matched / n_pred if n_pred else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan"))

        recall_visible = n_matched_visible / n_gt_visible if n_gt_visible else float("nan")
        per_threshold[thresh] = {"num_gt_visible": n_gt_visible, "num_matched_unseen": n_matched_unseen,
                                 "num_pred": n_pred, "num_matched": n_matched,
                                 "recall": recall, "recall_visible": recall_visible,
                                 "precision": precision, "f1": f1}

    def _means(d): return {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in d.items()}

    return {
        "per_threshold": per_threshold,
        "tp_threshold": tp_threshold,
        "tp_errors": _means(overall_tp_errors),
        "tp_errors_by_tag": {tag: _means(d) for tag, d in tp_errors_by_tag.items()},
        "config": {"thresholds": list(thresholds), "max_range": max_range},
    }

def _apply_matrix(M: np.ndarray, xyz) -> np.ndarray:
    p = np.array([xyz[0], xyz[1], xyz[2], 1.0])
    return (M @ p)[:3]

def local_to_world(reference_transform: carla.Transform, local_point) -> np.ndarray:
    M = np.asarray(reference_transform.get_matrix(), dtype=float)
    return _apply_matrix(M, (local_point.x, local_point.y, local_point.z))

def world_to_local(reference_transform: carla.Transform, world_xyz) -> np.ndarray:
    M = np.asarray(reference_transform.get_inverse_matrix(), dtype=float)
    xyz = (world_xyz.x, world_xyz.y, world_xyz.z) if hasattr(world_xyz, "x") else world_xyz
    return _apply_matrix(M, xyz)

def _ego_transform(ego_state: Ego3DState) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=ego_state.x, y=ego_state.y, z=ego_state.z),
        carla.Rotation(
            pitch=math.degrees(ego_state.pitch_rad),
            yaw=math.degrees(ego_state.yaw_rad),
            roll=math.degrees(ego_state.roll_rad),
        ),
    )

def global_boxes_to_ego(boxes: Sequence[Box], ego_state: Ego3DState) -> List[Box]:
    ego_transform = _ego_transform(ego_state)
    ego_yaw = ego_state.yaw_rad
    cos_yaw, sin_yaw = math.cos(ego_yaw), math.sin(ego_yaw)
    converted = []
    for box in boxes:
        velocity = box.velocity
        velocity_ego = np.array([
            cos_yaw * velocity[0] + sin_yaw * velocity[1],
            -sin_yaw * velocity[0] + cos_yaw * velocity[1],
        ])
        new_box = Box(
            translation=world_to_local(ego_transform, box.translation),
            size=box.size.copy(),
            yaw=wrap_angle_rad(box.yaw - ego_yaw),
            velocity=velocity_ego,
            tag=box.tag,
            frame_id=box.frame_id,
        )
        for extra in ('score', 'det_speed', 'n_radar'):
            setattr(new_box, extra, getattr(box, extra, float('nan')))
        converted.append(new_box)
    return converted


class CarlaGroundTruthExtractor:
    """
        Ground truth built once from the map, then read directly every frame
    """
    def __init__(self, world: carla.World, ego_actor_id: int,
                 static_max_range: float = 50.0, actor_max_range: float = 50.0,
                 min_points_for_visibility: int = 5, visibility_margin: float = 0.2):
        self.world = world
        self.ego_actor_id = ego_actor_id
        self.static_max_range = static_max_range
        self.actor_max_range = actor_max_range
        self.min_points_for_visibility = min_points_for_visibility
        self.visibility_margin = visibility_margin

        specific_tag_by_id: Dict[int, str] = {}
        for tag_id, name in hp.STATIC_VEHICLE_SUBTAGS.items():
            for obj in world.get_environment_objects(carla.CityObjectLabel(tag_id)):
                specific_tag_by_id[obj.id] = name
        refs = []
        env_vehicle_objs = []
        for label in hp.VEHICLE_LABELS:
            env_vehicle_objs.extend(world.get_environment_objects(label))
        for obj in env_vehicle_objs:
            bbox = obj.bounding_box
            refs.append(StaticVehicleRef(
                env_id=obj.id, tag=specific_tag_by_id.get(obj.id, "vehicle"),
                center_world=np.array([bbox.location.x, bbox.location.y, bbox.location.z]),
                size=np.array([2.0 * bbox.extent.x, 2.0 * bbox.extent.y, 2.0 * bbox.extent.z]),
                yaw_world=math.radians(bbox.rotation.yaw)))
        self.static_refs: List[StaticVehicleRef] = refs

    def _get_points_in_box(self, box: Box, lidar_points_ego: Optional[np.ndarray]) -> int:
        if lidar_points_ego is None:
            # no lidar available: do not fake invisibility, treat every box as visible
            return self.min_points_for_visibility
        if lidar_points_ego.size == 0:
            return 0
        rel = lidar_points_ego[:, :3] - box.translation
        c, s = math.cos(-box.yaw), math.sin(-box.yaw)
        x_local = c * rel[:, 0] - s * rel[:, 1]
        y_local = s * rel[:, 0] + c * rel[:, 1]
        z_local = rel[:, 2]
        l, w, h = box.size
        inside = ((np.abs(x_local) <= l / 2 + self.visibility_margin) &
                  (np.abs(y_local) <= w / 2 + self.visibility_margin) &
                  (np.abs(z_local) <= h / 2 + self.visibility_margin))
        return int(np.count_nonzero(inside))

    def _actor_boxes(self, ego_transform: carla.Transform, frame_id: str, lidar_points_ego: Optional[np.ndarray]) -> List[Box]:
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
        boxes = []
        all_actors = self.world.get_actors()
        candidates = list(all_actors.filter("vehicle.*")) + list(all_actors.filter("walker.pedestrian.*"))

        def _tag_from_type_id(type_id: str) -> str:
            t = type_id.lower()
            if t.startswith("walker."): return "pedestrian"
            for keyword in ("truck", "bus", "motorcycle", "bicycle", "rider", "train"):
                if keyword in t: return keyword
            if t.startswith("vehicle."): return "car"
            return t

        def box_from_actor(actor: carla.Actor, tag: str = "") -> Box:
            actor_transform = actor.get_transform()
            bbox = actor.bounding_box
            center_world = local_to_world(actor_transform, bbox.location)
            center_local = world_to_local(ego_transform, center_world)
            size = np.array([2.0 * bbox.extent.x, 2.0 * bbox.extent.y, 2.0 * bbox.extent.z])
            yaw_world = math.radians(actor_transform.rotation.yaw + bbox.rotation.yaw)
            ego_yaw = math.radians(ego_transform.rotation.yaw)
            yaw_local = wrap_angle_rad(yaw_world - ego_yaw)
            vel_world = actor.get_velocity()
            c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
            vx = c * vel_world.x - s * vel_world.y
            vy = s * vel_world.x + c * vel_world.y
            return Box.from_geometry(center_local[0], center_local[1], center_local[2],
                                     size[0], size[1], size[2], yaw_local, vx=vx, vy=vy, tag=tag)

        for actor in candidates:
            if self.ego_actor_id is not None and actor.id == self.ego_actor_id:
                continue
            loc = actor.get_transform().location
            if np.linalg.norm([loc.x - ego_pos[0], loc.y - ego_pos[1]]) > self.actor_max_range:
                continue
            box = box_from_actor(actor, tag=_tag_from_type_id(actor.type_id))
            box.frame_id = frame_id

            # occluded / sparse boxes are tagged, not deleted: they are never penalised as misses,
            # but they earn credit when the tracker finds them anyway
            is_visible = self._get_points_in_box(box, lidar_points_ego) >= self.min_points_for_visibility
            setattr(box, 'visible', is_visible)
            boxes.append(box)

        return boxes

    def _static_boxes(self, ego_transform: carla.Transform, frame_id: str, lidar_points_ego: Optional[np.ndarray]) -> List[Box]:
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
        nearby = [r for r in self.static_refs
                  if np.linalg.norm(r.center_world[:2] - ego_pos) <= self.static_max_range]

        def _box_from_static_ref(ref: StaticVehicleRef, ego_transform: carla.Transform, frame_id: str) -> Box:
            center_local = world_to_local(ego_transform, ref.center_world)
            ego_yaw = math.radians(ego_transform.rotation.yaw)
            yaw_local = wrap_angle_rad(ref.yaw_world - ego_yaw)
            return Box.from_geometry(center_local[0], center_local[1], center_local[2],
                                      ref.size[0], ref.size[1], ref.size[2], yaw_local,
                                      vx=0.0, vy=0.0, tag=ref.tag, frame_id=frame_id)

        boxes = []
        for r in nearby:
            box = _box_from_static_ref(r, ego_transform, frame_id=frame_id)
            is_visible = self._get_points_in_box(box, lidar_points_ego) >= self.min_points_for_visibility
            setattr(box, 'visible', is_visible)
            boxes.append(box)

        return boxes

    def update(self, ego_transform: carla.Transform, frame_id: str,
                lidar_points_ego: Optional[np.ndarray]) -> List[Box]:
        return (self._actor_boxes(ego_transform, frame_id, lidar_points_ego) +
                 self._static_boxes(ego_transform, frame_id, lidar_points_ego))

def print_results(metrics: dict) -> None:
    print()
    print("=" * 88)
    print("CLASS-AGNOSTIC BOX PRESENCE / ERROR EVALUATION  (misses only penalised for lidar-visible GT)")
    print("=" * 88)
    print(f"{'threshold (m)':>14}{'GT(Vis)':>10}{'+Unseen':>9}{'num_pred':>10}{'matched':>10}"
          f"{'recall':>10}{'rec(vis)':>10}{'precision':>10}{'F1':>8}")
    for thresh, m in metrics["per_threshold"].items():
        print(f"{thresh:>14.1f}{m['num_gt_visible']:>10}{m['num_matched_unseen']:>9}{m['num_pred']:>10}"
              f"{m['num_matched']:>10}{m['recall']:>10.3f}{m['recall_visible']:>10.3f}"
              f"{m['precision']:>10.3f}{m['f1']:>8.3f}")
    print("  +Unseen = matched GT with too few lidar points: counted as hits, never as misses")
    print("  recall = matched/(visible+matched unseen) is bonus-adjusted; rec(vis) = visible GT found / visible GT")
    print()
    print(f"TP errors at {metrics['tp_threshold']} m match threshold (matched pairs only):")
    e = metrics["tp_errors"]
    print(f"  ATE (m)     : {e['trans_err']:.3f}")
    print(f"  ASE (1-IoU) : {e['scale_err']:.3f}")
    print(f"  AOE (rad)   : {e['orient_err']:.3f}   (180-degree flip-folded: {e['orient_fold_err']:.3f})")
    print(f"  AVE (m/s)   : {e['vel_err']:.3f}")
    if metrics["tp_errors_by_tag"]:
        print("\n  by GT tag:")
        for tag, e in metrics["tp_errors_by_tag"].items():
            print(f"    {tag:<12} ATE={e['trans_err']:.3f}  ASE={e['scale_err']:.3f}  "
                  f"AOE={e['orient_err']:.3f}  AVE={e['vel_err']:.3f}")
    print("=" * 88)

def save_metrics(metrics: dict, path: Path) -> None:
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(sanitize(metrics), indent=2))

def _fmean(values) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

def _fmedian(values) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")

def diagnose_frames(gt_frames: Dict[str, List[Box]], pred_frames: Dict[str, List[Box]],
                    ego_speed: Dict[str, float], thresh: float = hp.TP_DISTANCE_THRESHOLD,
                    max_range: float = 50.0, range_edges: Sequence[float] = (0.0, 15.0, 30.0, 50.0),
                    static_speed: float = 0.2, score_edges: Sequence[float] = (0.2, 0.3, 0.4, 0.5),
                    ego_pose: Optional[Dict[str, Tuple[float, float, float]]] = None,
                    ghost_ctx: Optional[Dict[tuple, Tuple[str, str]]] = None) -> dict:
    """Where do the errors come from? Range, false-positive anatomy, precision vs score, heading flips, velocity."""
    inner = list(range_edges[1:-1])
    nb = len(range_edges) - 1
    score_edges = list(score_edges)

    def bucket(box: Box) -> int:
        return int(np.searchsorted(inner, float(np.linalg.norm(box.translation[:2])), side="right"))

    rng = [dict(vis=0, hit_vis=0, unseen=0, hit_unseen=0, pred=0, fp=0) for _ in range(nb)]
    sc = [dict(n=0, fp=0, ghost=0) for _ in range(len(score_edges) + 1)]
    fp_kind = {"duplicate": 0, "near_miss": 0, "ghost": 0}
    ghost_anatomy: Dict[Tuple[str, str], int] = {}
    ghost_rows: List[tuple] = []      # (frame, x_ego, y_ego, x_world, y_world, l, w, h, speed, score, range)
    # matched pairs: (vel_err, pred_speed, ego_speed, gt_speed, angle_deg, is_pedestrian, pvx, pvy, gvx, gvy, det_speed, n_radar)
    vel = {"static": [], "moving": []}
    ori = {"static": [], "moving": []}      # yaw error, non-pedestrians only

    for fid in sorted(set(gt_frames) | set(pred_frames)):
        gt = _filter_range(gt_frames.get(fid, []), max_range)
        pred = _filter_range(pred_frames.get(fid, []), max_range)
        pairs, _, un_pred = greedy_match(gt, pred, thresh)
        matched_gt = {gi for gi, _ in pairs}
        unmatched_pred = set(un_pred)
        for gi, g in enumerate(gt):
            r = rng[bucket(g)]
            if _is_visible(g):
                r["vis"] += 1
                r["hit_vis"] += int(gi in matched_gt)
            else:
                r["unseen"] += 1
                r["hit_unseen"] += int(gi in matched_gt)
        for pi, p in enumerate(pred):
            b = bucket(p)
            rng[b]["pred"] += 1
            s_ = float(getattr(p, "score", float("nan")))
            sb = int(np.searchsorted(score_edges, s_, side="right")) if np.isfinite(s_) else None
            if sb is not None:
                sc[sb]["n"] += 1
            if pi not in unmatched_pred:
                continue
            rng[b]["fp"] += 1
            d_matched = min((center_distance(gt[gi], p) for gi in matched_gt), default=math.inf)
            d_any = min((center_distance(g, p) for g in gt), default=math.inf)
            kind = "duplicate" if d_matched <= 2 * thresh else ("near_miss" if d_any <= 2 * thresh else "ghost")
            fp_kind[kind] += 1
            if sb is not None:
                sc[sb]["fp"] += 1
                sc[sb]["ghost"] += int(kind == "ghost")
            if kind == "ghost":
                size_cls = "ped-sized (<1.2 m)" if max(p.size[0], p.size[1]) < 1.2 else "vehicle-sized"
                motion = "moving" if float(np.linalg.norm(p.velocity)) >= 0.5 else "static"
                ghost_anatomy[(size_cls, motion)] = ghost_anatomy.get((size_cls, motion), 0) + 1
                pose = ego_pose.get(fid) if ego_pose else None
                if pose is not None:
                    ex0, ey0, eyaw = pose
                    cy_, sy_ = math.cos(eyaw), math.sin(eyaw)
                    wx = ex0 + cy_ * float(p.translation[0]) - sy_ * float(p.translation[1])
                    wy = ey0 + sy_ * float(p.translation[0]) + cy_ * float(p.translation[1])
                else:
                    wx = wy = float("nan")
                ctx = ("", "")
                if ghost_ctx:
                    ctx = ghost_ctx.get((fid, round(float(p.translation[0]), 3), round(float(p.translation[1]), 3)), ("", ""))
                ghost_rows.append((fid, float(p.translation[0]), float(p.translation[1]), wx, wy,
                                   float(p.size[0]), float(p.size[1]), float(p.size[2]),
                                   float(np.linalg.norm(p.velocity)), s_,
                                   float(np.linalg.norm(p.translation[:2])), ctx[0], ctx[1]))
        es = ego_speed.get(fid, float("nan"))
        for gi, pi in pairs:
            g, p = gt[gi], pred[pi]
            gs = float(np.linalg.norm(g.velocity))
            ps = float(np.linalg.norm(p.velocity))
            kind = "static" if gs < static_speed else "moving"
            ang = float("nan")
            if gs > 0.5 and ps > 0.5 and np.all(np.isfinite(p.velocity)):
                c = float(np.dot(g.velocity, p.velocity) / (gs * ps))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
            vel[kind].append((velocity_l2(g, p), ps, es, gs, ang, 1.0 if g.tag == "pedestrian" else 0.0,
                              float(p.velocity[0]), float(p.velocity[1]), float(g.velocity[0]), float(g.velocity[1]),
                              float(getattr(p, "det_speed", float("nan"))), float(getattr(p, "n_radar", float("nan")))))
            if g.tag != "pedestrian":
                ori[kind].append(yaw_diff(g, p))

    return {"thresh": thresh, "ranges": [(range_edges[i], range_edges[i + 1]) for i in range(nb)],
            "range_stats": rng, "score_edges": score_edges, "score_stats": sc, "fp_kind": fp_kind,
            "ghost_anatomy": ghost_anatomy, "ghost_rows": ghost_rows, "vel": vel, "ori": ori}

def print_diagnostics(d: dict) -> None:
    nan = float("nan")
    print(f"\nDIAGNOSTICS @ {d['thresh']} m match threshold")
    print(f"{'range (m)':>12}{'visGT':>8}{'rec(vis)':>10}{'unseenGT':>10}{'unseen-hit':>12}{'preds':>8}{'precision':>11}")
    for (lo, hi), r in zip(d["ranges"], d["range_stats"]):
        vr = r["hit_vis"] / r["vis"] if r["vis"] else nan
        uh = r["hit_unseen"] / r["unseen"] if r["unseen"] else nan
        pr = 1.0 - r["fp"] / r["pred"] if r["pred"] else nan
        print(f"{lo:>5.0f}-{hi:<6.0f}{r['vis']:>8}{vr:>10.3f}{r['unseen']:>10}{uh:>12.3f}{r['pred']:>8}{pr:>11.3f}")

    tot = sum(d["fp_kind"].values())
    if tot:
        k = d["fp_kind"]
        print(f"\nFalse positives: {tot}  |  duplicate/trailing {100*k['duplicate']/tot:.0f}%  "
              f"(near an already-matched GT)  |  near-miss {100*k['near_miss']/tot:.0f}%  "
              f"(<= 2x thresh from an unmatched GT)  |  ghost {100*k['ghost']/tot:.0f}%  (no GT nearby)")
    if d["ghost_anatomy"]:
        print("  ghosts by type: " + "  |  ".join(f"{sz}, {mo}: {n}" for (sz, mo), n in sorted(d["ghost_anatomy"].items())))

    rows = d["ghost_rows"]
    if rows:
        ranges = np.array([r[10] for r in rows])
        edges_r = [0.0, 3.0, 6.0, 10.0, 15.0, 30.0, 50.0001]
        hist, _ = np.histogram(ranges, bins=edges_r)
        print("\nGhost range histogram (ego frame): " + "  ".join(
            f"{edges_r[i]:.0f}-{min(edges_r[i+1], 50):.0f}m: {int(hist[i])} ({100*hist[i]/len(rows):.0f}%)"
            for i in range(len(hist))))
        print(f"  ghosts within 3 m of the ego origin (ego body / sensor artefacts): {int(hist[0])}")
        fin = [r for r in rows if np.isfinite(r[3])]
        if fin:
            cells: Dict[Tuple[int, int], List[tuple]] = {}
            for r in fin:
                cells.setdefault((int(round(r[3] / 2.0)), int(round(r[4] / 2.0))), []).append(r)
            ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))
            top10 = sum(len(v) for _, v in ranked[:10])
            print(f"  ghosts in the 10 busiest 2 m WORLD cells: {100*top10/len(fin):.0f}% of all ghosts "
                  f"(high = fixed scene objects missing from GT; low = scattered hallucinations)")
            print(f"  {'world x':>9}{'world y':>9}{'count':>7}{'frames':>8}{'mean l x w':>14}{'mean speed':>12}")
            for (cx, cy), v in ranked[:6]:
                print(f"  {cx*2.0:>9.1f}{cy*2.0:>9.1f}{len(v):>7}{len({r[0] for r in v}):>8}"
                      f"{np.mean([r[5] for r in v]):>8.1f}x{np.mean([r[6] for r in v]):<4.1f}{np.mean([r[8] for r in v]):>12.2f}")

    if rows and any(r[11] for r in rows):
        probed = [r for r in rows if r[11]]
        nothing = [r for r in probed if r[11] == "none" and r[12] == "none"]
        env_c = collections.Counter(r[11].split(":")[0] for r in probed if r[11] != "none")
        act_c = collections.Counter(r[12].split("@")[0] for r in probed if r[12] != "none")
        print(f"\nGhost context, probed in the LIVE world at report time ({len(probed)} ghosts probed):")
        print(f"  nothing within the probe radius (hallucination-like): {len(nothing)} ({100*len(nothing)/len(probed):.0f}%)")
        if env_c:
            print("  environment-object labels nearby: " + ", ".join(f"{k}: {v}" for k, v in env_c.most_common(6)))
        if act_c:
            print("  actor types nearby:               " + ", ".join(f"{k}: {v}" for k, v in act_c.most_common(6)))

    edges = d["score_edges"]
    if any(s["n"] for s in d["score_stats"]):
        labels = [f"<{edges[0]:.2f}"] + [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)] + [f">={edges[-1]:.2f}"]
        print(f"\nPrecision by detector score of the track's last matched detection")
        print(f"{'score':>12}{'preds':>9}{'precision':>11}{'ghost share':>13}")
        for lab, s in zip(labels, d["score_stats"]):
            if s["n"]:
                print(f"{lab:>12}{s['n']:>9}{1 - s['fp']/s['n']:>11.3f}{s['ghost']/s['n']:>13.3f}")
    else:
        print("\n(no detector scores on the tracker output: trackedbb rejected the extra 'score' attribute)")

    print("\nOrientation (non-pedestrian matches)       n   AOE raw  AOE flip-folded  flipped>120deg")
    for kind in ("static", "moving"):
        a = np.asarray(d["ori"][kind], dtype=float)
        if a.size:
            print(f"  GT {kind:<8}{'':>26}{a.size:>5}{a.mean():>10.3f}{np.minimum(a, math.pi - a).mean():>17.3f}"
                  f"{100*np.mean(a > 2*math.pi/3):>15.1f}%")

    print("\nVelocity (matched pairs)")
    a = np.asarray(d["vel"]["static"], dtype=float).reshape(-1, 12)
    if a.size:
        print(f"  GT static: n={len(a)}  AVE={_fmean(a[:, 0]):.2f}  mean pred speed={_fmean(a[:, 1]):.2f}")
        for lo, hi in ((0.0, 0.5), (0.5, 3.0), (3.0, math.inf)):
            m = (a[:, 2] >= lo) & (a[:, 2] < hi)
            if m.any():
                print(f"      ego speed {lo:>3.1f}-{hi:<4.1f} m/s: n={int(m.sum()):>6}  AVE={_fmean(a[m, 0]):.2f}")
    a = np.asarray(d["vel"]["moving"], dtype=float).reshape(-1, 12)
    if a.size:
        print(f"  GT moving:  {'group':<22}{'n':>7}{'AVE':>7}{'GT spd':>8}{'track spd':>10}{'raw det spd':>12}{'radar pts':>10}{'med angle err':>15}{'reversed>120':>14}")
        veh = a[:, 5] == 0
        groups = (("pedestrians", a[:, 5] == 1),
                  ("vehicles 0.2-2 m/s", veh & (a[:, 3] < 2)),
                  ("vehicles 2-6 m/s", veh & (a[:, 3] >= 2) & (a[:, 3] < 6)),
                  ("vehicles >6 m/s", veh & (a[:, 3] >= 6)))
        for name, m in groups:
            if m.any():
                sub = a[m]
                ang = sub[:, 4][np.isfinite(sub[:, 4])]
                rev = 100 * float(np.mean(ang > 120)) if ang.size else nan
                print(f"              {name:<22}{len(sub):>7}{_fmean(sub[:, 0]):>7.2f}{_fmean(sub[:, 3]):>8.2f}"
                      f"{_fmean(sub[:, 1]):>10.2f}{_fmean(sub[:, 10]):>12.2f}{_fmean(sub[:, 11]):>10.1f}"
                      f"{_fmedian(ang):>14.1f}d{rev:>13.1f}%")
        v = a[(a[:, 5] == 0) & np.isfinite(a[:, 6]) & np.isfinite(a[:, 7])]
        if len(v):
            pvx, pvy, gvx, gvy, es_ = v[:, 6], v[:, 7], v[:, 8], v[:, 9], v[:, 2]
            variants = (("as is", pvx, pvy), ("vy negated", pvx, -pvy), ("vx negated", -pvx, pvy),
                        ("+ ego speed on x", pvx + es_, pvy), ("- ego speed on x", pvx - es_, pvy),
                        ("predict zero", 0 * pvx, 0 * pvy))
            print("  Velocity convention test, moving vehicles: mean |v_variant - v_GT| in m/s (a variant far below 'as is' = convention bug)")
            print("    " + "   ".join(f"{n}: {_fmean(np.hypot(x - gvx, y - gvy)):.2f}" for n, x, y in variants))

class OnlineEvaluator:
    def __init__(self, world: carla.World, ego_actor_id: int, thresholds: Sequence[float] = hp.DIST_THRESHOLDS,
                 tp_threshold: float = hp.TP_DISTANCE_THRESHOLD, max_range: float = 50, warmup_frames: int = 40,
                 ghost_csv: Optional[str] = "bbs_ghosts.csv", stationary_stride: int = 1,
                 probe_ghosts: bool = True, probe_radius: float = 3.0, probe_refresh: int = 50):
        self.thresholds = list(thresholds)
        self.tp_threshold = tp_threshold
        self.max_range = max_range
        self.warmup_frames = warmup_frames      # EKF convergence / radar start / track confirmation
        self._seen = 0
        self.stationary_stride = stationary_stride   # >1: keep only every Nth frame while the ego is stopped
        self._stopped = 0
        self._ego_speed: Dict[str, float] = {}
        self._ego_pose: Dict[str, Tuple[float, float, float]] = {}
        # env var wins, so the path can be set from `docker run -e BBS_GHOST_CSV=/some/mounted/dir/bbs_ghosts.csv`
        self.ghost_csv = os.environ.get("BBS_GHOST_CSV") or ghost_csv
        self._ghost_path_announced = False
        self.probe_ghosts = probe_ghosts            # ask the live CARLA world what stands under each ghost
        self.probe_radius = probe_radius
        self.probe_refresh = probe_refresh          # frames between environment-object cache refreshes
        self._probe_calls = 0
        self._env_xy = np.zeros((0, 2))
        self._env_info: List[Tuple[str, str]] = []
        self._ghost_ctx: Dict[tuple, Tuple[str, str]] = {}
        self._gt_frames: Dict[str, List[Box]] = {}
        self._pred_frames: Dict[str, List[Box]] = {}
        self.world = world
        self.ego_actor_id = ego_actor_id
        self.gt_extractor: CarlaGroundTruthExtractor = CarlaGroundTruthExtractor(world, ego_actor_id)

        self.ego_actor = self.world.get_actor(self.ego_actor_id)
        if self.ego_actor is None:
            raise ValueError(f"Ego actor {self.ego_actor_id} not found in world.")

        self.lidar_actor = self._find_lidar()

    def _find_lidar(self):
        # the sensors may be spawned after this evaluator is built, so this is retried in add_frame
        for actor in self.world.get_actors().filter('sensor.lidar.ray_cast'):
            if actor.parent and actor.parent.id == self.ego_actor_id:
                return actor
        return None

    def add_frame(self, frame_id: str, ego_state: Ego3DState, pts_sensor: np.ndarray, track_bbs: List[trackedbb], tracker_is_right_handed: bool = False):
        self._seen += 1
        if self._seen <= self.warmup_frames:
            return
        spd = float(getattr(ego_state, "speed", float("nan")))
        if self.stationary_stride > 1 and np.isfinite(spd) and spd < 0.1:
            self._stopped += 1
            if self._stopped % self.stationary_stride:
                return
        self._ego_speed[frame_id] = spd

        if self.lidar_actor is None:
            self.lidar_actor = self._find_lidar()
            if self.lidar_actor is None:
                print("WARNING: ego lidar actor not found, treating all GT as visible")

        gt_ego_transform = self.ego_actor.get_transform()
        self._ego_pose[frame_id] = (float(gt_ego_transform.location.x), float(gt_ego_transform.location.y),
                                    math.radians(float(gt_ego_transform.rotation.yaw)))
        if self.lidar_actor:
            lidar_transform = self.lidar_actor.get_transform()
            world_to_ego = np.asarray(gt_ego_transform.get_inverse_matrix(), dtype=float)
            sensor_to_world = np.asarray(lidar_transform.get_matrix(), dtype=float)
            sensor_to_ego_matrix = world_to_ego @ sensor_to_world
            pts_hom = np.ones((pts_sensor.shape[0], 4))
            pts_hom[:, :3] = pts_sensor[:, :3]
            lidar_points_ego = (sensor_to_ego_matrix @ pts_hom.T).T[:, :3]
        else:
            lidar_points_ego = None

        self._gt_frames[frame_id] = list(self.gt_extractor.update(gt_ego_transform, frame_id, lidar_points_ego))

        pred_boxes : list[Box] = []
        for pb in track_bbs:
            box = Box.from_trackedbb(pb, frame_id=frame_id)
            for extra in ('score', 'det_speed', 'n_radar'):
                setattr(box, extra, float(getattr(pb, extra, float('nan'))))

            if tracker_is_right_handed:
                box.translation[1] = -box.translation[1]
                box.yaw = -box.yaw
                if np.all(np.isfinite(box.velocity)):
                    box.velocity[1] = -box.velocity[1]

            pred_boxes.append(box)
        pred_boxes = global_boxes_to_ego(pred_boxes, ego_state)
        self._pred_frames[frame_id] = pred_boxes

        gt_boxes = self._gt_frames[frame_id]
        if self.probe_ghosts:
            try:
                self._probe(frame_id, pred_boxes, gt_boxes, gt_ego_transform)
            except Exception as exc:
                print(f"[evaluator] ghost probe disabled: {exc}")
                self.probe_ghosts = False
        gt_total = len(gt_boxes)
        gt_visible = sum(1 for b in gt_boxes if _is_visible(b))
        pred_in_range = sum(1 for b in pred_boxes if float(np.linalg.norm(b.translation[:2])) <= self.max_range)
        print(f"Frame {frame_id:>5} | GT: {gt_visible:>2} visible ({gt_total:>2} total) | Preds in range: {pred_in_range:>2}")

    _PROBE_SKIP_LABELS = {"Roads", "RoadLines", "Sidewalks", "Terrain", "Ground", "Sky", "Water"}

    def _refresh_env_cache(self) -> None:
        xy, info = [], []
        for o in self.world.get_environment_objects(carla.CityObjectLabel.Any):
            label = str(o.type).split(".")[-1]
            if label in self._PROBE_SKIP_LABELS:
                continue
            loc = o.bounding_box.location
            xy.append((loc.x, loc.y))
            info.append((label, o.name))
        self._env_xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        self._env_info = info

    def _probe(self, frame_id: str, pred_boxes: List[Box], gt_boxes: List[Box], ego_tf: carla.Transform) -> None:
        """Ghost = prediction with no GT box of any kind within 2x the TP threshold. Ask the live world what is there."""
        ghosts = []
        for b in pred_boxes:
            if float(np.linalg.norm(b.translation[:2])) > self.max_range:
                continue
            if min((center_distance(g, b) for g in gt_boxes), default=math.inf) > 2 * self.tp_threshold:
                ghosts.append(b)
        if not ghosts:
            return
        if self._probe_calls % self.probe_refresh == 0:
            self._refresh_env_cache()
        self._probe_calls += 1

        all_actors = self.world.get_actors()
        actors = []
        for a in list(all_actors.filter("vehicle.*")) + list(all_actors.filter("static.*")) + list(all_actors.filter("walker.*")):
            if a.id == self.ego_actor_id:
                continue
            loc = a.get_location()
            actors.append((loc.x, loc.y, a.type_id))
        act_xy = np.asarray([(x, y) for x, y, _ in actors], dtype=float).reshape(-1, 2)

        c, s = math.cos(math.radians(ego_tf.rotation.yaw)), math.sin(math.radians(ego_tf.rotation.yaw))
        for b in ghosts:
            wx = ego_tf.location.x + c * float(b.translation[0]) - s * float(b.translation[1])
            wy = ego_tf.location.y + s * float(b.translation[0]) + c * float(b.translation[1])
            env_txt, act_txt = "none", "none"
            if len(self._env_xy):
                d = np.hypot(self._env_xy[:, 0] - wx, self._env_xy[:, 1] - wy)
                i = int(np.argmin(d))
                if d[i] <= self.probe_radius:
                    env_txt = f"{self._env_info[i][0]}:{self._env_info[i][1][:30]}@{d[i]:.1f}".replace(",", ";")
            if len(act_xy):
                d = np.hypot(act_xy[:, 0] - wx, act_xy[:, 1] - wy)
                i = int(np.argmin(d))
                if d[i] <= self.probe_radius:
                    act_txt = f"{actors[i][2]}@{d[i]:.1f}"
            self._ghost_ctx[(frame_id, round(float(b.translation[0]), 3), round(float(b.translation[1]), 3))] = (env_txt, act_txt)

    def reset(self) -> None:
        self._gt_frames.clear()
        self._pred_frames.clear()
        self._ego_speed.clear()
        self._ego_pose.clear()
        self._ghost_ctx.clear()
        self._stopped = 0
        self._seen = 0

    @property
    def num_frames(self) -> int:
        return len(set(self._gt_frames) | set(self._pred_frames))

    def evaluate(self) -> dict:
        return evaluate_frames(self._gt_frames, self._pred_frames, self.thresholds, self.tp_threshold, self.max_range)

    def diagnose(self) -> dict:
        return diagnose_frames(self._gt_frames, self._pred_frames, self._ego_speed, self.tp_threshold,
                               self.max_range, ego_pose=self._ego_pose, ghost_ctx=self._ghost_ctx)

    def save_ghosts_csv(self, rows: list, path: "str | Path") -> None:
        with open(path, "w") as f:
            f.write("frame,x_ego,y_ego,x_world,y_world,length,width,height,speed,score,range,ctx_env,ctx_actor\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")

    def print_report(self) -> None:
        print_results(self.evaluate())
        d = self.diagnose()
        print_diagnostics(d)
        if self.ghost_csv:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.ghost_csv)), exist_ok=True)
                self.save_ghosts_csv(d["ghost_rows"], self.ghost_csv)
                if not self._ghost_path_announced:
                    print(f"[evaluator] ghost CSV is written to {os.path.abspath(self.ghost_csv)} "
                          f"(must be inside a mounted volume to survive the container)")
                    self._ghost_path_announced = True
            except Exception as exc:
                print(f"[evaluator] could not write ghost CSV to {self.ghost_csv}: {exc}")

    def save_metrics_json(self, filename: "str | Path") -> None:
        save_metrics(self.evaluate(), Path(filename))