from __future__ import annotations

import json
import math
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

def evaluate_frames(gt_frames: Dict[str, List[Box]], pred_frames: Dict[str, List[Box]],
                    thresholds: Sequence[float] = hp.DIST_THRESHOLDS, tp_threshold: float = hp.TP_DISTANCE_THRESHOLD,
                    max_range: Optional[float] = None, by_tag: bool = True) -> dict:
    all_frames = sorted(set(gt_frames) | set(pred_frames))
    per_threshold = {}
    tp_errors_by_tag: Dict[str, Dict[str, List[float]]] = {}
    overall_tp_errors: Dict[str, List[float]] = {"trans_err": [], "scale_err": [], "orient_err": [], "vel_err": []}

    for thresh in thresholds:
        n_gt = n_pred = n_matched = 0
        for frame_id in all_frames:
            gt = gt_frames.get(frame_id, [])
            pred = pred_frames.get(frame_id, [])
            if max_range is not None:
                gt = _filter_range(gt, max_range)
                #pred = _filter_range(pred, max_range)
            pairs, _, _ = greedy_match(gt, pred, thresh)
            n_gt += len(gt)
            n_pred += len(pred)
            n_matched += len(pairs)

            if abs(thresh - tp_threshold) < 1e-9:
                for gi, pi in pairs:
                    g, p = gt[gi], pred[pi]
                    overall_tp_errors["trans_err"].append(center_distance(g, p))
                    overall_tp_errors["scale_err"].append(1.0 - scale_iou(g, p))
                    overall_tp_errors["orient_err"].append(yaw_diff(g, p))
                    overall_tp_errors["vel_err"].append(velocity_l2(g, p))
                    if by_tag and g.tag:
                        d = tp_errors_by_tag.setdefault(g.tag, {"trans_err": [], "scale_err": [],
                                                                "orient_err": [], "vel_err": []})
                        d["trans_err"].append(center_distance(g, p))
                        d["scale_err"].append(1.0 - scale_iou(g, p))
                        d["orient_err"].append(yaw_diff(g, p))
                        d["vel_err"].append(velocity_l2(g, p))

        recall = n_matched / n_gt if n_gt else float("nan")
        precision = n_matched / n_pred if n_pred else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan"))
        per_threshold[thresh] = {"num_gt": n_gt, "num_pred": n_pred, "num_matched": n_matched,
                                 "recall": recall, "precision": precision, "f1": f1}

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
        converted.append(Box(
            translation=world_to_local(ego_transform, box.translation),
            size=box.size.copy(),
            yaw=wrap_angle_rad(box.yaw - ego_yaw),
            velocity=velocity_ego,
            tag=box.tag,
            frame_id=box.frame_id,
        ))
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

    def _is_visible(self, box: Box, lidar_points_ego: np.ndarray) -> bool:
        if lidar_points_ego.size == 0:
            return False
        rel = lidar_points_ego[:, :3] - box.translation
        c, s = math.cos(-box.yaw), math.sin(-box.yaw)
        x_local = c * rel[:, 0] - s * rel[:, 1]
        y_local = s * rel[:, 0] + c * rel[:, 1]
        z_local = rel[:, 2]
        l, w, h = box.size
        inside = ((np.abs(x_local) <= l / 2 + self.visibility_margin) & (np.abs(y_local) <= w / 2 + self.visibility_margin)
                  & (np.abs(z_local) <= h / 2 + self.visibility_margin))
        return int(np.count_nonzero(inside)) >= self.min_points_for_visibility

    def _actor_boxes(self, ego_transform: carla.Transform, frame_id: str, lidar_points_ego: np.ndarray) -> List[Box]:
        ego_pos = np.array([ego_transform.location.x, ego_transform.location.y])
        boxes = []
        all_actors = self.world.get_actors()
        candidates = list(all_actors.filter("vehicle.*")) + list(all_actors.filter("walker.pedestrian.*"))
        
        def _tag_from_type_id(type_id: str) -> str:
            t = type_id.lower()
            if t.startswith("walker."):
                return "pedestrian"
            for keyword in ("truck", "bus", "motorcycle", "bicycle", "rider", "train"):
                if keyword in t:
                    return keyword
            if t.startswith("vehicle."):
                return "car"
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
            if self._is_visible(box, lidar_points_ego):
                boxes.append(box)
        return boxes

    def _static_boxes(self, ego_transform: carla.Transform, frame_id: str, lidar_points_ego: np.ndarray) -> List[Box]:
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
        
        boxes = [_box_from_static_ref(r, ego_transform, frame_id=frame_id) for r in nearby]
        return [b for b in boxes if self._is_visible(b, lidar_points_ego)]

    def update(self, ego_transform: carla.Transform, frame_id: str,
                lidar_points_ego: np.ndarray) -> List[Box]:
        return (self._actor_boxes(ego_transform, frame_id, lidar_points_ego) +
                 self._static_boxes(ego_transform, frame_id, lidar_points_ego))

def print_results(metrics: dict) -> None:
    print()
    print("=" * 78)
    print("CLASS-AGNOSTIC BOX PRESENCE / ERROR EVALUATION")
    print("=" * 78)
    print(f"{'threshold (m)':>14}{'num_gt':>10}{'num_pred':>10}{'matched':>10}"
          f"{'recall':>10}{'precision':>10}{'F1':>8}")
    for thresh, m in metrics["per_threshold"].items():
        print(f"{thresh:>14.1f}{m['num_gt']:>10}{m['num_pred']:>10}{m['num_matched']:>10}"
              f"{m['recall']:>10.3f}{m['precision']:>10.3f}{m['f1']:>8.3f}")
    print()
    print(f"TP errors at {metrics['tp_threshold']} m match threshold (matched pairs only):")
    e = metrics["tp_errors"]
    print(f"  ATE (m)     : {e['trans_err']:.3f}")
    print(f"  ASE (1-IoU) : {e['scale_err']:.3f}")
    print(f"  AOE (rad)   : {e['orient_err']:.3f}")
    print(f"  AVE (m/s)   : {e['vel_err']:.3f}")
    if metrics["tp_errors_by_tag"]:
        print("\n  by GT tag:")
        for tag, e in metrics["tp_errors_by_tag"].items():
            print(f"    {tag:<12} ATE={e['trans_err']:.3f}  ASE={e['scale_err']:.3f}  "
                  f"AOE={e['orient_err']:.3f}  AVE={e['vel_err']:.3f}")
    print("=" * 78)

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

class OnlineEvaluator:
    def __init__(self, world: carla.World, ego_actor_id: int, thresholds: Sequence[float] = hp.DIST_THRESHOLDS,
                 tp_threshold: float = hp.TP_DISTANCE_THRESHOLD, max_range: float = 50):
        self.thresholds = list(thresholds)
        self.tp_threshold = tp_threshold
        self.max_range = max_range
        self._gt_frames: Dict[str, List[Box]] = {}
        self._pred_frames: Dict[str, List[Box]] = {}
        self.gt_extractor: CarlaGroundTruthExtractor = CarlaGroundTruthExtractor(world, ego_actor_id)

    def add_frame(self, frame_id: str, ego_state: Ego3DState, lidar_points_ego: np.ndarray, track_bbs: List[trackedbb]):
        ego_transform = _ego_transform(ego_state)
        self._gt_frames[frame_id] = list(self.gt_extractor.update(ego_transform, frame_id, lidar_points_ego))
        pred_boxes : list[Box] = []
        for pb in track_bbs:
            pred_boxes.append(Box.from_trackedbb(pb, frame_id=frame_id))
        pred_boxes = global_boxes_to_ego(pred_boxes, ego_state)
        self._pred_frames[frame_id] = pred_boxes
        print(len(pred_boxes))

    def reset(self) -> None:
        self._gt_frames.clear()
        self._pred_frames.clear()

    @property
    def num_frames(self) -> int:
        return len(set(self._gt_frames) | set(self._pred_frames))

    def evaluate(self) -> dict:
        return evaluate_frames(self._gt_frames, self._pred_frames, self.thresholds, self.tp_threshold, self.max_range)

    def print_report(self) -> None:
        print_results(self.evaluate())

    def save_metrics_json(self, filename: "str | Path") -> None:
        save_metrics(self.evaluate(), Path(filename))
