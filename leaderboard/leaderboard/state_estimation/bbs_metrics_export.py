"""
bbs_metrics_export.py  -  everything the online evaluator knows, written to disk.

  <name>.json            run configuration, headline numbers, every diagnostics table, 95% confidence intervals
  <name>_frames.csv      one row per evaluated frame (for plots over time / per-frame distributions)

Put this file next to real_test_BBs.py (leaderboard/state_estimation/). It needs no changes when the
evaluator or the diagnostics change shape, apart from reading their result dictionaries.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import platform
import sys
from typing import Dict, List, Optional

import numpy as np

NAN = float("nan")


# ----------------------------------------------------------------------------------------------
# JSON safety: numpy scalars/arrays, NaN/inf -> null, tuple keys -> "a|b"
# ----------------------------------------------------------------------------------------------
def _key(k) -> str:
    if isinstance(k, tuple):
        return "|".join(str(x) for x in k)
    return str(k)


def _jsonable(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, dict):
        return {_key(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    return str(o)


def _ratio(a, b) -> float:
    return float(a) / float(b) if b else NAN


def _consts(mod) -> dict:
    """Plain scalar UPPER_CASE module constants (hyperparameters, debug switches)."""
    if mod is None:
        return {}
    return {k: v for k, v in vars(mod).items()
            if k.isupper() and not k.startswith("_") and isinstance(v, (bool, int, float, str))}


def _frame_sort_key(fid):
    try:
        return (0, int(fid), str(fid))
    except (TypeError, ValueError):
        return (1, 0, str(fid))


# ----------------------------------------------------------------------------------------------
# run configuration
# ----------------------------------------------------------------------------------------------
def _meta(ev, rt, extra: Optional[dict]) -> dict:
    speeds = np.array(list(ev._ego_speed.values()), dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    gtx = getattr(ev, "gt_extractor", None)
    modules = {}
    for name in ("leaderboard.state_estimation.hyperparams",
                 "leaderboard.state_estimation.bbs_KF",
                 "leaderboard.state_estimation.UKF_bb"):
        c = _consts(sys.modules.get(name))
        if c:
            modules[name.split(".")[-1]] = c
    return {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "frames_seen_including_warmup": int(getattr(ev, "_seen", 0)),
        "frames_evaluated": int(len(set(ev._gt_frames) | set(ev._pred_frames))),
        "ego_speed_m_s": {
            "mean": float(speeds.mean()) if speeds.size else None,
            "max": float(speeds.max()) if speeds.size else None,
            "fraction_below_0.5": float(np.mean(speeds < 0.5)) if speeds.size else None,
        },
        "evaluator": {
            "thresholds_m": list(getattr(ev, "thresholds", [])),
            "tp_threshold_m": ev.tp_threshold,
            "max_range_m": ev.max_range,
            "warmup_frames": getattr(ev, "warmup_frames", None),
            "stationary_stride": getattr(ev, "stationary_stride", None),
            "ghost_probe": getattr(ev, "probe_ghosts", None),
            "ghost_probe_radius_m": getattr(ev, "probe_radius", None),
        },
        "ground_truth": {
            "actor_max_range_m": getattr(gtx, "actor_max_range", None),
            "static_max_range_m": getattr(gtx, "static_max_range", None),
            "min_points_for_visibility": getattr(gtx, "min_points_for_visibility", None),
            "visibility_margin_m": getattr(gtx, "visibility_margin", None),
            "include_mesh_props": getattr(gtx, "include_mesh_props", "always on in this version"),
            "canonical_props": getattr(gtx, "canonical_props", "always on in this version"),
            "num_static_env_vehicles": len(getattr(gtx, "static_refs", []) or []),
        },
        "constants": modules,
        "extra": extra or {},
    }


def _summary(metrics: dict) -> dict:
    tp = metrics["tp_threshold"]
    m = next(v for k, v in metrics["per_threshold"].items() if abs(float(k) - tp) < 1e-9)
    e = metrics["tp_errors"]
    return {
        "match_threshold_m": tp,
        "precision": m["precision"],
        "recall_visible": m["recall_visible"],
        "recall_bonus_adjusted": m["recall"],
        "f1": m["f1"],
        "visible_gt": m["num_gt_visible"],
        "matched_unseen_gt": m["num_matched_unseen"],
        "predictions": m["num_pred"],
        "matched": m["num_matched"],
        "ATE_m": e["trans_err"],
        "ASE_1_minus_iou": e["scale_err"],
        "AOE_rad": e["orient_err"],
        "AOE_flip_folded_rad": e["orient_fold_err"],
        "AVE_m_s": e["vel_err"],
        "note": "recall_visible is the headline recall; recall_bonus_adjusted also credits unseen GT that was found",
    }


# ----------------------------------------------------------------------------------------------
# per-frame table + block bootstrap confidence intervals
# ----------------------------------------------------------------------------------------------
_FRAME_COLUMNS = ["frame", "ego_speed", "ego_x", "ego_y", "n_gt", "n_gt_visible", "n_pred", "matched",
                  "matched_visible", "matched_unseen", "false_pos", "precision", "rec_visible",
                  "mean_ate", "mean_vel_err", "ate_sum", "ate_n"]


def _per_frame(ev, rt) -> List[dict]:
    rows = []
    for fid in sorted(set(ev._gt_frames) | set(ev._pred_frames), key=_frame_sort_key):
        gt = rt._filter_range(ev._gt_frames.get(fid, []), ev.max_range)
        pred = rt._filter_range(ev._pred_frames.get(fid, []), ev.max_range)
        pairs, _, un_pred = rt.greedy_match(gt, pred, ev.tp_threshold)
        vis = [rt._is_visible(g) for g in gt]
        hit_vis = sum(1 for gi, _ in pairs if vis[gi])
        ates = [rt.center_distance(gt[gi], pred[pi]) for gi, pi in pairs]
        vels = [rt.velocity_l2(gt[gi], pred[pi]) for gi, pi in pairs]
        pose = ev._ego_pose.get(fid, (NAN, NAN, NAN))
        rows.append({
            "frame": fid, "ego_speed": ev._ego_speed.get(fid, NAN), "ego_x": pose[0], "ego_y": pose[1],
            "n_gt": len(gt), "n_gt_visible": int(sum(vis)), "n_pred": len(pred), "matched": len(pairs),
            "matched_visible": hit_vis, "matched_unseen": len(pairs) - hit_vis, "false_pos": len(un_pred),
            "precision": _ratio(len(pairs), len(pred)), "rec_visible": _ratio(hit_vis, sum(vis)),
            "mean_ate": float(np.mean(ates)) if ates else NAN,
            "mean_vel_err": float(np.nanmean(vels)) if vels and np.isfinite(vels).any() else NAN,
            "ate_sum": float(sum(ates)), "ate_n": len(ates),
        })
    return rows


def _write_frames_csv(rows: List[dict], path: str) -> None:
    with open(path, "w") as f:
        f.write(",".join(_FRAME_COLUMNS) + "\n")
        for r in rows:
            f.write(",".join("" if (isinstance(r[c], float) and not math.isfinite(r[c])) else str(r[c])
                             for c in _FRAME_COLUMNS) + "\n")


def _bootstrap(rows: List[dict], block: int = 20, n_boot: int = 300, seed: int = 0) -> dict:
    """Block bootstrap over frames (neighbouring frames are strongly correlated, so resample blocks)."""
    if len(rows) < 8:
        return {}
    keys = ("n_pred", "matched", "n_gt_visible", "matched_visible", "ate_sum", "ate_n")
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    block = max(1, min(block, len(rows) // 4))
    starts = np.arange(0, len(rows), block)
    sums = {k: np.add.reduceat(v, starts) for k, v in arr.items()}
    rng = np.random.default_rng(seed)
    draws = {"precision": [], "recall_visible": [], "ATE_m": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(starts), len(starts))
        s = {k: v[idx].sum() for k, v in sums.items()}
        draws["precision"].append(_ratio(s["matched"], s["n_pred"]))
        draws["recall_visible"].append(_ratio(s["matched_visible"], s["n_gt_visible"]))
        draws["ATE_m"].append(_ratio(s["ate_sum"], s["ate_n"]))
    out = {"method": f"block bootstrap, {block}-frame blocks, {n_boot} resamples, seed {seed}"}
    for k, v in draws.items():
        v = np.array(v, dtype=float)
        v = v[np.isfinite(v)]
        out[k] = {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} if v.size else None
    return out


# ----------------------------------------------------------------------------------------------
# diagnostics: the same tables print_diagnostics shows, as data
# ----------------------------------------------------------------------------------------------
def diagnostics_to_dict(d: dict, rt) -> dict:
    fmean, fmedian = rt._fmean, rt._fmedian
    out: dict = {"match_threshold_m": d["thresh"]}

    out["by_range"] = [{
        "range_m": [lo, hi], "visible_gt": r["vis"], "rec_visible": _ratio(r["hit_vis"], r["vis"]),
        "unseen_gt": r["unseen"], "unseen_hit_rate": _ratio(r["hit_unseen"], r["unseen"]),
        "predictions": r["pred"], "false_positives": r["fp"],
        "precision": 1.0 - _ratio(r["fp"], r["pred"]) if r["pred"] else NAN,
    } for (lo, hi), r in zip(d["ranges"], d["range_stats"])]

    e = d["ego_edges"]
    bounds = [(0.0, e[0])] + [(e[i], e[i + 1]) for i in range(len(e) - 1)] + [(e[-1], None)]
    out["by_ego_speed"] = [{
        "ego_speed_m_s": [lo, hi], "frames": b["frames"], "visible_gt": b["vis"],
        "rec_visible": _ratio(b["hit"], b["vis"]), "predictions": b["pred"],
        "precision": 1.0 - _ratio(b["fp"], b["pred"]) if b["pred"] else NAN,
        "ATE_m": _ratio(b["ate_sum"], b["ate_n"]),
    } for (lo, hi), b in zip(bounds, d["by_ego"])]

    tot = sum(d["fp_kind"].values())
    out["false_positives"] = {"total": tot, **{k: {"count": v, "share": _ratio(v, tot)} for k, v in d["fp_kind"].items()}}

    rows = d["ghost_rows"]
    ghosts: dict = {"total": len(rows), "by_type": {f"{sz}|{mo}": n for (sz, mo), n in d["ghost_anatomy"].items()}}
    if rows:
        rng_m = np.array([r[10] for r in rows], dtype=float)
        edges = [0.0, 3.0, 6.0, 10.0, 15.0, 30.0, 50.0001]
        hist, _ = np.histogram(rng_m, bins=edges)
        ghosts["range_histogram"] = [{"range_m": [edges[i], min(edges[i + 1], 50.0)], "count": int(hist[i]),
                                      "share": _ratio(hist[i], len(rows))} for i in range(len(hist))]
        ghosts["within_3m_of_ego"] = int(hist[0])
        cells: Dict[tuple, List[tuple]] = {}
        for r in rows:
            if np.isfinite(r[3]) and np.isfinite(r[4]):
                cells.setdefault((int(round(r[3] / 2.0)), int(round(r[4] / 2.0))), []).append(r)
        ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))
        n_fin = sum(len(v) for v in cells.values())
        ghosts["top10_world_cells_share"] = _ratio(sum(len(v) for _, v in ranked[:10]), n_fin)
        ghosts["top_world_cells"] = [{
            "x_world": cx * 2.0, "y_world": cy * 2.0, "count": len(v), "frames": len({r[0] for r in v}),
            "mean_length": float(np.mean([r[5] for r in v])), "mean_width": float(np.mean([r[6] for r in v])),
            "mean_speed": float(np.mean([r[8] for r in v])),
        } for (cx, cy), v in ranked[:20]]
        probed = [r for r in rows if r[11]]
        if probed:
            import collections
            ghosts["live_probe"] = {
                "probed": len(probed),
                "nothing_nearby": sum(1 for r in probed if r[11] == "none" and r[12] == "none"),
                "environment_labels": dict(collections.Counter(r[11].split(":")[0] for r in probed if r[11] != "none").most_common(20)),
                "actor_types": dict(collections.Counter(r[12].split("@")[0] for r in probed if r[12] != "none").most_common(20)),
            }
    out["ghosts"] = ghosts

    se = d["score_edges"]
    labels = [f"<{se[0]:.2f}"] + [f"{se[i]:.2f}-{se[i + 1]:.2f}" for i in range(len(se) - 1)] + [f">={se[-1]:.2f}"]
    out["precision_by_score"] = [{"score": lab, "predictions": s["n"],
                                  "precision": 1.0 - _ratio(s["fp"], s["n"]) if s["n"] else NAN,
                                  "ghost_share": _ratio(s["ghost"], s["n"])}
                                 for lab, s in zip(labels, d["score_stats"]) if s["n"]]

    out["orientation_non_pedestrian"] = {}
    for kind in ("static", "moving"):
        a = np.asarray(d["ori"][kind], dtype=float)
        if a.size:
            out["orientation_non_pedestrian"][kind] = {
                "n": int(a.size), "AOE_raw_rad": float(a.mean()),
                "AOE_flip_folded_rad": float(np.minimum(a, math.pi - a).mean()),
                "flipped_over_120deg_pct": 100.0 * float(np.mean(a > 2 * math.pi / 3)),
            }

    out["recall_by_gt_source"] = {src: {"visible_gt": v, "rec_visible": _ratio(hv, v), "unseen_gt": u,
                                        "unseen_hit_rate": _ratio(hu, u)}
                                  for src, (v, hv, u, hu) in d["rec_src"].items()}
    out["errors_by_gt_source"] = {}
    for src, vals in d["by_source"].items():
        a = np.asarray(vals, dtype=float)
        out["errors_by_gt_source"][src] = {
            "n": int(len(a)), "ATE_m": float(a[:, 0].mean()), "ASE": float(a[:, 1].mean()),
            "AOE_raw_rad": float(a[:, 2].mean()), "AOE_fold180_rad": float(a[:, 3].mean()),
            "AOE_fold90_rad": float(a[:, 4].mean()),
            "over_57deg_off_pct": 100.0 * float(np.mean(a[:, 3] > 1.0)),
            "flipped_over_120deg_pct": 100.0 * float(np.mean(a[:, 2] > 2 * math.pi / 3)),
        }

    st = np.asarray(d["vel"]["static"], dtype=float).reshape(-1, 14)
    mv = np.asarray(d["vel"]["moving"], dtype=float).reshape(-1, 14)
    vel: dict = {}
    ego_bins = ((0.0, 0.5), (0.5, 3.0), (3.0, math.inf))
    if len(st):
        vel["static_gt"] = {
            "n": int(len(st)), "AVE_m_s": fmean(st[:, 0]), "mean_pred_speed_m_s": fmean(st[:, 1]),
            "by_ego_speed": [{"ego_speed_m_s": [lo, None if math.isinf(hi) else hi],
                              "n": int(m.sum()), "AVE_m_s": fmean(st[m, 0])}
                             for lo, hi in ego_bins for m in [(st[:, 2] >= lo) & (st[:, 2] < hi)] if m.any()],
        }
    if len(mv):
        veh = mv[:, 5] == 0
        groups = (("pedestrians", mv[:, 5] == 1), ("vehicles 0.2-2 m/s", veh & (mv[:, 3] < 2)),
                  ("vehicles 2-6 m/s", veh & (mv[:, 3] >= 2) & (mv[:, 3] < 6)), ("vehicles >6 m/s", veh & (mv[:, 3] >= 6)))
        table = []
        for name, m in groups:
            if m.any():
                sub = mv[m]
                ang = sub[:, 4][np.isfinite(sub[:, 4])]
                table.append({
                    "group": name, "n": int(len(sub)), "AVE_m_s": fmean(sub[:, 0]), "gt_speed_m_s": fmean(sub[:, 3]),
                    "track_speed_m_s": fmean(sub[:, 1]), "raw_detector_speed_m_s": fmean(sub[:, 10]),
                    "radar_points_per_detection": fmean(sub[:, 11]), "median_angle_error_deg": fmedian(ang),
                    "reversed_over_120deg_pct": 100.0 * float(np.mean(ang > 120)) if ang.size else NAN,
                })
        vel["moving_gt"] = table
        v = mv[veh & np.isfinite(mv[:, 6]) & np.isfinite(mv[:, 7])]
        if len(v):
            pvx, pvy, gvx, gvy, es = v[:, 6], v[:, 7], v[:, 8], v[:, 9], v[:, 2]
            variants = {"as_is": (pvx, pvy), "vy_negated": (pvx, -pvy), "vx_negated": (-pvx, pvy),
                        "plus_ego_speed_on_x": (pvx + es, pvy), "minus_ego_speed_on_x": (pvx - es, pvy),
                        "predict_zero": (0 * pvx, 0 * pvy)}
            vel["tracker_velocity_convention_test_moving_vehicles"] = {
                k: fmean(np.hypot(x - gvx, y - gvy)) for k, (x, y) in variants.items()}
    raw = []
    for name, arr in (("static", st), ("moving_vehicles", mv[mv[:, 5] == 0] if len(mv) else mv)):
        for lo, hi in ego_bins:
            m = (arr[:, 2] >= lo) & (arr[:, 2] < hi) & np.isfinite(arr[:, 12]) & np.isfinite(arr[:, 13])
            if m.any():
                sub = arr[m]
                e_abs = float(np.hypot(sub[:, 12] - sub[:, 8], sub[:, 13] - sub[:, 9]).mean())
                e_rel = float(np.hypot(sub[:, 12] + sub[:, 2] - sub[:, 8], sub[:, 13] - sub[:, 9]).mean())
                raw.append({"gt": name, "ego_speed_m_s": [lo, None if math.isinf(hi) else hi], "n": int(m.sum()),
                            "error_if_absolute": e_abs, "error_if_relative": e_rel,
                            "verdict": "identical at standstill" if hi <= 0.5 else ("ABSOLUTE" if e_abs < e_rel else "RELATIVE")})
    if raw:
        vel["raw_detector_velocity_convention"] = raw
    out["velocity"] = vel
    return out


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------
def export_metrics(ev, path: str, extra_meta: Optional[dict] = None,
                   metrics: Optional[dict] = None, diag: Optional[dict] = None) -> dict:
    """Write <path> (JSON) and <path minus .json>_frames.csv. Returns the JSON-ready dictionary."""
    rt = sys.modules[type(ev).__module__]          # the evaluator's own module: no import-path assumptions
    metrics = metrics if metrics is not None else ev.evaluate()
    diag = diag if diag is not None else ev.diagnose()
    frames = _per_frame(ev, rt)

    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stem = path[:-5] if path.lower().endswith(".json") else path
    frames_csv = stem + "_frames.csv"
    _write_frames_csv(frames, frames_csv)

    result = {
        "meta": _meta(ev, rt, extra_meta),
        "files": {"per_frame_csv": frames_csv},
        "summary": _summary(metrics),
        "confidence_intervals_95": _bootstrap(frames),
        "per_threshold": metrics["per_threshold"],
        "tp_errors": metrics["tp_errors"],
        "tp_errors_by_tag": metrics["tp_errors_by_tag"],
        "diagnostics": diagnostics_to_dict(diag, rt),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(result), f, indent=2, allow_nan=False)
    os.replace(tmp, path)                           # never leaves a half-written file behind
    return result