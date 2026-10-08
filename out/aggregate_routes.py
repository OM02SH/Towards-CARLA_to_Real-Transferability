#!/usr/bin/env python3
"""
Aggregate all per-route evaluation files into route-level statistics.

Usage:
  python aggregate_routes.py --dir /path/to/all_files --out agg/

Expects in --dir (any number of routes, tag like Town12_35 where the trailing
number is the route index in results.json):
  bbs_<tag>.json, ego_<tag>.json, criteria_<tag>.json (optional), results.json
Resampling unit = ROUTE (cluster bootstrap), not frame.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_VIS_GT = 20      # routes with fewer visible-GT frames are excluded from recall
N_BOOT, SEED = 2000, 0


def load_routes(d):
    d = Path(d)
    recs = {r["index"]: r for r in json.load(open(d / "results.json"))["_checkpoint"]["records"]}
    rows = []
    for f in sorted(d.glob("bbs_*.json")):
        tag = f.stem[4:]
        idx = int(re.search(r"(\d+)$", tag).group(1))
        b = json.load(open(f)); s = b["summary"]
        e = json.load(open(d / f"ego_{tag}.json")) if (d / f"ego_{tag}.json").exists() else None
        r = recs.get(idx, {})
        n_match = s["matched"]
        rows.append(dict(
            tag=tag, idx=idx, scenario=r.get("scenario_name"), status=r.get("status"),
            score=r.get("scores", {}).get("score_composed"),
            frames=b["meta"]["frames_evaluated"], pred=s["predictions"], matched=n_match,
            vis_gt=s["visible_gt"], matched_vis=round(s["recall_visible"] * s["visible_gt"]),
            ate=s["ATE_m"], ave=s["AVE_m_s"], aoe=s["AOE_flip_folded_rad"], ase=s["ASE_1_minus_iou"],
            fp=b["diagnostics"]["false_positives"]["total"],
            fp_near=b["diagnostics"]["false_positives"]["near_miss"]["count"],
            ghosts=b["diagnostics"]["ghosts"]["total"],
            ego_pos_rmse=e["aggregate"]["position_2d"]["rmse"] if e else np.nan,
            ego_long_bias=e["aggregate"]["position_longitudinal"]["bias"] if e else np.nan,
            ego_yaw_p95=e["fields"]["yaw_rad"]["p95_abs"] if e else np.nan,
            ego_path=e["trajectory"]["path_length_m"] if e else np.nan,
            ego_drift_pct=e["trajectory"]["drift_percent"] if e else np.nan))
    return pd.DataFrame(rows)


def pooled(df):
    """Pooled metrics over a set of routes (ratio of sums; ATE etc. weighted by matches)."""
    v = df[df.vis_gt >= MIN_VIS_GT]
    w = df.matched.values
    return dict(
        precision=df.matched.sum() / max(df.pred.sum(), 1),
        recall_visible=v.matched_vis.sum() / max(v.vis_gt.sum(), 1) if len(v) else np.nan,
        ATE_m=np.average(df.ate, weights=w) if w.sum() else np.nan,
        AVE_m_s=np.average(df.ave, weights=w) if w.sum() else np.nan,
        ghosts_per_route=df.ghosts.mean(),
        ego_pos_rmse_m=df.ego_pos_rmse.mean(),
        ego_drift_pct=np.average(df.ego_drift_pct.fillna(0), weights=df.ego_path.fillna(0) + 1e-9))


def cluster_ci(df):
    rng = np.random.default_rng(SEED)
    pt = pooled(df)
    boots = [pooled(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(N_BOOT)]
    b = pd.DataFrame(boots)
    return pd.DataFrame({"estimate": pt, "lo95": b.quantile(.025), "hi95": b.quantile(.975)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="."); ap.add_argument("--out", default="agg")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    df = load_routes(a.dir)
    df["precision"] = df.matched / df.pred.replace(0, np.nan)
    df["recall_vis"] = np.where(df.vis_gt >= MIN_VIS_GT, df.matched_vis / df.vis_gt.replace(0, np.nan), np.nan)
    df.to_csv(out / "per_route.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(f"{len(df)} routes; {int((df.vis_gt >= MIN_VIS_GT).sum())} have >= {MIN_VIS_GT} visible-GT frames")
    print("\n== pooled, cluster bootstrap over routes ==\n", cluster_ci(df).round(3))
    print("\n== worst routes ==")
    print(df.sort_values("ate", ascending=False)[["tag", "scenario", "score", "ate", "ave", "precision", "recall_vis", "fp", "ghosts"]].head(8).round(3).to_string(index=False))
    if df.score.notna().sum() >= 8:
        print("\n== Spearman vs driving score (route level) ==")
        print(df[["score", "ate", "ave", "precision", "recall_vis", "ghosts", "ego_pos_rmse"]].corr(method="spearman")["score"].round(2))
    else:
        print("\n(too few routes with a score to correlate perception with driving outcome)")
    if "scenario" in df and df.scenario.notna().any():
        df["family"] = df.scenario.str.replace(r"_\d+$", "", regex=True)
        g = df.groupby("family").agg(routes=("tag", "count"), ATE=("ate", "mean"), prec=("precision", "mean"),
                                    rec=("recall_vis", "mean"), score=("score", "mean"))
        print("\n== per scenario family (n is small, treat as descriptive) ==\n", g.round(3))

    # figure: route-level distributions + metric vs outcome
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    ax[0].hist(df.ate, bins=15); ax[0].set_xlabel("route ATE [m]"); ax[0].set_title("ATE across routes")
    ax[1].scatter(df.ego_pos_rmse, df.ate); ax[1].set_xlabel("ego position RMSE [m]"); ax[1].set_ylabel("perception ATE [m]")
    ax[1].set_title("ego error vs perception error")
    c = ["tab:red" if (s is not None and s < 100) else "tab:green" for s in df.score]
    ax[2].scatter(df.precision, df.recall_vis, c=c); ax[2].set_xlabel("precision"); ax[2].set_ylabel("recall (visible)")
    ax[2].set_title("per route (red = score < 100)")
    fig.savefig(out / "route_level_overview.png", dpi=130)
