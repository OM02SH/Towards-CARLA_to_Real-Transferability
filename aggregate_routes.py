#!/usr/bin/env python3
"""
Aggregate all per-route evaluation files into route-level statistics.

Usage:
  python aggregate_routes.py --dir out --out agg/ [--min-vis 20]

Expects in --dir (tag like Town12_35; trailing number = index in results.json):
  bbs_<tag>.json (required), ego_<tag>.json (optional), results.json (optional)

Scoring rule (verified against the per_threshold blocks of the bbs json):
  visible GT : matched -> TP (counts for recall), missed -> FN
  unseen  GT : matched -> bonus (TP for precision, credited in R_adj), missed -> nothing
  P     = (TP_vis + TP_unseen) / N_pred
  R_vis = TP_vis / N_vis                      (headline recall)
  R_adj = (TP_vis + TP_unseen) / (N_vis + TP_unseen)
  F1    = harmonic mean of P and R_adj
Uncertainty: cluster bootstrap over ROUTES (pooled ratio of sums), N_BOOT resamples.
Outputs: per_route.csv, *.csv tables, *.tex tables (booktabs), *.png figures.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THR = ["0.5", "1.0", "2.0", "4.0"]
N_BOOT, SEED = 2000, 0


def fl(x):
    return np.nan if x is None else float(x)


# ----------------------------------------------------------------------------- loading
def load(d):
    d = Path(d)
    recs = {}
    if (d / "results.json").exists():
        recs = {r["index"]: r for r in json.load(open(d / "results.json"))["_checkpoint"]["records"]}
    rows, strata, gtsrc = [], [], []
    for fp in sorted(d.glob("bbs_*.json")):
        tag = fp.stem[4:]
        try:
            idx = int(re.search(r"(\d+)$", tag).group(1))
            b = json.load(open(fp)); s = b["summary"]; dg = b.get("diagnostics", {}); pt = b["per_threshold"]
            ep = d / f"ego_{tag}.json"
            e = json.load(open(ep)) if ep.exists() else None
            r = recs.get(idx, {})
            row = dict(tag=tag, idx=idx, scenario=r.get("scenario_name"), status=r.get("status"),
                       score=r.get("scores", {}).get("score_composed"),
                       frames=b["meta"].get("frames_evaluated"),
                       mean_speed=fl(b["meta"].get("ego_speed_m_s", {}).get("mean")),
                       pred=pt["2.0"]["num_pred"] or 0, vis_gt=pt["2.0"]["num_gt_visible"] or 0)
            for t in THR:
                row[f"tp_{t}"] = pt[t]["num_matched"] or 0
                row[f"tpu_{t}"] = pt[t]["num_matched_unseen"] or 0
            row.update(ate=fl(s.get("ATE_m")), ase=fl(s.get("ASE_1_minus_iou")), aoe=fl(s.get("AOE_rad")),
                       aoe_fold=fl(s.get("AOE_flip_folded_rad")), ave=fl(s.get("AVE_m_s")))
            fpd = dg.get("false_positives", {})
            row.update(fp=fl(fpd.get("total")), fp_dup=fl(fpd.get("duplicate", {}).get("count")),
                       fp_near=fl(fpd.get("near_miss", {}).get("count")),
                       fp_ghost=fl(fpd.get("ghost", {}).get("count")),
                       ghosts=fl(dg.get("ghosts", {}).get("total")))
            if e:
                ag, tr = e["aggregate"], e["trajectory"]
                row.update(ego_pos_rmse=ag["position_2d"]["rmse"], ego_pos_p95=ag["position_2d"]["p95_abs"],
                           ego_long_bias=ag["position_longitudinal"]["bias"],
                           ego_lat_bias=ag["position_lateral"]["bias"],
                           ego_rpe_rmse=ag["rpe_translation"]["rmse"], ego_rpe_yaw_rmse=ag["rpe_yaw"]["rmse"],
                           ego_path=tr["path_length_m"], ego_drift_pct=tr["drift_percent"],
                           ego_final_yaw_err=tr["final_yaw_err_deg"])
            rows.append(row)
            for kind, key in (("range", "by_range"), ("speed", "by_ego_speed")):
                for it in dg.get(key, []):
                    rng_ = it.get("range_m") or it.get("ego_speed_m_s")
                    lo, hi = rng_
                    label = f"{lo:g}-{'inf' if hi is None else format(hi, 'g')}"
                    vis = it.get("visible_gt") or 0
                    rv = it.get("rec_visible")
                    pred = it.get("predictions") or 0
                    prec = it.get("precision")
                    unseen = it.get("unseen_gt") or 0
                    uh = it.get("unseen_hit_rate")
                    strata.append(dict(route=tag, kind=kind, label=label, lo=lo, vis=vis,
                                       mv=(rv or 0) * vis, unseen=unseen, mu=(uh or 0) * unseen, pred=pred,
                                       fp=(pred * (1 - prec)) if (prec is not None and pred) else 0.0,
                                       tp=(pred * prec) if (prec is not None and pred) else 0.0))
            for src, v in dg.get("recall_by_gt_source", {}).items():
                gtsrc.append(dict(route=tag, source=src, visible_gt=v.get("visible_gt") or 0,
                                  unseen_gt=v.get("unseen_gt") or 0))
        except Exception as ex:
            print(f"!! skipped {fp.name}: {type(ex).__name__}: {ex}")
    return pd.DataFrame(rows), pd.DataFrame(strata), pd.DataFrame(gtsrc)


# ----------------------------------------------------------------------------- statistics
def make_idx(n):
    return np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))


def boot(num, den, idx, mask=None):
    """Pooled ratio of sums over routes + its bootstrap distribution (resampling routes)."""
    num = np.nan_to_num(np.asarray(num, float)); den = np.nan_to_num(np.asarray(den, float))
    if mask is not None:
        num = num * mask; den = den * mask
    pt = num.sum() / den.sum() if den.sum() > 0 else np.nan
    with np.errstate(all="ignore"):
        b = num[idx].sum(1) / den[idx].sum(1)
    return pt, b


def ci(b, n):
    if n < 2 or not np.isfinite(b).any():
        return np.nan, np.nan
    return tuple(np.nanpercentile(b, [2.5, 97.5]))


def fmt(pt, lo, hi, p=3):
    if not np.isfinite(pt):
        return "n/a"
    if not np.isfinite(lo):
        return f"{pt:.{p}f}"
    return f"{pt:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def tex_escape(x):
    return str(x).replace("_", r"\_").replace("%", r"\%")


def write_tex(df, path, caption, label):
    cols = list(df.columns)
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(tex_escape(c) for c in cols) + " \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(" & ".join(tex_escape(v) for v in r.values) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


# ----------------------------------------------------------------------------- tables
def threshold_table(df, idx, min_vis):
    n = len(df); rows = []; curves = {}
    vm = (df.vis_gt >= min_vis).astype(float).values
    for t in THR:
        tp, tpu = df[f"tp_{t}"].values, df[f"tpu_{t}"].values
        P, bP = boot(tp, df.pred.values, idx)
        Rv, bRv = boot(tp - tpu, df.vis_gt.values, idx, vm)
        Ra, bRa = boot(tp, df.vis_gt.values + tpu, idx)
        with np.errstate(all="ignore"):
            F, bF = 2 * P * Ra / (P + Ra), 2 * bP * bRa / (bP + bRa)
        curves[t] = {k: (pt, *ci(b, n)) for k, (pt, b) in
                     dict(P=(P, bP), R_vis=(Rv, bRv), R_adj=(Ra, bRa), F1=(F, bF)).items()}
        rows.append({"threshold_m": t, "precision": fmt(P, *ci(bP, n)), "recall_visible": fmt(Rv, *ci(bRv, n)),
                     "recall_bonus_adj": fmt(Ra, *ci(bRa, n)), "F1": fmt(F, *ci(bF, n))})
    return pd.DataFrame(rows), curves


def headline_table(df, idx, min_vis):
    n = len(df); rows = []
    vm = (df.vis_gt >= min_vis).astype(float).values
    tp, tpu = df["tp_2.0"].values, df["tpu_2.0"].values

    def add(name, pt, b, p=3):
        rows.append({"metric": name, "pooled [95% CI over routes]": fmt(pt, *ci(b, n), p)})

    add("precision (2 m)", *boot(tp, df.pred.values, idx))
    add("recall, visible GT (2 m)", *boot(tp - tpu, df.vis_gt.values, idx, vm))
    add("recall, bonus-adjusted (2 m)", *boot(tp, df.vis_gt.values + tpu, idx))
    # macro (route-mean) variants: every route counts equally, so long/stationary routes cannot dominate
    pr = df["tp_2.0"] / df.pred.replace(0, np.nan)
    rr = ((df["tp_2.0"] - df["tpu_2.0"]) / df.vis_gt.replace(0, np.nan)).where(df.vis_gt >= min_vis)
    for name, v in (("precision, route-mean (macro)", pr), ("recall visible, route-mean (macro)", rr),
                    ("ATE [m], route-mean (macro)", df.ate)):
        ok = v.notna().astype(float).values
        add(name, *boot(v.fillna(0).values, ok, idx))
    for col, name, p in (("ate", "ATE [m]", 3), ("ase", "ASE (1-IoU)", 3), ("aoe", "AOE raw [rad]", 3),
                         ("aoe_fold", "AOE flip-folded [rad]", 3), ("ave", "AVE [m/s]", 3)):
        ok = df[col].notna().astype(float).values
        add(name, *boot(df[col].fillna(0).values * tp, tp * ok, idx), p)
    add("false positives per route", *boot(df.fp.values, np.ones(n), idx), 1)
    add("ghosts per route (mean)", *boot(df.ghosts.values, np.ones(n), idx), 1)
    med = np.nanmedian(df.ghosts.values)
    bm = np.nanmedian(np.nan_to_num(df.ghosts.values)[idx], axis=1)
    add("ghosts per route (median)", med, bm, 1)
    if "ego_pos_rmse" in df:
        ones = df.ego_pos_rmse.notna().astype(float).values
        for col, name in (("ego_pos_rmse", "ego position RMSE [m] (route mean)"),
                          ("ego_pos_p95", "ego position p95 [m] (route mean)"),
                          ("ego_long_bias", "ego longitudinal bias [m] (route mean)"),
                          ("ego_lat_bias", "ego lateral bias [m] (route mean)"),
                          ("ego_rpe_rmse", "ego RPE translation RMSE [m] (route mean)")):
            add(name, *boot(df[col].fillna(0).values, ones, idx))
        w = df.ego_path.fillna(0).values
        add("ego drift [% of path] (path-weighted)", *boot(df.ego_drift_pct.fillna(0).values * w, w, idx))
    return pd.DataFrame(rows)


def strata_table(df, sdf, idx, kind):
    n = len(df); rows = []
    if sdf.empty:
        return pd.DataFrame()
    sub = sdf[sdf.kind == kind]
    for (lo, label), g in sub.groupby(["lo", "label"]):
        g = g.set_index("route").reindex(df.tag).fillna(0)
        P, bP = boot(g.tp.values, g.pred.values, idx)
        R, bR = boot(g.mv.values, g.vis.values, idx)
        U, bU = boot(g.mu.values, g.unseen.values, idx)
        row = {"bin": label, "visible_GT": int(g.vis.sum()), "routes_with_visible_GT": int((g.vis > 0).sum()),
               "precision": fmt(P, *ci(bP, n)), "recall_visible": fmt(R, *ci(bR, n))}
        if g.unseen.sum() > 0:   # the ego-speed strata carry no unseen-GT counts
            row["unseen_hit_rate"] = fmt(U, *ci(bU, n))
        row.update({"_P": P, "_R": R, "_Pci": ci(bP, n), "_Rci": ci(bR, n), "_lo": lo})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("_lo").reset_index(drop=True)


def fp_taxonomy(df, idx):
    n = len(df); rows = []
    for c, name in (("fp_near", "near-miss"), ("fp_dup", "duplicate"), ("fp_ghost", "ghost")):
        pt, b = boot(df[c].values, df.fp.values, idx)
        rows.append({"type": name, "count": int(np.nansum(df[c])), "share of FPs [95% CI]": fmt(pt, *ci(b, n))})
    return pd.DataFrame(rows)


def ego_lag(df, idx):
    """Does the along-track bias scale with speed? bias = -lag * v  -> lag = -sum(v*b)/sum(v^2)."""
    if "ego_long_bias" not in df:
        return None
    d = df.dropna(subset=["ego_long_bias", "mean_speed"])
    d = d[d.mean_speed > 0.5]
    if len(d) < 2:
        return None
    v, b = d.mean_speed.values, d.ego_long_bias.values
    lag = -(v * b).sum() / (v * v).sum()
    rng = np.random.default_rng(SEED).integers(0, len(d), (N_BOOT, len(d)))
    bl = -(v[rng] * b[rng]).sum(1) / (v[rng] ** 2).sum(1)
    lo, hi = np.percentile(bl, [2.5, 97.5])
    return dict(lag_s=lag, lo=lo, hi=hi, n=len(d), d=d)


# ----------------------------------------------------------------------------- figures
def fig_thresholds(curves, out):
    x = [float(t) for t in THR]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for k, lab, c in (("P", "precision", "tab:blue"), ("R_vis", "recall (visible)", "tab:green"),
                      ("R_adj", "recall (bonus-adjusted)", "tab:orange")):
        pt = [curves[t][k][0] for t in THR]; lo = [curves[t][k][1] for t in THR]; hi = [curves[t][k][2] for t in THR]
        ax.plot(x, pt, "o-", color=c, label=lab)
        if np.isfinite(lo).all():
            ax.fill_between(x, lo, hi, color=c, alpha=.2)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(THR)
    ax.set_xlabel("match threshold [m]"); ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("pooled over routes, 95% CI (route bootstrap)")
    fig.savefig(out / "fig_thresholds.png", dpi=150); plt.close(fig)


def fig_strata(tabs, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for a, (name, t) in zip(ax, tabs.items()):
        if t.empty:
            continue
        x = np.arange(len(t)); w = .38
        for off, col, cic, lab, c in ((-w / 2, "_P", "_Pci", "precision", "tab:blue"),
                                      (w / 2, "_R", "_Rci", "recall (visible)", "tab:green")):
            v = t[col].values.astype(float)
            err = np.array([[v[i] - t[cic][i][0], t[cic][i][1] - v[i]] for i in range(len(t))]).T
            a.bar(x + off, v, w, yerr=err if np.isfinite(err).all() else None, color=c, label=lab, capsize=3)
        a.set_xticks(x)
        a.set_xticklabels([f"{b}\n(n_vis={n})" for b, n in zip(t.bin, t.visible_GT)], fontsize=8)
        a.set_ylim(0, 1.05); a.set_title(name); a.legend(fontsize=8)
    fig.savefig(out / "fig_strata.png", dpi=150); plt.close(fig)


def fig_ego_lag(res, out):
    d = res["d"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    ax.scatter(d.mean_speed, d.ego_long_bias)
    xs = np.linspace(0, d.mean_speed.max(), 50)
    ax.plot(xs, -res["lag_s"] * xs, "r--",
            label=f"lag = {res['lag_s']*1e3:.0f} ms [{res['lo']*1e3:.0f}, {res['hi']*1e3:.0f}]\n= {res['lag_s']*20:.2f} frames @20 Hz")
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("route mean ego speed [m/s]"); ax.set_ylabel("longitudinal position bias [m]")
    ax.set_title("does along-track bias scale with speed?"); ax.legend(fontsize=8)
    fig.savefig(out / "fig_ego_bias_vs_speed.png", dpi=150); plt.close(fig)


def fig_scenarios(df, out):
    if df.scenario.isna().all():
        return
    d = df.copy(); d["family"] = d.scenario.str.replace(r"_\d+$", "", regex=True)
    d["recall_vis"] = np.where(d.vis_gt > 0, (d["tp_2.0"] - d["tpu_2.0"]) / d.vis_gt.replace(0, np.nan), np.nan)
    d["precision"] = d["tp_2.0"] / d.pred.replace(0, np.nan)
    fams = sorted(d.family.dropna().unique())
    fig, ax = plt.subplots(1, 3, figsize=(15, 0.35 * len(fams) + 2), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(0)
    for a, (col, lab) in zip(ax, (("ate", "ATE [m]"), ("recall_vis", "recall (visible)"), ("precision", "precision"))):
        for i, fam in enumerate(fams):
            g = d[d.family == fam]
            a.scatter(g[col], i + rng.uniform(-.15, .15, len(g)), s=22,
                      c=["tab:red" if (s is not None and s < 100) else "tab:blue" for s in g.score])
        a.set_xlabel(lab); a.grid(axis="x", alpha=.3)
    ax[0].set_yticks(range(len(fams)))
    ax[0].set_yticklabels([f"{f} (n={int((d.family == f).sum())})" for f in fams], fontsize=8)
    fig.suptitle("each dot = one route (red = driving score < 100); no family means shown")
    fig.savefig(out / "fig_scenario_dots.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="."); ap.add_argument("--out", default="agg")
    ap.add_argument("--min-vis", type=int, default=20, help="min visible-GT frames for a route to enter recall_visible")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)

    df, sdf, gsrc = load(a.dir)
    if df.empty:
        raise SystemExit("no routes loaded")
    df["precision"] = df["tp_2.0"] / df.pred.replace(0, np.nan)
    df["recall_vis"] = np.where(df.vis_gt >= a.min_vis, (df["tp_2.0"] - df["tpu_2.0"]) / df.vis_gt.replace(0, np.nan), np.nan)
    df.to_csv(out / "per_route.csv", index=False)
    idx = make_idx(len(df))
    print(f"{len(df)} routes loaded; {int((df.vis_gt >= a.min_vis).sum())} have >= {a.min_vis} visible-GT frames "
          f"(only these enter recall_visible)")

    head = headline_table(df, idx, a.min_vis)
    thr, curves = threshold_table(df, idx, a.min_vis)
    rng_t, spd_t = strata_table(df, sdf, idx, "range"), strata_table(df, sdf, idx, "speed")
    fpt = fp_taxonomy(df, idx)
    for name, t in (("headline", head), ("thresholds", thr), ("fp_taxonomy", fpt)):
        t.to_csv(out / f"table_{name}.csv", index=False)
    show = lambda t: t[[c for c in t.columns if not c.startswith("_")]]
    for name, t in (("by_range_m", rng_t), ("by_ego_speed_m_s", spd_t)):
        if not t.empty:
            show(t).to_csv(out / f"table_{name}.csv", index=False)

    print("\n== headline (pooled over routes; CI = route-level cluster bootstrap) ==\n", head.to_string(index=False))
    print("\n== per match threshold ==\n", thr.to_string(index=False))
    print("\n== by range [m] ==\n", show(rng_t).to_string(index=False) if not rng_t.empty else "n/a")
    print("\n== by ego speed [m/s] ==\n", show(spd_t).to_string(index=False) if not spd_t.empty else "n/a")
    print("\n== false-positive taxonomy (2 m) ==\n", fpt.to_string(index=False))
    if not gsrc.empty:
        gs = gsrc.groupby("source")[["visible_gt", "unseen_gt"]].sum()
        gs.to_csv(out / "table_gt_sources.csv")
        print("\n== GT classes present (frames summed over routes) ==\n", gs.to_string())

    lag = ego_lag(df, idx)
    if lag:
        fig_ego_lag(lag, out)
        print(f"\n== ego along-track bias vs speed ==\n implied lag = {lag['lag_s']*1e3:.0f} ms "
              f"[{lag['lo']*1e3:.0f}, {lag['hi']*1e3:.0f}] = {lag['lag_s']*20:.2f} frames at 20 Hz  (n={lag['n']} routes)"
              "\n assumes error = estimate - GT and uses the mean speed over evaluated frames; indicative only")
    print("\n== worst routes by ATE ==")
    print(df.sort_values("ate", ascending=False)[["tag", "scenario", "score", "ate", "ave", "precision", "recall_vis",
                                                  "fp", "ghosts"]].head(8).round(3).to_string(index=False))

    write_tex(head, out / "table_headline.tex", "Pooled detection/tracking and ego-state results over all routes (95\\% CI from a route-level cluster bootstrap).", "tab:headline")
    write_tex(thr, out / "table_thresholds.tex", "Detection metrics at each match threshold.", "tab:thresholds")
    for name, t, lab in (("range", rng_t, "range [m]"), ("speed", spd_t, "ego speed [m/s]")):
        if not t.empty:
            write_tex(show(t), out / f"table_by_{name}.tex", f"Detection metrics stratified by {lab}.", f"tab:by{name}")
    fig_thresholds(curves, out)
    fig_strata({"by range [m]": rng_t, "by ego speed [m/s]": spd_t}, out)
    fig_scenarios(df, out)
    print(f"\nwrote tables (.csv/.tex) and figures (.png) to {out}/")