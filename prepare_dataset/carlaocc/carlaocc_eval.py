"""Detection evaluation for CarlaOcc (nuScenes-style, BEV center-distance AP).

Put this file next to carlaocc_dataset.py (pcdet/datasets/carlaocc/).

Metrics
  AP   per class, at BEV center-distance thresholds (default 0.5/1/2/4 m),
       101-point interpolation with min recall / min precision 0.1
       (nuScenes-style; precision at each recall level is taken where that level is first reached).
  mAP  mean of the per-class APs (also split into dynamic / static classes).
  ATE  mean BEV center error (m) of true positives at `tp_threshold`.
  AOE  mean absolute yaw error (rad, 0..pi) of true positives at
       `tp_threshold`; reported for dynamic classes only, because the
       orientation of poles, lights, buildings etc. is not meaningful.

Center distance is used instead of IoU because it is much fairer to thin
objects (poles, lights) whose boxes are ~0.2 m wide.
"""
import numpy as np

DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
TP_THRESHOLD = 2.0
DYNAMIC = ('car', 'truck', 'bus', 'motorcycle', 'bicycle',
           'pedestrian', 'rider', 'train')


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)


def build_prediction_dicts(batch_dict, pred_dicts, class_names):
    """One annotation dict per sample in the batch. Labels from the head are
    1-based (same convention as OpenPCDet's nuScenes dataset)."""
    names = np.asarray(class_names)
    annos = []
    for i, pd in enumerate(pred_dicts):
        labels = _np(pd['pred_labels']).astype(np.int64)
        annos.append({
            'frame_id': str(batch_dict['frame_id'][i]),
            'name': names[labels - 1] if len(labels) else np.zeros(0, dtype='<U32'),
            'score': _np(pd['pred_scores']).astype(np.float32),
            'boxes_lidar': _np(pd['pred_boxes']).astype(np.float32),
        })
    return annos


def _boxes7(b):
    b = np.asarray(b, dtype=np.float64)
    return b.reshape(-1, b.shape[-1])[:, :7] if b.size else np.zeros((0, 7))


def _ap(prec, rec, min_recall=0.1, min_precision=0.1):
    grid = np.linspace(0.0, 1.0, 101)
    # several detections can share one recall value (FPs after a TP); use the
    # precision at the FIRST point reaching it, so trailing FPs don't count
    # against a recall level that was already reached
    rec, first = np.unique(rec, return_index=True)
    p = np.interp(grid, rec, prec[first], right=0.0)
    p = p[int(round(100 * min_recall)) + 1:]        # drop recalls <= min_recall
    p = np.clip(p - min_precision, 0.0, None)       # drop precision <= min_precision
    return float(p.mean() / (1.0 - min_precision))


def _eval_class(frame_data, n_gt, thresholds, tp_thr, with_aoe):
    """frame_data: list of (det_scores (D,), det_boxes (D,7), gt_boxes (G,7)),
    only for frames that have at least one detection of this class."""
    n_det = sum(len(s) for s, _, _ in frame_data)
    if n_det == 0:
        return {'ap': {t: 0.0 for t in thresholds}, 'ate': np.nan, 'aoe': np.nan}

    scores = np.concatenate([s for s, _, _ in frame_data])
    fidx = np.concatenate([np.full(len(s), i) for i, (s, _, _) in enumerate(frame_data)])
    ridx = np.concatenate([np.arange(len(s)) for s, _, _ in frame_data])
    order = np.argsort(-scores, kind='stable')
    dists = [np.linalg.norm(db[:, None, :2] - gb[None, :, :2], axis=-1)
             for _, db, gb in frame_data]                       # (D, G) per frame

    out = {'ap': {}, 'ate': np.nan, 'aoe': np.nan}
    for thr in sorted(set(thresholds) | {tp_thr}):
        taken = [np.zeros(gb.shape[0], dtype=bool) for _, _, gb in frame_data]
        tp = np.zeros(n_det)
        errs, yaws = [], []
        for k, j in enumerate(order):
            fi, ri = fidx[j], ridx[j]
            d = dists[fi][ri]
            if d.size == 0:
                continue
            d = np.where(taken[fi], np.inf, d)      # nearest not-yet-matched GT
            g = int(np.argmin(d))
            if d[g] < thr:
                taken[fi][g] = True
                tp[k] = 1
                if thr == tp_thr:
                    errs.append(d[g])
                    if with_aoe:
                        yp = frame_data[fi][1][ri, 6]
                        yg = frame_data[fi][2][g, 6]
                        yaws.append(abs((yp - yg + np.pi) % (2 * np.pi) - np.pi))
        tp_cum = np.cumsum(tp)
        prec = tp_cum / np.arange(1, n_det + 1)
        rec = tp_cum / n_gt
        if thr in thresholds:
            out['ap'][thr] = _ap(prec, rec)
        if thr == tp_thr and errs:
            out['ate'] = float(np.mean(errs))
            if with_aoe:
                out['aoe'] = float(np.mean(yaws))
    return out


def evaluate_single_score_thresh(det_annos, gt_by_frame, class_names, pc_range,
                                 score_thresh, thresholds=DIST_THRESHOLDS, tp_threshold=TP_THRESHOLD):
    """Evaluates detections filtered by a single confidence score threshold."""
    r = np.asarray(pc_range, dtype=np.float64)

    def inside(b):
        return ((b[:, 0] >= r[0]) & (b[:, 0] <= r[3]) &
                (b[:, 1] >= r[1]) & (b[:, 1] <= r[4]))

    frames = [a for a in det_annos if a['frame_id'] in gt_by_frame]
    n_skipped = len(det_annos) - len(frames)

    per_class = {}
    for c in class_names:
        n_gt, frame_data = 0, []
        for a in frames:
            g = gt_by_frame[a['frame_id']]
            gb, gn = _boxes7(g['boxes']), np.asarray(g['name'])
            gb = gb[(gn == c) & inside(gb)] if len(gb) else gb
            n_gt += len(gb)
            
            db, dn = _boxes7(a['boxes_lidar']), np.asarray(a['name'])
            ds = np.asarray(a['score'], dtype=np.float64)
            
            # Apply class matching, pc_range, AND score_thresh filtering
            m = (dn == c) & inside(db) & (ds >= score_thresh) if len(db) else np.zeros(0, dtype=bool)
            if m.any():
                frame_data.append((ds[m], db[m], gb))
        if n_gt == 0:
            continue                                  # class absent from this split
        per_class[c] = _eval_class(frame_data, n_gt, thresholds, tp_threshold, c in DYNAMIC)

    res = {}
    for c, v in per_class.items():
        res[f'AP/{c}'] = float(np.mean(list(v['ap'].values())))
        for t in thresholds:
            res[f'AP/{c}@{t}m'] = v['ap'][t]
        if not np.isnan(v['ate']):
            res[f'ATE/{c}'] = v['ate']
        if not np.isnan(v['aoe']):
            res[f'AOE/{c}'] = v['aoe']

    def mean_ap(cs):
        vals = [res[f'AP/{c}'] for c in cs if f'AP/{c}' in res]
        return float(np.mean(vals)) if vals else float('nan')

    res['mAP'] = mean_ap(per_class)
    res['mAP_dynamic'] = mean_ap([c for c in per_class if c in DYNAMIC])
    res['mAP_static'] = mean_ap([c for c in per_class if c not in DYNAMIC])
    for t in thresholds:
        vals = [v['ap'][t] for v in per_class.values()]
        res[f'mAP@{t}m'] = float(np.mean(vals)) if vals else float('nan')

    # printable table
    hdr = f'{"class":14s}' + ''.join(f'{"AP@" + str(t) + "m":>9s}' for t in thresholds) + \
          f'{"AP":>8s}{"ATE(m)":>9s}{"AOE(rad)":>10s}'
    lines = [f"=== Evaluation with Score Threshold: {score_thresh:.1f} ===", hdr]
    for c, v in per_class.items():
        lines.append(f'{c:14s}' + ''.join(f'{v["ap"][t]:9.3f}' for t in thresholds) +
                     f'{res["AP/" + c]:8.3f}{v["ate"]:9.3f}{v["aoe"]:10.3f}')
    lines.append(f'mAP {res["mAP"]:.4f} | dynamic {res["mAP_dynamic"]:.4f} | '
                 f'static {res["mAP_static"]:.4f}   ({len(frames)} frames'
                 + (f', {n_skipped} without GT skipped' if n_skipped else '') + ')')
    return '\n'.join(lines), res


def evaluate(det_annos, gt_by_frame, class_names, pc_range,
             thresholds=DIST_THRESHOLDS, tp_threshold=TP_THRESHOLD,
             score_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """Runs evaluation for multiple score thresholds."""
    all_strs = []
    combined_results = {}

    for score_thresh in score_thresholds:
        r_str, r_dict = evaluate_single_score_thresh(
            det_annos, gt_by_frame, class_names, pc_range,
            score_thresh=score_thresh, thresholds=thresholds, tp_threshold=tp_threshold
        )
        all_strs.append(r_str)
        # Store dict metrics indexed by score threshold
        combined_results[f'score_thresh_{score_thresh:.1f}'] = r_dict

    final_str = '\n\n'.join(all_strs)
    return final_str, combined_results


def evaluate_detections(det_annos, infos, class_names, pc_range, **kwargs):
    """Entry point used by CarlaOccDataset.evaluation()."""
    wanted = {a['frame_id'] for a in det_annos}
    gt = {}
    for info in infos:
        fid = str(info['frame_id'])
        if fid in wanted:
            gt[fid] = {'name': np.asarray(info['gt_names']),
                       'boxes': np.asarray(info['gt_boxes'], dtype=np.float64).reshape(-1, 7)}
    result_str, result_dict = evaluate(det_annos, gt, class_names, pc_range)
    result_str = '\n' + result_str + '\n'
    return result_str, result_dict
