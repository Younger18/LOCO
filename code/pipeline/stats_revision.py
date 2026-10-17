#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary statistics reported alongside the headline results.

(1) Per-fold training summary (epochs completed, best epoch, in-domain val metrics)
    -> addresses R2-M3 (convergence) and R2-M4 (saturation ceiling reference)
(2) Paired leakage statistics: per-fold differences, sign count, exact Wilcoxon
    signed-rank test, bootstrap 95% CI of the mean leakage
    -> addresses R1-M3 (no uncertainty / significance on the headline number)
(3) Domain-shift gap = in-domain val mAP50-95 - LOCO-test mAP50-95 for the SAME model
    -> addresses R1-M4 (B vs C confound: training-set size vs test-domain exposure)
"""
import os
import json
import csv
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
RNG = np.random.default_rng(42)


def col(row, key):
    try:
        return float(row[key])
    except Exception:  # noqa: BLE001
        return float('nan')


def training_summary(tag, name):
    p = os.path.join(HERE, 'runs', 'detect', name, 'results.csv')
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    ep = [int(col(r, 'epoch')) for r in rows]
    m95 = np.array([col(r, 'metrics/mAP50-95(B)') for r in rows])
    m50 = np.array([col(r, 'metrics/mAP50(B)') for r in rows])
    bi = int(np.nanargmax(m95))
    return {
        'epochs_run': len(rows),
        'last_epoch': ep[-1],
        'best_epoch': ep[bi],
        'val_mAP50_best': float(m50[bi]),
        'val_mAP50_95_best': float(m95[bi]),
        'val_mAP50_final': float(m50[-1]),
        'val_mAP50_95_final': float(m95[-1]),
        'wall_s': float(col(rows[-1], 'time')),
    }


def paired_stats(diff, label, out):
    d = np.asarray(diff, dtype=float)
    n = len(d)
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    mean = float(d.mean())
    try:
        w = wilcoxon(d, alternative='greater')
        p = float(w.pvalue)
    except Exception:  # noqa: BLE001
        p = float('nan')
    # bootstrap CI of the mean (resample the 8 folds)
    bs = RNG.choice(d, size=(10000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rec = {'label': label, 'n': n, 'mean': mean, 'sd': float(d.std(ddof=1)),
           'n_positive': n_pos, 'n_negative': n_neg,
           'wilcoxon_p_greater': p,
           'boot95': [float(lo), float(hi)],
           'per_fold': [float(x) for x in d]}
    out.append(rec)
    print('  %-26s mean=%+.4f  sd=%.4f  sign=%d+/%d-  Wilcoxon p=%.4f  boot95=[%+.4f, %+.4f]'
          % (label, mean, rec['sd'], n_pos, n_neg, p, lo, hi))


def main():
    print('=' * 78)
    print('(1) PER-FOLD TRAINING SUMMARY')
    print('=' * 78)
    tsum = {}
    for nm, d in [('random_split', 'n_random_split_256')] + [('fold_' + c, 'n_fold_%s_256' % c) for c in COUNTIES]:
        s = training_summary('n', d)
        tsum[nm] = s
        if s:
            print('  %-12s epochs=%3d (last=%3d) best@%3d  in-domain val mAP50=%.4f mAP50-95=%.4f | final mAP50-95=%.4f | %.0f s'
                  % (nm, s['epochs_run'], s['last_epoch'], s['best_epoch'],
                     s['val_mAP50_best'], s['val_mAP50_95_best'], s['val_mAP50_95_final'], s['wall_s']))
    early = sum(1 for s in tsum.values() if s and s['epochs_run'] <= 175)
    print('  -> early-stopped folds (<=175 of 200 epochs): %d/9' % early)

    print()
    print('=' * 78)
    print('(2) PAIRED LEAKAGE STATISTICS (n = 8 counties, paired over identical test images)')
    print('=' * 78)
    det = json.load(open(os.path.join(HERE, 'loco_results_n_256.json')))['counties']
    cls = json.load(open(os.path.join(HERE, 'cls_leakage.json')))['counties']
    out = []
    paired_stats([cls[c]['leak']['AP'] for c in COUNTIES], 'classification AP', out)
    paired_stats([cls[c]['leak']['F1'] for c in COUNTIES], 'classification F1', out)
    paired_stats([cls[c]['leak']['TPR'] for c in COUNTIES], 'classification recall', out)
    paired_stats([-cls[c]['leak']['FPR'] for c in COUNTIES], 'classification FPR (rise)', out)
    paired_stats([det[c]['B_seen']['official']['mAP50'] - det[c]['C_unseen']['official']['mAP50'] for c in COUNTIES],
                 'detection mAP50', out)
    paired_stats([det[c]['B_seen']['official']['mAP50-95'] - det[c]['C_unseen']['official']['mAP50-95'] for c in COUNTIES],
                 'detection mAP50-95', out)
    paired_stats([det[c]['leakage_all_AP50'] for c in COUNTIES], 'detection sizeAP-all@0.5', out)

    print()
    print('=' * 78)
    print('(3) DOMAIN-SHIFT GAP WITHIN THE SAME MODEL (LOCO model only)')
    print('    in-domain val (15%% of the 7 training counties) vs out-of-domain test (held-out county)')
    print('=' * 78)
    gaps = []
    for c in COUNTIES:
        s = tsum['fold_' + c]
        if not s:
            continue
        t50 = det[c]['C_unseen']['official']['mAP50']
        t95 = det[c]['C_unseen']['official']['mAP50-95']
        g50 = s['val_mAP50_best'] - t50
        g95 = s['val_mAP50_95_best'] - t95
        gaps.append((g50, g95))
        print('  %-4s in-domain  mAP50=%.4f mAP50-95=%.4f  |  out-domain mAP50=%.4f mAP50-95=%.4f  |  gap %+.4f / %+.4f'
              % (c, s['val_mAP50_best'], s['val_mAP50_95_best'], t50, t95, g50, g95))
    g = np.array(gaps)
    print('  mean gap: mAP50 %+.4f   mAP50-95 %+.4f   (ratio %.1fx)'
          % (g[:, 0].mean(), g[:, 1].mean(), g[:, 1].mean() / max(g[:, 0].mean(), 1e-9)))
    print('  same-model domain-shift is only visible in the stricter metric,'
          ' consistent with the B-vs-C result.')

    json.dump({'training_summary': tsum, 'paired': out,
               'domain_shift_gap': {'per_fold': [[float(a), float(b)] for a, b in gaps],
                                    'mean_mAP50': float(g[:, 0].mean()),
                                    'mean_mAP50_95': float(g[:, 1].mean())}},
              open(os.path.join(HERE, 'supp_stats.json'), 'w'), indent=2)
    print()
    print('[done] wrote supp_stats.json')


if __name__ == '__main__':
    main()
