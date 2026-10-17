#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the P2 high-resolution detection branch under the
leave-one-county-out protocol on Bijie, so that the P2 claim is supported by a
direct experiment instead of by prior work on a redistributed variant.

Compares, on identical LOCO test images and paired over the 8 counties:
    C_n   : YOLO26n, never saw the test county
    C_p2  : YOLO26n + stride-4 (P2) detection level, never saw the test county
    B_n / B_p2 : the corresponding random-split models (saw every county)

Also reports, for each architecture, its own B - C leakage, so that the question
"does P2 also reduce the leakage?" can be answered.
"""
import os
import json
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
RNG = np.random.default_rng(42)


def paired(d, label):
    d = np.asarray(d, dtype=float)
    n = len(d)
    try:
        p = float(wilcoxon(d, alternative='greater').pvalue)
    except Exception:  # noqa: BLE001
        p = float('nan')
    bs = RNG.choice(d, size=(10000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('  %-34s mean=%+.4f  sd=%.4f  sign=%d+/%d-  p=%.4f  boot95=[%+.4f, %+.4f]'
          % (label, d.mean(), d.std(ddof=1), int((d > 0).sum()), int((d < 0).sum()), p, lo, hi))
    return {'label': label, 'mean': float(d.mean()), 'sd': float(d.std(ddof=1)),
            'n_pos': int((d > 0).sum()), 'n_neg': int((d < 0).sum()),
            'wilcoxon_p': p, 'boot95': [float(lo), float(hi)],
            'per_fold': [float(x) for x in d]}


def main():
    fp_n = os.path.join(HERE, 'loco_results_n_256.json')
    fp_p2 = os.path.join(HERE, 'loco_results_p2_256.json')
    miss = [p for p in (fp_n, fp_p2) if not os.path.exists(p)]
    if miss:
        print('[missing]', miss)
        print('  run:  python eval_loco.py loco p2 256')
        return
    N = json.load(open(fp_n))['counties']
    P = json.load(open(fp_p2))['counties']
    Nfull = json.load(open(fp_n))
    Pfull = json.load(open(fp_p2))

    out = {}
    print('=' * 78)
    print('A. ARCHITECTURE COMPARISON UNDER LOCO (C models only, identical test images)')
    print('=' * 78)
    rows = []
    for c in COUNTIES:
        r = {'county': c,
             'n_mAP50': N[c]['C_unseen']['official']['mAP50'],
             'p2_mAP50': P[c]['C_unseen']['official']['mAP50'],
             'n_mAP50_95': N[c]['C_unseen']['official']['mAP50-95'],
             'p2_mAP50_95': P[c]['C_unseen']['official']['mAP50-95']}
        r['d_mAP50'] = r['p2_mAP50'] - r['n_mAP50']
        r['d_mAP50_95'] = r['p2_mAP50_95'] - r['n_mAP50_95']
        rows.append(r)
        print('  %-4s n=%-4d n: mAP50=%.4f mAP50-95=%.4f | p2: mAP50=%.4f mAP50-95=%.4f | d %+.4f / %+.4f'
              % (c, N[c]['n_test'], r['n_mAP50'], r['n_mAP50_95'],
                 r['p2_mAP50'], r['p2_mAP50_95'], r['d_mAP50'], r['d_mAP50_95']))
    out['per_county'] = rows
    print()
    out['paired'] = [
        paired([r['d_mAP50_95'] for r in rows], 'P2 - n : mAP50-95 (LOCO)'),
        paired([r['d_mAP50'] for r in rows], 'P2 - n : mAP50 (LOCO)'),
        paired([P[c]['C_unseen']['sizeAP']['medium'] - N[c]['C_unseen']['sizeAP']['medium'] for c in COUNTIES],
               'P2 - n : medium-object AP'),
        paired([P[c]['C_unseen']['sizeAP']['large'] - N[c]['C_unseen']['sizeAP']['large'] for c in COUNTIES],
               'P2 - n : large-object AP'),
    ]

    print()
    print('=' * 78)
    print('B. POOLED SIZE-STRATIFIED AP@0.5 UNDER LOCO')
    print('=' * 78)
    for tag, D in (('n', Nfull['pooled']), ('p2', Pfull['pooled'])):
        print('  %-4s C_unseen %s' % (tag, {k: round(v, 4) for k, v in D['C_unseen'].items()}))
    print('  GT (effective @256):', Nfull['pooled']['gt_effective'])
    out['pooled'] = {'n': Nfull['pooled'], 'p2': Pfull['pooled']}

    print()
    print('=' * 78)
    print('C. DOES P2 CHANGE THE LEAKAGE ITSELF?  (B - C for each architecture)')
    print('=' * 78)
    for tag, D in (('n', N), ('p2', P)):
        d50 = [D[c]['B_seen']['official']['mAP50'] - D[c]['C_unseen']['official']['mAP50'] for c in COUNTIES]
        d95 = [D[c]['B_seen']['official']['mAP50-95'] - D[c]['C_unseen']['official']['mAP50-95'] for c in COUNTIES]
        print('  %-4s B-C: mAP50 mean=%+.4f (sd %.4f)   mAP50-95 mean=%+.4f (sd %.4f)'
              % (tag, np.mean(d50), np.std(d50, ddof=1), np.mean(d95), np.std(d95, ddof=1)))
        out.setdefault('leakage_by_arch', {})[tag] = {
            'mAP50': [float(x) for x in d50], 'mAP50_95': [float(x) for x in d95],
            'mAP50_mean': float(np.mean(d50)), 'mAP50_95_mean': float(np.mean(d95))}
    d = [out['leakage_by_arch']['p2']['mAP50_95'][i] - out['leakage_by_arch']['n']['mAP50_95'][i]
         for i in range(len(COUNTIES))]
    paired(d, 'leakage(P2) - leakage(n), mAP50-95')

    json.dump(out, open(os.path.join(HERE, 'p2_comparison.json'), 'w'), indent=2)
    print()
    print('[done] wrote p2_comparison.json')


if __name__ == '__main__':
    main()
