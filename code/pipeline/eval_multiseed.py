#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seed-variance of the headline LOCO leakage.

For each seed, evaluates B (random-split model) and C (per-fold LOCO model) on
the IDENTICAL test images of every county with the official Ultralytics
validator (the source of Tables 5 and 6), then reports the B-C leakage
averaged over the 8 counties. Aggregating over seeds gives the seed-level
mean +/- SD of each headline leakage number.

Handles the original seed-42 run names (n_<dir>_256) and the multi-seed names
(n_<dir>_256_s<seed>).

Usage: python eval_multiseed.py 42 0 7
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
IMGSZ = 256


def wp(seed, datadir):
    if seed == 42:
        return os.path.join(HERE, 'runs', 'detect', 'n_%s_%d' % (datadir, IMGSZ),
                            'weights', 'best.pt')
    return os.path.join(HERE, 'runs', 'detect', 'n_%s_%d_s%d' % (datadir, IMGSZ, seed),
                        'weights', 'best.pt')


def official(model, fold, imgsz=IMGSZ):
    data = os.path.join(HERE, 'fold_' + fold, 'data.yaml')
    met = model.val(data=data, split='test', imgsz=imgsz, device=0, verbose=False)
    rd = met.results_dict
    def pick(sub):
        for k, v in rd.items():
            kl = k.lower()
            if sub == 'mAP50' and ('map50' in kl) and '95' not in kl and '75' not in kl:
                return v
            if sub == 'mAP50-95' and '95' in kl and 'map' in kl:
                return v
        return None
    return {'mAP50': pick('mAP50'), 'mAP50-95': pick('mAP50-95')}


def eval_seed(seed):
    from ultralytics import YOLO
    Rp = wp(seed, 'random_split')
    if not os.path.exists(Rp):
        print('[missing] seed %d random-split model' % seed)
        return None
    R = YOLO(Rp)
    rows = {}
    for c in COUNTIES:
        Cp = wp(seed, 'fold_' + c)
        if not os.path.exists(Cp):
            print('[missing] seed %d fold %s' % (seed, c))
            continue
        L = YOLO(Cp)
        b = official(R, c)
        cc = official(L, c)
        rows[c] = {
            'B_mAP50': b['mAP50'], 'C_mAP50': cc['mAP50'],
            'B_mAP50_95': b['mAP50-95'], 'C_mAP50_95': cc['mAP50-95'],
            'leak_mAP50': (b['mAP50'] - cc['mAP50']) if None not in (b['mAP50'], cc['mAP50']) else None,
            'leak_mAP50_95': (b['mAP50-95'] - cc['mAP50-95']) if None not in (b['mAP50-95'], cc['mAP50-95']) else None,
        }
        del L
    del R
    lm50 = [rows[c]['leak_mAP50'] for c in rows if rows[c]['leak_mAP50'] is not None]
    lm95 = [rows[c]['leak_mAP50_95'] for c in rows if rows[c]['leak_mAP50_95'] is not None]
    return {
        'per_county': rows,
        'mean_leak_mAP50': float(np.mean(lm50)) if lm50 else None,
        'mean_leak_mAP50_95': float(np.mean(lm95)) if lm95 else None,
        'sd_leak_mAP50': float(np.std(lm50, ddof=1)) if len(lm50) > 1 else None,
        'sd_leak_mAP50_95': float(np.std(lm95, ddof=1)) if len(lm95) > 1 else None,
    }


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [42, 0, 7]
    res = {'seeds': seeds, 'per_seed': {}}
    for s in seeds:
        r = eval_seed(s)
        if r:
            res['per_seed'][s] = r
            print('seed %2d | leak mAP50 = %+.4f (+-%.4f) | mAP50-95 = %+.4f (+-%.4f)'
                  % (s, r['mean_leak_mAP50'], r['sd_leak_mAP50'] or 0,
                     r['mean_leak_mAP50_95'], r['sd_leak_mAP50_95'] or 0), flush=True)
    for key in ('mean_leak_mAP50', 'mean_leak_mAP50_95'):
        vals = [res['per_seed'][s][key] for s in res['per_seed']
                if res['per_seed'][s][key] is not None]
        if len(vals) > 1:
            res[key + '_across_seeds'] = {
                'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)),
                'min': float(np.min(vals)), 'max': float(np.max(vals)), 'n': len(vals)}
    out = os.path.join(HERE, 'multiseed_leakage.json')
    json.dump(res, open(out, 'w'), indent=2)
    print('[done] wrote multiseed_leakage.json')
    for key in ('mean_leak_mAP50', 'mean_leak_mAP50_95'):
        if key + '_across_seeds' in res:
            d = res[key + '_across_seeds']
            print('  %s across seeds: %.4f +/- %.4f (range %.4f..%.4f, n=%d)'
                  % (key, d['mean'], d['sd'], d['min'], d['max'], d['n']))


if __name__ == '__main__':
    main()
