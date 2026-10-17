#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-seed replication for the headline LOCO leakage.
comment 1: seed-level variance not quantified).

Trains B (random_split) + the 8 LOCO folds (tag 'n') at each requested seed,
using hyper-parameters IDENTICAL to train_loco.py (imgsz 256, epochs 200,
batch 32, AdamW defaults, patience 30, close_mosaic 10). Only `seed` changes.
Results are written to run names carrying a _s<seed> suffix so the original
seed-42 models are never overwritten.

Usage: python train_multiseed.py <seed0> [seed1 ...]
   e.g. python train_multiseed.py 0 7
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YM = r'yolo_models_root\yolo_models'
PROJECT = os.path.join(HERE, 'runs', 'detect')
DATADIRS = ['random_split', 'fold_qx', 'fold_qxg', 'fold_zj', 'fold_hz',
            'fold_js', 'fold_ny', 'fold_wn', 'fold_df']


def train_one(datadir, seed, imgsz=256, epochs=200, batch=32):
    from ultralytics import YOLO
    data = os.path.join(HERE, datadir, 'data.yaml')
    model = YOLO(os.path.join(YM, 'yolo26n.pt'))
    name = 'n_%s_%d_s%d' % (datadir, imgsz, seed)
    print('===== TRAIN seed=%d data=%s =====' % (seed, datadir), flush=True)
    model.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, device=0,
                seed=seed, name=name, patience=30, close_mosaic=10, workers=4,
                project=PROJECT, exist_ok=True)
    print('===== DONE %s =====' % name, flush=True)


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [0]
    for seed in seeds:
        for dd in DATADIRS:
            train_one(dd, seed)


if __name__ == '__main__':
    main()
