#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train one model on one Bijie split.

Usage: python train_loco.py <tag> <datadir> [imgsz] [epochs] [batch]

  tag      'n'  -> yolo26n          ; 'p2' -> yolo26n-p2 (+ partial transfer from yolo26n.pt)
  datadir  'fold_qx' ... 'fold_df'  or 'random_split'
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
YM = r'yolo_models_root\yolo_models'
PROJECT = os.path.join(HERE, 'runs', 'detect')


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'n'
    datadir = sys.argv[2] if len(sys.argv) > 2 else 'fold_qx'
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    batch = int(sys.argv[5]) if len(sys.argv) > 5 else 32

    from ultralytics import YOLO

    data = os.path.join(HERE, datadir, 'data.yaml')
    if tag == 'p2':
        model = YOLO('yolo26n-p2.yaml')
        pre = os.path.join(YM, 'yolo26n.pt')
        try:
            model.load(pre)
            print('[p2] partial transfer from yolo26n.pt OK', flush=True)
        except Exception as e:  # noqa: BLE001
            print('[p2] transfer failed, train from scratch:', e, flush=True)
    elif tag == 'v8n':
        model = YOLO(os.path.join(YM, 'yolov8n.pt')) if os.path.exists(
            os.path.join(YM, 'yolov8n.pt')) else YOLO('yolov8n.pt')
    else:
        model = YOLO(os.path.join(YM, 'yolo26n.pt'))

    name = '%s_%s_%d' % (tag, datadir, imgsz)
    print('===== TRAIN tag=%s data=%s imgsz=%d epochs=%d batch=%d ====='
          % (tag, datadir, imgsz, epochs, batch), flush=True)
    model.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch, device=0, seed=42,
                name=name, patience=30, close_mosaic=10, workers=4,
                project=PROJECT, exist_ok=True)
    print('===== DONE %s =====' % name, flush=True)


if __name__ == '__main__':
    main()
