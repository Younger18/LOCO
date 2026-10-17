#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The FPPI / FPR result depends on the confidence threshold, so it is measured across
where almost the whole anchor tail survives. Report false-alarm behaviour across a
range of operating thresholds so the operational claim can be judged.

Usage:  python eval_fppi_thresholds.py <tag> <imgsz> [n_neg]
"""
import os
import sys
import json
import random

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
THRESHOLDS = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
IOU_NMS = 0.7


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'n'
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    n_neg = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    from ultralytics import YOLO

    negs = [l.strip() for l in open(os.path.join(HERE, 'negatives.txt')) if l.strip()]
    random.Random(42).shuffle(negs)
    negs = negs[:n_neg]
    print('  negatives used: %d' % len(negs), flush=True)

    models = {'random_split': os.path.join(HERE, 'runs', 'detect', '%s_random_split_%d' % (tag, imgsz), 'weights', 'best.pt')}
    for c in ['qx', 'qxg', 'zj']:            # three representative folds
        models['fold_' + c] = os.path.join(HERE, 'runs', 'detect', '%s_fold_%s_%d' % (tag, c, imgsz), 'weights', 'best.pt')

    out = {}
    for name, p in models.items():
        if not os.path.exists(p):
            print('[skip]', name, '(no weights)')
            continue
        m = YOLO(p)
        rec = {}
        for t in THRESHOLDS:
            boxes, imgs = 0, 0
            for imp in negs:
                r = m.predict(imp, imgsz=imgsz, conf=t, iou=IOU_NMS, verbose=False)[0]
                n = len(r.boxes)
                boxes += n
                if n:
                    imgs += 1
            rec['%.3f' % t] = {'FPPI': boxes / len(negs),
                               'FPR_images': imgs / len(negs),
                               'mean_boxes': boxes / len(negs)}
            print('  %-13s conf=%.3f  FPPI=%7.3f  FPR=%.4f'
                  % (name, t, rec['%.3f' % t]['FPPI'], rec['%.3f' % t]['FPR_images']), flush=True)
        out[name] = rec
        del m
    json.dump(out, open(os.path.join(HERE, 'fppi_thresholds_%s_%d.json' % (tag, imgsz)), 'w'), indent=2)
    print('[done] wrote fppi_thresholds_%s_%d.json' % (tag, imgsz))


if __name__ == '__main__':
    main()
