#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AP as a function of the IoU matching threshold, to test the mAP50 "saturation"
mechanism directly.

Protocol is IDENTICAL to eval_loco.py:
  * confidence floor 0.001, NMS IoU 0.7, imgsz 256
  * predictions matched to ground truth in the ORIGINAL pixel coordinate system
  * 101-point interpolated VOC average precision
  * pooled over all 770 instances; B = random-split model, C = per-fold LOCO model
The ONLY thing varied is the AP *matching* IoU threshold (0.50 ... 0.95).
Predictions do not depend on that threshold, so they are cached once and the
AP is recomputed cheaply at every IoU level.

If mAP50 is "saturated" because low-IoU boxes are trivially satisfied, the B-C
gap should GROW monotonically with the IoU threshold -- that is the mechanism
the study asserts but does not yet measure.

Usage: python eval_iou_sweep.py [tag] [imgsz]
"""
import os
import sys
import json
import glob

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
CONF = 0.001
IOU_NMS = 0.7
IOUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def voc_ap_101(scored, n_gt):
    if n_gt == 0:
        return 0.0
    scored.sort(key=lambda x: -x[0])
    tp = np.array([s[1] for s in scored], dtype=float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-8)
    return float(sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0.0
                     for t in np.linspace(0, 1, 101)) / 101.0)


def load_gt(fold, imgsz):
    idir = os.path.join(HERE, 'fold_' + fold, 'images', 'test')
    ldir = os.path.join(HERE, 'fold_' + fold, 'labels', 'test')
    gts = {}
    for imp in sorted(glob.glob(os.path.join(idir, '*.png'))):
        stem = os.path.splitext(os.path.basename(imp))[0]
        W, H = Image.open(imp).size
        s = min(imgsz / W, imgsz / H)
        native, eff = [], []
        lp = os.path.join(ldir, stem + '.txt')
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                _, cx, cy, nw, nh = (float(x) for x in p[:5])
                x1, y1 = (cx - nw / 2) * W, (cy - nh / 2) * H
                x2, y2 = (cx + nw / 2) * W, (cy + nh / 2) * H
                native.append([x1, y1, x2, y2])
                eff.append([x1 * s, y1 * s, x2 * s, y2 * s])
        gts[imp] = (native, eff)
    return gts


def predict(model, imp, imgsz):
    res = model.predict(imp, imgsz=imgsz, conf=CONF, iou=IOU_NMS,
                        device=0, verbose=False)[0]
    d = res.boxes.data.cpu().numpy()
    return [(float(b[4]), [float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            for b in d]


def match_tp_fp(preds, gts, thr):
    """Per-image greedy matching at IoU `thr`; returns [(conf, is_tp), ...]."""
    used = [False] * len(gts)
    out = []
    for conf, pb in preds:
        best, bi = -1, 0.0
        for gi, gb in enumerate(gts):
            if used[gi]:
                continue
            v = iou(pb, gb)
            if v > bi:
                bi, best = v, gi
        if best >= 0 and bi >= thr:
            used[best] = True
            out.append((conf, 1))
        else:
            out.append((conf, 0))
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'n'
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    from ultralytics import YOLO

    R = YOLO(os.path.join(HERE, 'runs', 'detect',
                          '%s_random_split_%d' % (tag, imgsz), 'weights', 'best.pt'))
    # ---- cache predictions once (B and C on every test image of every county)
    per_county = {}          # c -> list of (native, predsB, predsC)
    for c in COUNTIES:
        L = YOLO(os.path.join(HERE, 'runs', 'detect',
                              '%s_fold_%s_%d' % (tag, c, imgsz), 'weights', 'best.pt'))
        gts = load_gt(c, imgsz)
        rows = []
        for imp, (native, eff) in gts.items():
            pB = predict(R, imp, imgsz)
            pC = predict(L, imp, imgsz)
            rows.append((native, pB, pC))
        per_county[c] = rows
        print('  cached %-4s : %d images' % (c, len(rows)), flush=True)
        del L

    # ---- recompute pooled AP at every IoU threshold
    results = {'imgsz': imgsz, 'tag': tag, 'ious': IOUS, 'curve': []}
    for thr in IOUS:
        poolB, poolC, ngt = [], [], 0
        for c in COUNTIES:
            for native, pB, pC in per_county[c]:
                ngt += len(native)
                poolB += match_tp_fp(pB, native, thr)
                poolC += match_tp_fp(pC, native, thr)
        apB = voc_ap_101(poolB, ngt)
        apC = voc_ap_101(poolC, ngt)
        results['curve'].append({'iou': thr, 'B': apB, 'C': apC,
                                 'leak': apB - apC, 'n_gt': ngt})
        print('IoU %.2f | B=%.4f  C=%.4f  leak=%+.4f  (n_gt=%d)'
              % (thr, apB, apC, apB - apC, ngt), flush=True)

    out = os.path.join(HERE, 'iou_sweep_%s_%d.json' % (tag, imgsz))
    json.dump(results, open(out, 'w'), indent=2)
    print('[done] wrote', os.path.basename(out))


if __name__ == '__main__':
    main()
