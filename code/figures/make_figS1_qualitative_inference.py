# -*- coding: utf-8 -*-
"""Three-panel qualitative figure: original patch / ground-truth label / YOLO26n detection.

Protocol B (random split), seed 42, model = runs/detect/n_random_split_256/weights/best.pt
Test set = random_split/images/test (154 patches, one landslide box each).
"""
import os
import glob
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = r'.'
IMGDIR = os.path.join(ROOT, 'random_split', 'images', 'test')
LBLDIR = os.path.join(ROOT, 'random_split', 'labels', 'test')
WEIGHTS = os.path.join(ROOT, 'runs', 'detect', 'n_random_split_256', 'weights', 'best.pt')
OUTDIR = os.path.join(ROOT, 'figs')
os.makedirs(OUTDIR, exist_ok=True)

CONF = 0.25          # YOLO default
IOU_TH = 0.50        # TP/FP/FN criterion, consistent with the paper's IoU sweep

# ---------------------------------------------------------------- run inference
model = YOLO(WEIGHTS)
files = sorted(glob.glob(os.path.join(IMGDIR, '*.png')))
results = model(files, conf=CONF, iou=0.7, verbose=False, save=False, device=0)

recs = []
for f, r in zip(files, results):
    stem = os.path.splitext(os.path.basename(f))[0]
    # ground truth (YOLO normalized xywh)
    gt = []
    lp = os.path.join(LBLDIR, stem + '.txt')
    if os.path.exists(lp):
        for line in open(lp).read().strip().split('\n'):
            p = line.split()
            if len(p) >= 5:
                gt.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    W, H = r.orig_shape[1], r.orig_shape[0]
    pred = []
    for b in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        pred.append({'xyxy': [x1, y1, x2, y2], 'conf': float(b.conf[0])})
    # match GT <-> pred by IoU (greedy, highest IoU first)
    used = set()
    matches = []
    if gt and pred:
        ious = []
        for gi, g in enumerate(gt):
            gx1, gy1 = (g[0] - g[2] / 2) * W, (g[1] - g[3] / 2) * H
            gx2, gy2 = (g[0] + g[2] / 2) * W, (g[1] + g[3] / 2) * H
            garea = max(0, gx2 - gx1) * max(0, gy2 - gy1)
            for pi, p in enumerate(pred):
                px1, py1, px2, py2 = p['xyxy']
                ix1, iy1 = max(gx1, px1), max(gy1, py1)
                ix2, iy2 = min(gx2, px2), min(gy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                parea = max(0, px2 - px1) * max(0, py2 - py1)
                union = garea + parea - inter
                ious.append((inter / union if union > 0 else 0.0, gi, pi))
        ious.sort(reverse=True)
        for iou, gi, pi in ious:
            if iou >= IOU_TH and gi not in used and pi not in used:
                used.add(gi)
                matches.append((gi, pi, iou))
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    recs.append({
        'stem': stem, 'path': f, 'W': W, 'H': H, 'gt': gt, 'pred': pred,
        'tp': tp, 'fp': fp, 'fn': fn, 'matches': matches,
        'best_iou': max([m[2] for m in matches], default=0.0),
        'top_conf': max([p['conf'] for p in pred], default=0.0),
    })

with open(os.path.join(OUTDIR, '_qual_cases_raw.json'), 'w', encoding='utf-8') as fh:
    json.dump(recs, fh, ensure_ascii=False, indent=1)

n_tp = sum(r['tp'] for r in recs)
n_fp = sum(r['fp'] for r in recs)
n_fn = sum(r['fn'] for r in recs)
print(f'inference done on {len(recs)} test patches @ conf {CONF}')
print(f'  TP={n_tp} FP={n_fp} FN={n_fn}')
perfect = [r for r in recs if r['fp'] == 0 and r['fn'] == 0]
print(f'  clean TP-only patches: {len(perfect)}')
print('  top-10 by conf (clean):',
      [(r['stem'], round(r['top_conf'], 3), round(r['best_iou'], 3), r['W']) for r in
       sorted(perfect, key=lambda x: -x['top_conf'])[:10]])
