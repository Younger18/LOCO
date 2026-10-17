#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation for the Bijie LOCO study.

Two protocols share the IDENTICAL test set (all images of county X):
   B_seen  : model trained on a RANDOM split over all counties (has seen county X)
   C_unseen: model trained by leave-one-county-out (never saw county X)   -> unbiased
   leakage = B - C

Size stratification
-------------------
Because YOLO letterboxes every image to a fixed imgsz, the scale at which the network
actually processes an object is  area_native * s^2  with  s = min(imgsz/W, imgsz/H).
We bin by that EFFECTIVE area (COCO thresholds 32^2 / 96^2) and additionally record the
native-area distribution for reporting.

Also computes (E4) the false-positive study on the 2003 non-landslide images:
   FPPI = false-positive boxes / number of negative images
   FPR  = fraction of negative images with >=1 detection

Usage:
  python eval_loco.py loco <tag> <imgsz>
  python eval_loco.py fppi <tag> <imgsz>
"""
import os
import sys
import json
import glob

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
SMALL_MAX = 32 * 32
LARGE_MIN = 96 * 96
CONF = 0.001
IOU_NMS = 0.7


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bin_of(area):
    return 'small' if area < SMALL_MAX else ('large' if area > LARGE_MIN else 'medium')


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
    """Return {imgpath: (boxes_native, boxes_effective, size(W,H))}."""
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
        gts[imp] = (native, eff, (W, H), s)
    return gts


def predict(model, imp, imgsz):
    res = model.predict(imp, imgsz=imgsz, conf=CONF, iou=IOU_NMS, verbose=False)[0]
    d = res.boxes.data.cpu().numpy()
    return [(float(b[4]), [float(b[0]), float(b[1]), float(b[2]), float(b[3])]) for b in d]


def score(model, gts, imgsz, use_eff=True):
    """Size-stratified AP.

    IMPORTANT: ultralytics `predict()` returns boxes rescaled back to the ORIGINAL image
    coordinate system, so IoU matching MUST be done against the NATIVE boxes. The
    letterboxed ('effective') boxes are used ONLY to decide which size bin a ground-truth
    object belongs to, because the bin should reflect the scale the network processed.
    """
    bins = {'small': [], 'medium': [], 'large': [], 'all': []}
    ngt = {'small': 0, 'medium': 0, 'large': 0, 'all': 0}
    ngt_native = {'small': 0, 'medium': 0, 'large': 0}
    for imp, (native, eff, _, _) in gts.items():
        pl = predict(model, imp, imgsz)          # boxes in ORIGINAL pixel coords
        boxes = native                             # match in the same coordinate system
        bin_of_box = [bin_of((e[2] - e[0]) * (e[3] - e[1])) for e in eff] if use_eff else \
                     [bin_of((b[2] - b[0]) * (b[3] - b[1])) for b in native]
        used = [False] * len(boxes)
        for i, bb in enumerate(bin_of_box):
            ngt[bb] += 1
            ngt['all'] += 1
        nb = native
        for i in range(len(nb)):
            ngt_native[bin_of((nb[i][2] - nb[i][0]) * (nb[i][3] - nb[i][1]))] += 1
        for conf, pb in pl:
            best, bi = -1, 0.0
            for gi, gb in enumerate(boxes):
                if used[gi]:
                    continue
                v = iou(pb, gb)
                if v > bi:
                    bi, best = v, gi
            if best >= 0 and bi >= 0.5:
                used[best] = True
                bb = bin_of_box[best]
                bins['all'].append((conf, 1))
                bins[bb].append((conf, 1))
            else:
                bins['all'].append((conf, 0))
                for b in ('small', 'medium', 'large'):
                    bins[b].append((conf, 0))
    return {b: voc_ap_101(bins[b], ngt[b]) for b in bins}, ngt, ngt_native


def official(model, fold_dir, imgsz):
    data = os.path.join(HERE, fold_dir, 'data.yaml')
    met = model.val(data=data, split='test', imgsz=imgsz, verbose=False)

    def g(name):
        best = None
        for k, v in met.results_dict.items():
            if not k.endswith('(B)'):
                continue
            kl = k.lower()
            if name == 'mAP50' and ('mAP50' in kl or 'map50' in kl) and '95' not in kl and '75' not in kl:
                best = v
            elif name == 'mAP50-95' and '95' in kl and ('mAP' in kl or 'map' in kl):
                best = v
            elif name == 'P' and 'precision' in kl:
                best = v
            elif name == 'R' and 'recall' in kl:
                best = v
        return best
    return {'mAP50': g('mAP50'), 'mAP50-95': g('mAP50-95'), 'P': g('P'), 'R': g('R')}


def run_loco(tag, imgsz):
    from ultralytics import YOLO
    rnd_path = os.path.join(HERE, 'runs', 'detect', '%s_random_split_%d' % (tag, imgsz), 'weights', 'best.pt')
    if not os.path.exists(rnd_path):
        print('[missing] random-split model:', rnd_path)
        return
    R = YOLO(rnd_path)
    res = {'imgsz': imgsz, 'tag': tag, 'counties': {}}
    pb = {'small': [], 'medium': [], 'large': [], 'all': []}
    pc = {'small': [], 'medium': [], 'large': [], 'all': []}
    ngb = {'small': 0, 'medium': 0, 'large': 0, 'all': 0}
    ngc = {'small': 0, 'medium': 0, 'large': 0, 'all': 0}
    nat = {'small': 0, 'medium': 0, 'large': 0}

    for c in COUNTIES:
        lp = os.path.join(HERE, 'runs', 'detect', '%s_fold_%s_%d' % (tag, c, imgsz), 'weights', 'best.pt')
        if not os.path.exists(lp):
            print('[skip] no LOCO model for', c)
            continue
        L = YOLO(lp)
        gts = load_gt(c, imgsz)
        apB, ngB, natB = score(R, gts, imgsz)
        apC, ngC, natC = score(L, gts, imgsz)
        offB = official(R, 'fold_' + c, imgsz)
        offC = official(L, 'fold_' + c, imgsz)
        res['counties'][c] = {
            'n_test': len(gts),
            'B_seen': {'sizeAP': apB, 'official': offB},
            'C_unseen': {'sizeAP': apC, 'official': offC},
            'leakage_mAP50': (offB['mAP50'] - offC['mAP50']) if (offB['mAP50'] is not None and offC['mAP50'] is not None) else None,
            'leakage_all_AP50': apB['all'] - apC['all'],
            'gt_effective': ngC, 'gt_native': natC,
        }
        print('  %-4s n=%3d | B_all=%.4f C_all=%.4f leak=%+.4f | B_med=%.3f C_med=%.3f | B_lrg=%.3f C_lrg=%.3f'
              % (c, len(gts), apB['all'], apC['all'], apB['all'] - apC['all'],
                 apB['medium'], apC['medium'], apB['large'], apC['large']), flush=True)
        for b in ('small', 'medium', 'large', 'all'):
            # pooled curves rebuilt below from per-image lists; store counts only
            ngb[b] += ngB[b]
            ngc[b] += ngC[b]
        for b in ('small', 'medium', 'large'):
            nat[b] += natC[b]
        del L

    # pooled AP: re-run scoring across ALL counties with each county judged by its own model
    poolB = {'small': [], 'medium': [], 'large': [], 'all': []}
    poolC = {'small': [], 'medium': [], 'large': [], 'all': []}
    ngp = {'small': 0, 'medium': 0, 'large': 0, 'all': 0}
    for c in res['counties']:
        lp = os.path.join(HERE, 'runs', 'detect', '%s_fold_%s_%d' % (tag, c, imgsz), 'weights', 'best.pt')
        L = YOLO(lp)
        gts = load_gt(c, imgsz)
        for imp, (native, eff, _, _) in gts.items():
            pB = predict(R, imp, imgsz)
            pC = predict(L, imp, imgsz)
            used = [[False] * len(native), [False] * len(native)]
            bin_of_box = [bin_of((e[2] - e[0]) * (e[3] - e[1])) for e in eff]
            for i, bb in enumerate(bin_of_box):
                ngp[bb] += 1
                ngp['all'] += 1
            for k, pl in enumerate((pB, pC)):
                tgt = poolB if k == 0 else poolC
                for conf, pb in pl:
                    best, bi = -1, 0.0
                    for gi, gb in enumerate(native):     # match in ORIGINAL coords
                        if used[k][gi]:
                            continue
                        v = iou(pb, gb)
                        if v > bi:
                            bi, best = v, gi
                    if best >= 0 and bi >= 0.5:
                        used[k][best] = True
                        bb = bin_of_box[best]
                        tgt['all'].append((conf, 1))
                        tgt[bb].append((conf, 1))
                    else:
                        tgt['all'].append((conf, 0))
                        for b in ('small', 'medium', 'large'):
                            tgt[b].append((conf, 0))
        del L
    res['pooled'] = {
        'B_seen': {b: voc_ap_101(poolB[b], ngp[b]) for b in poolB},
        'C_unseen': {b: voc_ap_101(poolC[b], ngp[b]) for b in poolC},
        'gt_effective': ngp, 'gt_native': nat,
    }
    res['pooled']['leakage'] = {b: res['pooled']['B_seen'][b] - res['pooled']['C_unseen'][b]
                                for b in poolB}
    out = os.path.join(HERE, 'loco_results_%s_%d.json' % (tag, imgsz))
    json.dump(res, open(out, 'w'), indent=2)
    print('\n=== POOLED (each county judged by its own LOCO model) ===')
    print('  B_seen   :', {k: round(v, 4) for k, v in res['pooled']['B_seen'].items()})
    print('  C_unseen :', {k: round(v, 4) for k, v in res['pooled']['C_unseen'].items()})
    print('  leakage  :', {k: round(v, 4) for k, v in res['pooled']['leakage'].items()})
    print('  GT (effective @%d):' % imgsz, ngp)
    print('  GT (native)       :', nat)
    print('[done] wrote', os.path.basename(out))


def run_fppi(tag, imgsz, limit=None):
    from ultralytics import YOLO
    import random
    negs = [l.strip() for l in open(os.path.join(HERE, 'negatives.txt')) if l.strip()]
    if limit and limit < len(negs):
        random.Random(42).shuffle(negs)
        negs = negs[:limit]
        print('  (using a random subsample of %d negative patches)' % limit)
    models = {'random_split': os.path.join(HERE, 'runs', 'detect', '%s_random_split_%d' % (tag, imgsz), 'weights', 'best.pt')}
    for c in COUNTIES:
        models['fold_' + c] = os.path.join(HERE, 'runs', 'detect', '%s_fold_%s_%d' % (tag, c, imgsz), 'weights', 'best.pt')
    out = {}
    for name, p in models.items():
        if not os.path.exists(p):
            continue
        m = YOLO(p)
        fp_boxes = 0
        fp_images = 0
        for imp in negs:
            r = predict(m, imp, imgsz)
            fp_boxes += len(r)
            if r:
                fp_images += 1
        out[name] = {'n_images': len(negs), 'fp_boxes': fp_boxes,
                     'FPPI': fp_boxes / len(negs), 'FPR_images': fp_images / len(negs)}
        print('  %-14s FPPI=%.4f  FPR=%.4f  (fp_boxes=%d / %d)'
              % (name, out[name]['FPPI'], out[name]['FPR_images'], fp_boxes, len(negs)), flush=True)
        del m
    json.dump(out, open(os.path.join(HERE, 'fppi_%s_%d.json' % (tag, imgsz)), 'w'), indent=2)
    print('[done] wrote fppi_%s_%d.json' % (tag, imgsz))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'loco'
    tag = sys.argv[2] if len(sys.argv) > 2 else 'n'
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    if mode == 'loco':
        run_loco(tag, imgsz)
    else:
        run_fppi(tag, imgsz, limit)
