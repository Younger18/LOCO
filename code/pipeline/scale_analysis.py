#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
How should we stratify landslide size?  A data-driven assessment for the Bijie dataset.

Two distinct questions are often conflated:
  Q1. Which threshold SCHEME is defensible?   (COCO pixel convention vs. metric vs. data-driven)
  Q2. What is the size-bin composition AFTER letterbox resizing to the training imgsz?
      -> YOLO resizes every image to a fixed imgsz, so the imgsz choice ITSELF shifts the
         size distribution. This is usually ignored but is decisive here.

Ground sample distance is constant (TripleSat, 0.8 m/px), so pixel and metric thresholds
are equivalent: area_m2 = area_px2 * 0.8^2 = area_px2 * 0.64
"""
import os
import re
import json
import collections

import numpy as np
from PIL import Image

ROOT = r'bijie-landslide-dataset'
OUT = r'.'
GSD = 0.8  # m per pixel


def prefix_of(stem):
    m = re.match(r'^([a-zA-Z]+)', stem)
    return m.group(1).lower() if m else '?'


def load():
    """Return list of (stem, W, H, bbox_w, bbox_h) for landslide samples; zdf merged into df."""
    idir = os.path.join(ROOT, 'landslide', 'image')
    pdir = os.path.join(ROOT, 'landslide', 'polygon_coordinate')
    recs = []
    for f in sorted(os.listdir(idir)):
        stem = os.path.splitext(f)[0]
        W, H = Image.open(os.path.join(idir, f)).size
        pts = []
        with open(os.path.join(pdir, stem + '.txt')) as fh:
            for line in fh:
                m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$', line.strip())
                if m:
                    pts.append((float(m.group(1)), float(m.group(2))))
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        pref = prefix_of(stem)
        if pref == 'zdf':
            pref = 'df'          # merge the single anomalous sample into Dafang
        recs.append((stem, pref, W, H, bw, bh))
    return recs


def bin_counts(areas, small_max, large_min):
    small = int(np.sum(areas < small_max))
    large = int(np.sum(areas > large_min))
    medium = int(len(areas) - small - large)
    return small, medium, large


def main():
    recs = load()
    n = len(recs)
    W = np.array([r[2] for r in recs], float)
    H = np.array([r[3] for r in recs], float)
    bw = np.array([r[4] for r in recs], float)
    bh = np.array([r[5] for r in recs], float)
    area = bw * bh
    pref = [r[1] for r in recs]

    print('=== A. dataset after merging zdf -> df (n=%d) ===' % n)
    cnt = collections.Counter(pref)
    for k, v in cnt.most_common():
        print('   %-6s %4d' % (k, v))

    print('\n=== B. native (unresized) object statistics ===')
    print('   bbox area px2   p10=%.0f  median=%.0f  p90=%.0f  mean=%.0f  max=%.0f'
          % (np.percentile(area, 10), np.median(area), np.percentile(area, 90), area.mean(), area.max()))
    print('   bbox side px    p10=%.1f  median=%.1f  p90=%.1f'
          % (np.percentile(np.sqrt(area), 10), np.median(np.sqrt(area)), np.percentile(np.sqrt(area), 90)))
    print('   in METRES (GSD=0.8): side median=%.1f m, area median=%.0f m2'
          % (np.median(np.sqrt(area)) * GSD, np.median(area) * GSD ** 2))

    print('\n=== C. candidate pixel-threshold schemes (applied at NATIVE scale) ===')
    schemes = [
        ('COCO standard   32^2 / 96^2', 32 ** 2, 96 ** 2),
        ('smaller         24^2 / 64^2', 24 ** 2, 64 ** 2),
        ('much smaller    16^2 / 48^2', 16 ** 2, 48 ** 2),
        ('larger          48^2 /128^2', 48 ** 2, 128 ** 2),
        ('metric ~500/2000 m2', (500 / (GSD ** 2)), (2000 / (GSD ** 2))),
        ('metric ~1000/5000 m2', (1000 / (GSD ** 2)), (5000 / (GSD ** 2))),
    ]
    rows = []
    for name, sm, lg in schemes:
        s, m, l = bin_counts(area, sm, lg)
        rows.append((name, sm, lg, s, m, l))
        print('   %-26s thr=(%6.0f,%7.0f)  small=%3d  medium=%3d  large=%3d' % (name, sm, lg, s, m, l))

    print('\n=== D. effect of the training imgsz (letterbox) on the size distribution ===')
    print('    (YOLO letterboxes every image to imgsz; scale = min(S/W, S/H))')
    print('    %-8s %-28s %s' % ('imgsz', 'small/medium/large', 'median side after resize'))
    img_sizes = [256, 320, 384, 512]
    img_rows = []
    for S in img_sizes:
        sc = np.minimum(S / W, S / H)
        a2 = area * sc ** 2
        s, m, l = bin_counts(a2, 32 ** 2, 96 ** 2)
        img_rows.append({'imgsz': S, 'small': s, 'medium': m, 'large': l,
                         'median_side_px': float(np.median(np.sqrt(a2))),
                         'mean_scale': float(sc.mean())})
        print('    %-8d %-28s %.1f px   (mean up/down-scale %.2fx)'
              % (S, '%d / %d / %d' % (s, m, l), np.median(np.sqrt(a2)), sc.mean()))
    print('\n    NOTE: at imgsz=512 the median object is inflated ~%.1fx in area vs native,'
          % ((512 / 240.) ** 2))
    print('    which is exactly the artefact that made Abbott-max look "large-dominated".')

    print('\n=== E. minimum-bin reliability check ===')
    print('    Rule of thumb: >=50 instances per bin for a stable AP estimate.')
    for name, sm, lg, s, m, l in rows:
        worst = min(s, m, l)
        flag = 'OK' if worst >= 50 else ('THIN' if worst >= 20 else 'UNRELIABLE')
        print('    %-26s min bin = %3d  -> %s' % (name, worst, flag))

    json.dump({'n': n, 'prefix_counts': dict(cnt),
               'native_bbox_area_px2': {'p10': float(np.percentile(area, 10)),
                                        'median': float(np.median(area)),
                                        'p90': float(np.percentile(area, 90)),
                                        'mean': float(area.mean()), 'max': float(area.max())},
               'schemes': [{'name': r[0], 'small_thr': r[1], 'large_thr': r[2],
                            'small': r[3], 'medium': r[4], 'large': r[5]} for r in rows],
               'imgsz_effect': img_rows},
              open(os.path.join(OUT, 'scale_analysis.json'), 'w'), indent=2, ensure_ascii=False)
    print('\n[done] wrote scale_analysis.json')


if __name__ == '__main__':
    main()
