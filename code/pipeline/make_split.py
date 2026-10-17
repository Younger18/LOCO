#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E0 — build the YOLO datasets for the Bijie leave-one-county-out (LOCO) study.

Design (per research_plan.md v2)
--------------------------------
* input   : ONLY the RGB images (no DEM) + polygon boundary coordinates
* classes : single class `landslide`
* counties: 8 units;  zdf (n=1) merged into df
* folds   : 8. For fold X -> test = ALL images of county X;
            the remaining 7 counties are split train/val = 85/15 (seed 42)
* random  : one random split over all 770 images (train 70 / val 10 / test 20, seed 42)
            used to reproduce the "optimistic" protocol of Ji et al. (2020, 2:1)
* negatives: file list of the 2003 non-landslide images (for the FPPI / false-alarm study)

Polygon -> YOLO box: one polygon per file; box = axis-aligned bbox of the polygon,
written as `0 cx cy w h` with values normalised to [0, 1] and clamped.
"""
import os
import re
import json
import random
import shutil
import collections

ROOT = r'bijie-landslide-dataset'
OUT = r'.'
SEED = 42

IMG = os.path.join(ROOT, 'landslide', 'image')
POLY = os.path.join(ROOT, 'landslide', 'polygon_coordinate')
NEG = os.path.join(ROOT, 'non-landslide', 'image')

COUNTIES = {'qx': '黔西市 Qianxi', 'qxg': '七星关区 Qixingguan', 'zj': '织金县 Zhijin',
            'hz': '赫章县 Hezhang', 'js': '金沙县 Jinsha', 'ny': '纳雍县 Nayong',
            'wn': '威宁彝族回族苗族自治县 Weining', 'df': '大方县 Dafang'}
MERGE = {'zdf': 'df'}


def prefix_of(stem):
    m = re.match(r'^([a-zA-Z]+)', stem)
    return m.group(1).lower() if m else '?'


def read_polygon(stem):
    """Return (xmin, ymin, xmax, ymax) of the polygon in native pixel coords."""
    p = os.path.join(POLY, stem + '.txt')
    pts = []
    with open(p) as fh:
        for line in fh:
            m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$', line.strip())
            if m:
                pts.append((float(m.group(1)), float(m.group(2))))
    if len(pts) < 3:
        return None
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]
    return min(xs), min(ys), max(xs), max(ys)


def yolo_line(box, W, H):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return '0 %.6f %.6f %.6f %.6f' % (cx, cy, bw, bh)


def emit(dst, stems, sizes, boxes):
    """Copy images + write YOLO labels under dst/{images,labels}/{train,val,test}."""
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(dst, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dst, 'labels', split), exist_ok=True)
    for split, names in stems.items():
        for s in names:
            src = os.path.join(IMG, s + '.png')
            shutil.copy2(src, os.path.join(dst, 'images', split, s + '.png'))
            W, H = sizes[s]
            with open(os.path.join(dst, 'labels', split, s + '.txt'), 'w') as fh:
                fh.write(yolo_line(boxes[s], W, H) + '\n')
    with open(os.path.join(dst, 'data.yaml'), 'w', encoding='utf-8') as fh:
        fh.write('path: %s\n' % dst.replace('\\', '/'))
        fh.write('train: images/train\nval: images/val\ntest: images/test\n')
        fh.write('nc: 1\nnames: [\'landslide\']\n')


def main():
    from PIL import Image

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMG) if f.endswith('.png'))
    sizes, boxes, county = {}, {}, {}
    for s in stems:
        im = Image.open(os.path.join(IMG, s + '.png'))
        sizes[s] = im.size
        b = read_polygon(s)
        if b is None:
            continue
        boxes[s] = b
        p = prefix_of(s)
        county[s] = MERGE.get(p, p)

    stems = [s for s in stems if s in boxes]
    by_county = collections.defaultdict(list)
    for s in stems:
        by_county[county[s]].append(s)

    manifest = {'n': len(stems), 'counties': {}, 'folds': {}, 'random': {}}
    for k in sorted(by_county, key=lambda x: -len(by_county[x])):
        manifest['counties'][k] = {'name_cn': COUNTIES.get(k, k), 'n': len(by_county[k])}

    rng = random.Random(SEED)

    # ---------------- LOCO folds ----------------
    for c in sorted(by_county):
        test = sorted(by_county[c])
        rest = sorted(s for s in stems if county[s] != c)
        rng.shuffle(rest)
        n_val = max(int(round(0.15 * len(rest))), 10)
        val = rest[:n_val]
        train = rest[n_val:]
        dst = os.path.join(OUT, 'fold_' + c)
        emit(dst, {'train': train, 'val': val, 'test': test}, sizes, boxes)
        manifest['folds'][c] = {'test_county': c, 'name_cn': COUNTIES.get(c, c),
                                'n_test': len(test), 'n_val': len(val), 'n_train': len(train)}

    # ---------------- random split (reproduces the optimistic protocol) -------------
    all_s = sorted(stems)
    rng2 = random.Random(SEED)
    rng2.shuffle(all_s)
    n_test = max(int(round(0.20 * len(all_s))), 1)
    n_val = max(int(round(0.10 * len(all_s))), 1)
    test_r = all_s[:n_test]
    val_r = all_s[n_test:n_test + n_val]
    train_r = all_s[n_test + n_val:]
    dst = os.path.join(OUT, 'random_split')
    emit(dst, {'train': train_r, 'val': val_r, 'test': test_r}, sizes, boxes)
    cnt = collections.Counter(county[s] for s in test_r)
    manifest['random'] = {'n_train': len(train_r), 'n_val': len(val_r), 'n_test': len(test_r),
                          'test_county_composition': dict(cnt)}

    # ---------------- negatives file list (FPPI study) ----------------
    negs = sorted(os.path.splitext(f)[0] for f in os.listdir(NEG) if f.endswith('.png'))
    with open(os.path.join(OUT, 'negatives.txt'), 'w') as fh:
        for s in negs:
            fh.write(os.path.join(NEG, s + '.png') + '\n')
    manifest['negatives'] = {'n': len(negs)}

    json.dump(manifest, open(os.path.join(OUT, 'loco_manifest.json'), 'w'),
              indent=2, ensure_ascii=False)

    print('total samples: %d' % len(stems))
    print('\ncounties (test-set size per LOCO fold):')
    for k in sorted(manifest['counties'], key=lambda x: -manifest['counties'][x]['n']):
        print('  %-5s %-30s %4d' % (k, manifest['counties'][k]['name_cn'], manifest['counties'][k]['n']))
    print('\nfold sizes:')
    for c, d in manifest['folds'].items():
        print('  fold_%-4s train=%3d val=%3d test=%3d' % (c, d['n_train'], d['n_val'], d['n_test']))
    print('\nrandom split: train=%d val=%d test=%d' % (manifest['random']['n_train'],
                                                       manifest['random']['n_val'],
                                                       manifest['random']['n_test']))
    print('negatives: %d' % manifest['negatives']['n'])
    print('\n[done] wrote loco_manifest.json')


if __name__ == '__main__':
    main()
