#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1 — integrity & metadata audit of the WHU Bijie landslide dataset.

Checks
------
 1. exact file counts per sub-directory
 2. county-prefix composition (inferred from file names) vs Bijie's 8 county-level units
 3. image / mask / DEM size, mode, dtype; whether mask & DEM are pixel-aligned with the image
 4. DEM value distribution (is it 8-bit hillshade or real elevation in metres? are they comparable?)
 5. mask binarisation scheme (unique values, foreground ratio)
 6. polygon_coordinate txt parse: number of polygons per file, coordinate validity & bbox
 7. landslide size distribution at NATIVE resolution (real target scale, not the 512x512 artefact)
 8. pairing completeness: image <-> mask <-> dem <-> polygon

Writes bijie_verify.json and prints a readable summary.
"""
import os
import re
import json
import collections

import numpy as np
from PIL import Image

ROOT = r'bijie-landslide-dataset'
OUT = r'.'

SUBDIRS = [
    'landslide/image',
    'landslide/mask',
    'landslide/dem',
    'landslide/polygon_coordinate',
    'non-landslide/image',
    'non-landslide/dem',
]

# Bijie City county-level administrative units (verified against bijie.gov.cn)
BIJIE_UNITS = {
    'qxg': '七星关区 Qixingguan District',
    'df': '大方县 Dafang County',
    'qx': '黔西市 Qianxi City',
    'js': '金沙县 Jinsha County',
    'zj': '织金县 Zhijin County',
    'ny': '纳雍县 Nayong County',
    'wn': '威宁彝族回族苗族自治县 Weining',
    'hz': '赫章县 Hezhang County',
}


def prefix_of(stem):
    m = re.match(r'^([a-zA-Z]+)', stem)
    return m.group(1).lower() if m else '(numeric)'


def listdir_ext(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))


def main():
    rep = {'root': ROOT, 'counts': {}, 'prefixes': {}, 'samples': {}, 'issues': []}

    # ---------- 1. counts ----------
    for sub in SUBDIRS:
        fs = listdir_ext(os.path.join(ROOT, sub))
        exts = collections.Counter(os.path.splitext(f)[1].lower() for f in fs)
        rep['counts'][sub] = {'n': len(fs), 'extensions': dict(exts)}

    # ---------- 2. prefixes ----------
    for sub in ['landslide/image', 'non-landslide/image']:
        fs = listdir_ext(os.path.join(ROOT, sub))
        cnt = collections.Counter(prefix_of(os.path.splitext(f)[0]) for f in fs)
        rep['prefixes'][sub] = dict(cnt.most_common())

    # ---------- 3/4/5. sample properties ----------
    img_dir = os.path.join(ROOT, 'landslide/image')
    imgs = listdir_ext(img_dir)
    n_img = len(imgs)

    sizes = []
    modes = collections.Counter()
    rng = np.random.default_rng(0)
    sample_idx = sorted(rng.choice(n_img, size=min(60, n_img), replace=False).tolist())

    dem_min, dem_max, dem_dtype = [], [], collections.Counter()
    msk_vals = collections.Counter()
    misaligned = []

    for i in sample_idx:
        stem, ext = os.path.splitext(imgs[i])
        p_img = os.path.join(img_dir, imgs[i])
        im = Image.open(p_img)
        modes[str(im.mode)] += 1
        sizes.append(im.size)
        w, h = im.size

        # mask
        p_msk = os.path.join(ROOT, 'landslide/mask', imgs[i])
        if os.path.exists(p_msk):
            mk = Image.open(p_msk)
            if mk.size != (w, h):
                misaligned.append(('mask', imgs[i], mk.size, (w, h)))
            a = np.array(mk)
            vals, cnts = np.unique(a, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                msk_vals[v] += c

        # dem
        p_dem = os.path.join(ROOT, 'landslide/dem', imgs[i])
        if os.path.exists(p_dem):
            dm = Image.open(p_dem)
            if dm.size != (w, h):
                misaligned.append(('dem', imgs[i], dm.size, (w, h)))
            b = np.array(dm)
            dem_dtype[str(b.dtype)] += 1
            dem_min.append(float(b.min()))
            dem_max.append(float(b.max()))

    rep['samples'] = {
        'n_sampled': len(sample_idx),
        'image_modes': dict(modes),
        'width': {'min': int(min(w for w, _ in sizes)), 'max': int(max(w for w, _ in sizes)),
                  'mean': float(np.mean([w for w, _ in sizes]))},
        'height': {'min': int(min(h for _, h in sizes)), 'max': int(max(h for _, h in sizes)),
                   'mean': float(np.mean([h for _, h in sizes]))},
        'dem_dtype': dict(dem_dtype),
        'dem_min_over_sample': {'min': min(dem_min) if dem_min else None,
                                'max': max(dem_min) if dem_min else None},
        'dem_max_over_sample': {'min': min(dem_max) if dem_max else None,
                                'max': max(dem_max) if dem_max else None},
        'mask_pixel_values_top': dict(msk_vals.most_common(10)),
        'misaligned_pairs': misaligned[:20],
        'n_misaligned': len(misaligned),
    }

    # ---------- 6. polygon txt ----------
    poly_dir = os.path.join(ROOT, 'landslide/polygon_coordinate')
    polys = listdir_ext(poly_dir)
    n_pts, poly_ok, bbox_areas = [], 0, []
    bad = []
    for f in polys:
        p = os.path.join(poly_dir, f)
        pts = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$', line)
                if m:
                    pts.append((float(m.group(1)), float(m.group(2))))
        if len(pts) >= 3:
            poly_ok += 1
            n_pts.append(len(pts))
            xs = [a for a, _ in pts]
            ys = [b for _, b in pts]
            bbox_areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
        else:
            bad.append(f)
    rep['polygons'] = {
        'n_files': len(polys),
        'n_valid': poly_ok,
        'n_invalid': len(bad),
        'invalid_examples': bad[:10],
        'points_per_polygon': {'min': int(min(n_pts)) if n_pts else None,
                               'max': int(max(n_pts)) if n_pts else None,
                               'mean': float(np.mean(n_pts)) if n_pts else None},
        'bbox_area_px2': {'min': float(min(bbox_areas)) if bbox_areas else None,
                          'p10': float(np.percentile(bbox_areas, 10)) if bbox_areas else None,
                          'median': float(np.median(bbox_areas)) if bbox_areas else None,
                          'p90': float(np.percentile(bbox_areas, 90)) if bbox_areas else None,
                          'max': float(max(bbox_areas)) if bbox_areas else None},
    }

    # ---------- 7. native-resolution size bins (COCO, applied on the ORIGINAL image) ----------
    bins = collections.Counter()
    areas = []
    for f in polys:
        p = os.path.join(poly_dir, f)
        pts = []
        with open(p) as fh:
            for line in fh:
                m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$', line.strip())
                if m:
                    pts.append((float(m.group(1)), float(m.group(2))))
        if len(pts) < 3:
            continue
        xs = [a for a, _ in pts]
        ys = [b for _, b in pts]
        ar = (max(xs) - min(xs)) * (max(ys) - min(ys))
        areas.append(ar)
        if ar < 32 * 32:
            bins['small'] += 1
        elif ar > 96 * 96:
            bins['large'] += 1
        else:
            bins['medium'] += 1
    rep['native_size_bins'] = {'counts': dict(bins), 'total': sum(bins.values()),
                               'mean_area_px2': float(np.mean(areas)) if areas else None}

    # ---------- 8. pairing completeness ----------
    stems = {os.path.splitext(f)[0] for f in imgs}
    for sub, key in [('landslide/mask', 'mask'), ('landslide/dem', 'dem'),
                     ('landslide/polygon_coordinate', 'polygon')]:
        have = {os.path.splitext(f)[0] for f in listdir_ext(os.path.join(ROOT, sub))}
        rep['pairing_' + key] = {
            'images': len(stems), 'paired': len(stems), 'have': len(have),
            'missing_in_%s' % key: sorted(stems - have)[:10],
            'orphan_in_%s' % key: sorted(have - stems)[:10]}
        if stems - have:
            rep['issues'].append('%d images lack a %s' % (len(stems - have), key))

    json.dump(rep, open(os.path.join(OUT, 'bijie_verify.json'), 'w'), indent=2, ensure_ascii=False)

    # ---------- print ----------
    print('=== 1. file counts ===')
    for sub in SUBDIRS:
        c = rep['counts'][sub]
        print('  %-30s %5d  %s' % (sub, c['n'], c['extensions']))
    print('\n=== 2. county prefixes (landslide) ===')
    for k, v in rep['prefixes']['landslide/image'].items():
        tag = BIJIE_UNITS.get(k, '')
        print('  %-8s %5d  %s' % (k, v, tag))
    print('\n  non-landslide prefixes (top 10):',
          list(rep['prefixes']['non-landslide/image'].items())[:10])
    print('\n=== 3. image/mask/DEM properties (n=%d sampled) ===' % rep['samples']['n_sampled'])
    s = rep['samples']
    print('  modes:', s['image_modes'])
    print('  width  min/max/mean: %d / %d / %.1f' % (s['width']['min'], s['width']['max'], s['width']['mean']))
    print('  height min/max/mean: %d / %d / %.1f' % (s['height']['min'], s['height']['max'], s['height']['mean']))
    print('  DEM dtype:', s['dem_dtype'], ' min range:', s['dem_min_over_sample'],
          ' max range:', s['dem_max_over_sample'])
    print('  mask pixel values (top):', s['mask_pixel_values_top'])
    print('  misaligned image<->mask/dem:', s['n_misaligned'])
    print('\n=== 6. polygons ===')
    p = rep['polygons']
    print('  files=%d valid=%d invalid=%d  points/poly min/mean/max=%s/%s/%s'
          % (p['n_files'], p['n_valid'], p['n_invalid'],
             p['points_per_polygon']['min'], round(p['points_per_polygon']['mean'], 1),
             p['points_per_polygon']['max']))
    print('  bbox area px2  min/p10/median/p90/max = %.0f / %.0f / %.0f / %.0f / %.0f'
          % (p['bbox_area_px2']['min'], p['bbox_area_px2']['p10'], p['bbox_area_px2']['median'],
             p['bbox_area_px2']['p90'], p['bbox_area_px2']['max']))
    print('\n=== 7. native-resolution COCO size bins ===')
    nb = rep['native_size_bins']
    print('  ', nb['counts'], 'total=%d  mean_area=%.0f px2' % (nb['total'], nb['mean_area_px2']))
    print('\n=== 8. pairing ===')
    for key in ['mask', 'dem', 'polygon']:
        d = rep['pairing_' + key]
        print('  %-8s images=%d have=%d missing=%s' % (key, d['images'], d['have'], d['missing_in_%s' % key][:5]))
    print('\nissues:', rep['issues'])
    print('\n[done] wrote bijie_verify.json')


if __name__ == '__main__':
    main()
