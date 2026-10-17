#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combine four single-panel figures into one 2x2 journal-grade illustration.

Sources (single-panel, landscape):
  fig3  leakage by metric          -> panel (a)
  fig5  size-stratified AP50       -> panel (b)
  fig7  FPPI across 9 models       -> panel (c)
  fig11 seed robustness           -> panel (d)

Each source is flattened onto white, padded to a common aspect (1.92) that
matches the subplot axes so imshow fills without distortion.  TNR bold
(a)-(d) labels sit in the top-left corner of every panel.  Saved as a
standalone PNG (single-column width, 300 dpi); NOT embedded in the text.

Output: paper/figs/fig_combined_multimetric.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)

# panel order -> (filename, label)
PANELS = [
    ('fig3_leakage_by_metric.png', 'a'),
    ('fig5_size_stratified.png', 'b'),
    ('fig7_fppi_models.png', 'c'),
    ('fig11_seed_robustness.png', 'd'),
]

# Matches the subplot axes aspect computed from figsize + margins so imshow
# fills exactly with no distortion.
TARGET_ASPECT = 1.92


def load_white(path):
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert('RGB')


def pad_to_aspect(im, aspect):
    w, h = im.size
    if (w / h) > aspect:                 # too wide -> pad top/bottom
        new_h = int(round(w / aspect))
        pad = new_h - h
        top, bot = pad // 2, pad - pad // 2
        return ImageOps.expand(im, border=(0, top, 0, bot), fill=(255, 255, 255))
    else:                                # too tall -> pad left/right
        new_w = int(round(h * aspect))
        pad = new_w - w
        left, right = pad // 2, pad - pad // 2
        return ImageOps.expand(im, border=(left, 0, right, 0), fill=(255, 255, 255))


def main():
    imgs = []
    for fn, _ in PANELS:
        p = os.path.join(FIGS, fn)
        if not os.path.exists(p):
            raise FileNotFoundError('missing source: ' + p)
        imgs.append(np.asarray(pad_to_aspect(load_white(p), TARGET_ASPECT)))

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'savefig.facecolor': 'white',
    })

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 4.47))
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (fn, lbl), (r, c), im in zip(PANELS, pos, imgs):
        ax = axes[r][c]
        ax.imshow(im)
        ax.axis('off')

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98,
                        bottom=0.02, wspace=0.03, hspace=0.05)

    out = os.path.join(FIGS, 'fig_combined_multimetric.png')
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print('  wrote', out, '(%dx%d px)' % (Image.open(out).size))


if __name__ == '__main__':
    main()
