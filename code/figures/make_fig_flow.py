#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2 (new): end-to-end methodology flow chart.

Layout: the main pipeline runs left to right along the top band (audited
dataset -> the two split protocols -> identical training -> evaluation on
identical test images -> the paired B - C contrast); an elbow arrow drops
into the lower band, which carries the controls, the robustness analyses and
the conclusion. Lane colours are consistent across figures: blue = B
(county seen), red = C (county unseen).

No figure-level title inside the image (the caption lives below the figure).
All text Times New Roman, 300 dpi, white background.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False,
    'savefig.facecolor': 'white',
})
INK = '#1a1a1a'
EDGE = '#666666'
C_B, C_C = '#1f77b4', '#d62728'


def box(ax, x, y, w, h, title, lines, fill, edge, title_size=8.6, body_size=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=0.9',
                                linewidth=1.1, edgecolor=edge, facecolor=fill))
    cx = x + w / 2.0
    ty = y + h - 2.1
    ax.text(cx, ty, title, ha='center', va='top', fontsize=title_size,
            fontweight='bold', color=INK)
    ty -= 2.55
    for ln in lines:
        if ln:
            ax.text(cx, ty, ln, ha='center', va='top', fontsize=body_size, color=INK)
        ty -= 2.05


def arrow(ax, p0, p1):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='-|>', mutation_scale=11, lw=1.1,
                                color='#555555', shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(10.0, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis('off')

    # ---------------- top band: the main pipeline ----------------
    box(ax, 1.5, 36, 15, 24, 'Bijie benchmark', [
        '(audited release)',
        '',
        '770 landslide patches,',
        'one instance each',
        '2003 non-landslide',
        'patches; county',
        'attribution from file',
        'names; no coordinates,',
        'no buffer zone',
    ], fill='#f2f2f2', edge=EDGE)

    box(ax, 20.5, 49, 19, 11, 'Protocol B \u2014 random split', [
        '(reference protocol), seed 42',
        '70/10/20; test spans all 8',
        'counties \u2192 county SEEN',
    ], fill='#dbe9f6', edge=C_B)

    box(ax, 20.5, 36, 19, 11, 'Protocol C \u2014 LOCO', [
        '8 folds, seed 42; 512-624',
        'train / 15 % val / all of',
        'county X \u2192 county UNSEEN',
    ], fill='#f9e2e2', edge=C_C)

    box(ax, 43.5, 36, 15, 24, 'Identical training', [
        '',
        'YOLO26n, 256 px',
        'AdamW, \u2264200 epochs,',
        'early stopping',
        '(patience 30); plus a',
        'ResNet-18 classifier',
        'replication',
    ], fill='#f2f2f2', edge=EDGE)

    box(ax, 62, 36, 15, 24, 'Evaluation', [
        'on identical test images',
        '',
        'mAP50, mAP50-95,',
        'size-stratified AP50,',
        'classification AP / F1 /',
        'TPR / FPR, FPPI and',
        'image-level FPR at six',
        'thresholds',
    ], fill='#f2f2f2', edge=EDGE)

    box(ax, 81, 36, 17.5, 24, 'Paired contrast', [
        'B \u2212 C per county on the',
        'same test images: leakage',
        'with bootstrap 95 % CI',
        'and exact Wilcoxon',
        'signed-rank p-value',
    ], fill='#e8eef5', edge='#33517a')

    arrow(ax, (16.9, 54.5), (20.1, 54.5))
    arrow(ax, (16.9, 41.5), (20.1, 41.5))
    arrow(ax, (39.9, 54.5), (43.1, 54.5))
    arrow(ax, (39.9, 41.5), (43.1, 41.5))
    arrow(ax, (58.9, 48), (61.6, 48))
    arrow(ax, (77.4, 48), (80.6, 48))

    # ---------------- lower band: controls, robustness, conclusion ----------------
    box(ax, 6, 4, 28, 16, 'Controls against a training-set confound', [
        '',
        'C1: in-domain parity of the two families',
        'C2: within-C, in-domain vs withheld county',
    ], fill='white', edge=EDGE)

    box(ax, 38, 4, 28, 16, 'Robustness of the account', [
        '',
        'IoU sweep 0.50-0.95; training seeds 0 / 7 / 42;',
        'P2 branch replication under LOCO',
    ], fill='white', edge=EDGE)

    box(ax, 70, 4, 28.5, 16, 'Conclusion', [
        '',
        'the reported metric decides whether',
        'cross-county leakage is visible at all',
    ], fill='#f5f5f5', edge='#333333')

    # elbow arrow: contrast box -> lower band
    ax.plot([89.75, 89.75], [35.6, 28], color='#555555', lw=1.1)
    ax.plot([89.75, 20], [28, 28], color='#555555', lw=1.1)
    arrow(ax, (20, 28), (20, 20.6))
    arrow(ax, (34.4, 12), (37.6, 12))
    arrow(ax, (66.4, 12), (69.6, 12))

    out = os.path.join(FIGS, 'fig2_flow.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    main()
