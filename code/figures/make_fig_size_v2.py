#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5 v2: pooled size-stratified AP50 (redraw of fig5_size_stratified.png).

Hard constraint: rendered at FINAL on-page width 8 cm (1:1, no
down-scaling at placement); every main text element is exactly 6 pt so
it reads 6 pt at 8 cm. Saved WITHOUT bbox_inches='tight' so the PNG
width is exactly 8 cm.

The legend is stacked in TWO rows (ncol=1) and placed in the blank
space above the FIRST (small) column, whose bars are tiny (~0.03), so
that corner is empty. A bbox self-check auto-picks the largest legend
font size (from 6 pt down) that does NOT overlap any bar, value label,
or leakage label, satisfying "adjust its size so it never covers the
graph".

Output: paper/figs/fig5_size_stratified_v2.png (600 dpi) ONLY. The
original fig5_size_stratified.png is left untouched for comparison.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 600, 'font.size': 6,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white',
})
C_B, C_C = '#1f77b4', '#d62728'


def main():
    P = json.load(open(os.path.join(HERE, 'loco_results_n_256.json')))['pooled']
    bins = ['small', 'medium', 'large', 'all']
    labels = ['small (< 32\u00b2 px\u00b2)\nn = 13 *', 'medium (32\u00b2-96\u00b2 px\u00b2)\nn = 421',
              'large (> 96\u00b2 px\u00b2)\nn = 336', 'all objects\nn = 770']
    B = [P['B_seen'][b] for b in bins]
    C = [P['C_unseen'][b] for b in bins]
    L = [P['leakage'][b] for b in bins]

    x = np.arange(len(bins))
    w = 0.30
    CM = 1.0 / 2.54
    FW = 8.0 * CM                       # 3.1496 in -> exactly 8 cm at save
    FH = FW * (3.7 / 7.4)               # 1.575 in -> matches original 7.4:3.7 aspect
    fig, ax = plt.subplots(figsize=(FW, FH))
    bars_b = ax.bar(x - w / 2, B, w, color=C_B, label='B (county seen)')
    bars_c = ax.bar(x + w / 2, C, w, color=C_C, label='C (county unseen)')
    ann = []
    for xi, v in zip(x - w / 2, B):
        ann.append(ax.text(xi, v + 0.012, '%.4f' % v, ha='center', fontsize=4))
    for xi, v in zip(x + w / 2, C):
        ann.append(ax.text(xi, v + 0.012, '%.4f' % v, ha='center', fontsize=4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel('pooled AP50 (IoU 0.50,\nconfidence floor 0.001)')
    fig.tight_layout()

    # leakage (+labels): push 2 pt further above the value labels below them.
    # Convert 2 pt to data-y using the (post-tight_layout) axes height.
    fig.canvas.draw()
    _axh_pt = ax.get_window_extent().height / fig.dpi * 72.0
    _shift = 2.0 * (1.13 / _axh_pt)
    for xi, v, b, c in zip(x, L, B, C):
        ann.append(ax.text(xi, max(b, c) + 0.062 + _shift, '%+.4f' % v, ha='center',
                           fontsize=6, color='#333333'))

    # everything the legend must not overlap: bars + value/leakage labels
    blockers = list(ann) + list(bars_b.patches) + list(bars_c.patches)
    inv = ax.transData.inverted()

    def overlaps(leg):
        fig.canvas.draw()
        lb = leg.get_window_extent().transformed(inv)
        hits = []
        for a in blockers:
            ab = a.get_window_extent().transformed(inv)
            if lb.overlaps(ab):
                hits.append(a.get_text() if hasattr(a, 'get_text') else 'bar')
        return lb, hits

    # auto-pick the largest legend font (>=3 pt) with zero overlap
    best = None
    for fs in [6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0]:
        leg = ax.legend(fontsize=fs, loc='upper left', bbox_to_anchor=(0.012, 0.985),
                        ncol=1, frameon=True, framealpha=0.9, edgecolor='0.6')
        lb, hits = overlaps(leg)
        leg.remove()
        if not hits:
            best = fs
            print('  legend fontsize %.1f pt -> NO overlap (bbox x[%.3f,%.3f] y[%.3f,%.3f])'
                  % (fs, lb.x0, lb.x1, lb.y0, lb.y1))
            break
        print('  legend fontsize %.1f pt -> overlaps %s; trying smaller' % (fs, hits))
    FS = best if best is not None else 3.0
    ax.legend(fontsize=FS, loc='upper left', bbox_to_anchor=(0.012, 0.985),
              ncol=1, frameon=True, framealpha=0.9, edgecolor='0.6')

    out = os.path.join(FIGS, 'fig5_size_stratified_v2.png')
    fig.savefig(out, facecolor='white')      # no tight bbox: keep width exactly 8 cm
    plt.close(fig)
    print('  wrote', out, '| final legend fontsize', FS)


if __name__ == '__main__':
    main()
