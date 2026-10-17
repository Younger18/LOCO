#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4 v2: B vs C per county, mAP50 vs mAP50-95 (panels a-b).

Redraw of fig4_per_county.png with ONE change: the legend is moved from
subplot (a) lower-left to subplot (b) UPPER-RIGHT. Everything else
(data, 9.6x3.5 layout, fonts, 300 dpi, tight bbox, TNR) is identical to
make_figures.fig_per_county so the panels stay consistent.

A bbox self-check reports whether the (b) legend overlaps any bar.

Output: paper/figs/fig4_per_county_v2.png ONLY (not inserted into the
document; the original fig4_per_county.png is left untouched).
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

COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
COUNTY_LABEL = {'qx': 'Qianxi', 'qxg': 'Qixingguan', 'zj': 'Zhijin', 'hz': 'Hezhang',
                'js': 'Jinsha', 'ny': 'Nayong', 'wn': 'Weining', 'df': 'Dafang'}

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 10,
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
    D = json.load(open(os.path.join(HERE, 'loco_results_n_256.json')))['counties']
    x = np.arange(len(COUNTIES))
    b50 = [D[c]['B_seen']['official']['mAP50'] for c in COUNTIES]
    c50 = [D[c]['C_unseen']['official']['mAP50'] for c in COUNTIES]
    b95 = [D[c]['B_seen']['official']['mAP50-95'] for c in COUNTIES]
    c95 = [D[c]['C_unseen']['official']['mAP50-95'] for c in COUNTIES]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=False)
    specs = [(b50, c50, 'mAP50 (saturated)'), (b95, c95, 'mAP50-95 (sensitive)')]
    for k, (ax, (bb, cc, ttl)) in enumerate(zip(axes, specs)):
        w = 0.38
        ax.bar(x - w / 2, bb, w, color=C_B, label='B (county seen)')
        ax.bar(x + w / 2, cc, w, color=C_C, label='C (county withheld)')
        ax.set_xticks(x)
        ax.set_xticklabels([COUNTY_LABEL[c] for c in COUNTIES], rotation=35, ha='right', fontsize=8)
        ax.set_title('(%s) %s' % ('ab'[k], ttl), fontsize=9)
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel('score')
    # legend moved to subplot (b) upper-right (was subplot (a) lower-left)
    leg = axes[1].legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    # bbox self-check: does the (b) legend overlap any bar in subplot (b)?
    fig.canvas.draw()
    inv = axes[1].transData.inverted()
    lb = leg.get_window_extent().transformed(inv)
    hit = [p for p in axes[1].patches
           if lb.overlaps(p.get_window_extent().transformed(inv))]
    print('  subplot(b) legend bbox x[%.3f, %.3f] y[%.3f, %.3f] -> overlaps %d bar(s)'
          % (lb.x0, lb.x1, lb.y0, lb.y1, len(hit)))

    out = os.path.join(FIGS, 'fig4_per_county_v2.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    main()
