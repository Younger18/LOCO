#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 7 v2: false positives per image for all nine detection models on
the full set of 2003 non-landslide patches, at the confidence floor
0.001 used throughout the paper.

Redraw of fig7_fppi_models.png with one hard constraint: the figure is
rendered at its FINAL on-page width of 8 cm (1:1, no down-scaling at
placement), and every text element is exactly 6 pt, so that when placed
at 8 cm wide all text reads at 6 pt. Saved WITHOUT bbox_inches='tight'
so the PNG width is exactly 8 cm (the original used tight bbox, which
would let the width float with content and break the 1:1 guarantee).

The long x-axis label is hand-wrapped into three lines so it fits the
8 cm width at 6 pt. Y-axis tick labels drop the "LOCO, " prefix (just
the county name + "held out"); the legend is set to 4 pt and placed at
the lower-left corner of the WHOLE figure (outside the axes, in a
reserved bottom band via fig.legend), so it never overlaps the bars,
value labels, or the x-axis label. The plot keeps the original 7.4:4.3
aspect; only a ~0.6 cm bottom band is added for the legend.

Saved as paper/figs/fig7_fppi_models_v2.png ONLY (600 dpi). The original
fig7_fppi_models.png is left untouched, so the earlier figure is
unaffected.
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

COUNTY_LABEL = {'qx': 'Qianxi', 'qxg': 'Qixingguan', 'zj': 'Zhijin', 'hz': 'Hezhang',
                'js': 'Jinsha', 'ny': 'Nayong', 'wn': 'Weining', 'df': 'Dafang'}

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
    D = json.load(open(os.path.join(HERE, 'fppi_full_n_256.json')))
    rows = []
    for name, rec in D.items():
        if name == 'random_split':
            lab = 'random split (B)'
            col = C_B
        else:
            c = name.replace('fold_', '')
            lab = '%s held out' % COUNTY_LABEL.get(c, c)
            col = C_C
        rows.append((rec['FPPI'], lab, col, rec))
    rows.sort(key=lambda r: r[0])                      # ascending: smallest at the bottom
    fppi = [r[0] for r in rows]
    labs = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    fpr = [r[3]['FPR_images'] for r in rows]

    y = np.arange(len(rows))
    CM = 1.0 / 2.54
    FW = 8.0 * CM                       # 3.1496 in -> exactly 8 cm at save
    FH = FW * (4.3 / 7.4) + 0.6 * CM    # plot at original 7.4:4.3 aspect + 0.6 cm bottom band for the figure legend
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.barh(y, fppi, 0.62, color=cols)
    for yi, v in zip(y, fppi):
        ax.text(v + 0.25, yi, '%.2f' % v, va='center', fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6)
    ax.set_xlim(0, 25.5)
    ax.set_xlabel('false-positive boxes per image (FPPI)\n'
                  'at confidence floor 0.001, 2003 negative patches\n'
                  '(image-level false-alarm rate %.3f-%.3f, all nine models)'
                  % (min(fpr), max(fpr)))
    from matplotlib.patches import Patch
    handles = [Patch(color=C_B, label='random split (county seen)'),
               Patch(color=C_C, label='LOCO folds (county unseen)')]
    fig.tight_layout(rect=[0, 0.14, 1, 1])   # reserve a bottom band for the figure legend
    fig.legend(handles=handles, loc='lower left', fontsize=4,
               frameon=True, framealpha=0.9, edgecolor='0.6')
    out = os.path.join(FIGS, 'fig7_fppi_models_v2.png')
    fig.savefig(out, facecolor='white')      # no bbox_inches='tight': keep width exactly 8 cm
    plt.close(fig)
    print('  wrote', out)
    for yi, v, l in zip(y, fppi, labs):
        print('    %-28s %.4f' % (l, v))


if __name__ == '__main__':
    main()
