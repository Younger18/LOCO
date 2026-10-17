#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 7 (new, replaces Table 10): false positives per image for all nine
detection models on the full set of 2003 non-landslide patches, at the
confidence floor 0.001 used throughout the paper.

All nine models were trained WITHOUT a single negative sample, so they are
equally negative-naive: the spread between them reflects threshold and
calibration behaviour, not spatial leakage. The image-level false-alarm rate
is 0.997-1.000 for every model.

Data: fppi_full_n_256.json (full 2003-negative run of eval_loco.py fppi).
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
    D = json.load(open(os.path.join(HERE, 'fppi_full_n_256.json')))
    rows = []
    for name, rec in D.items():
        if name == 'random_split':
            lab = 'random split (B)'
            col = C_B
        else:
            c = name.replace('fold_', '')
            lab = 'LOCO, %s held out' % COUNTY_LABEL.get(c, c)
            col = C_C
        rows.append((rec['FPPI'], lab, col, rec))
    rows.sort(key=lambda r: r[0])                      # ascending: smallest at the bottom
    fppi = [r[0] for r in rows]
    labs = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    fpr = [r[3]['FPR_images'] for r in rows]

    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.barh(y, fppi, 0.62, color=cols)
    for yi, v in zip(y, fppi):
        ax.text(v + 0.25, yi, '%.2f' % v, va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9.5)
    ax.set_xlim(0, 25.5)
    ax.set_xlabel('false-positive boxes per image (FPPI) at confidence floor 0.001, '
                  '2003 negative patches\n(image-level false-alarm rate %.3f-%.3f '
                  'for all nine models)' % (min(fpr), max(fpr)))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_B, label='random split (county seen)'),
                       Patch(color=C_C, label='LOCO folds (county unseen)')],
              fontsize=9, loc='lower right', framealpha=0.9, edgecolor='0.6')
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig7_fppi_models.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)
    for yi, v, l in zip(y, fppi, labs):
        print('    %-28s %.4f' % (l, v))


if __name__ == '__main__':
    main()
