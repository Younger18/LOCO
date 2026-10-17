#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5 (new, replaces Table 8): pooled size-stratified AP50.

Objects are binned at the TRAINING scale (effective area after letterboxing to
256 px, COCO thresholds 32^2 / 96^2). B = random-split model (county seen),
C = per-fold LOCO model (county unseen); each county is judged by its own C
model, predictions pooled over all 770 instances.

Data: loco_results_n_256.json -> "pooled" (written by eval_loco.py).
The small bin holds only 13 instances and is descriptive only (marked *).
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
    P = json.load(open(os.path.join(HERE, 'loco_results_n_256.json')))['pooled']
    bins = ['small', 'medium', 'large', 'all']
    labels = ['small (< 32\u00b2 px\u00b2)\nn = 13 *', 'medium (32\u00b2-96\u00b2 px\u00b2)\nn = 421',
              'large (> 96\u00b2 px\u00b2)\nn = 336', 'all objects\nn = 770']
    B = [P['B_seen'][b] for b in bins]
    C = [P['C_unseen'][b] for b in bins]
    L = [P['leakage'][b] for b in bins]

    x = np.arange(len(bins))
    w = 0.30
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.bar(x - w / 2, B, w, color=C_B, label='B (county seen)')
    ax.bar(x + w / 2, C, w, color=C_C, label='C (county unseen)')
    for xi, v in zip(x - w / 2, B):
        ax.text(xi, v + 0.012, '%.4f' % v, ha='center', fontsize=8.5)
    for xi, v in zip(x + w / 2, C):
        ax.text(xi, v + 0.012, '%.4f' % v, ha='center', fontsize=8.5)
    for xi, v, b, c in zip(x, L, B, C):
        ax.text(xi, max(b, c) + 0.062, '%+.4f' % v, ha='center', fontsize=9,
                color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel('pooled AP50 (IoU 0.50, confidence floor 0.001)')
    ax.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 1.005),
              ncol=2, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig5_size_stratified.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    main()
