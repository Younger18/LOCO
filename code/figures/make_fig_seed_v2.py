#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness of the leakage to the training seed.

Redraw of fig11_seed_robustness.png with one hard constraint:
the figure is rendered at its FINAL on-page width of 8 cm (1:1, no
down-scaling at placement), and every text element is exactly 6 pt, so
that when the PNG is placed at 8 cm wide all text reads at 6 pt.
The legend is placed at the upper-right corner.

Saved as paper/figs/fig11_seed_robustness_v2.png (600 dpi) ONLY.
The original fig11_seed_robustness.png is left untouched for comparison.
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
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white',
})
C_B, C_C = '#1f77b4', '#d62728'

d = json.load(open(os.path.join(HERE, 'multiseed_leakage.json')))
seeds = d['seeds']                                   # [42, 0, 7]
m50 = [d['per_seed'][str(s)]['mean_leak_mAP50'] for s in seeds]
m95 = [d['per_seed'][str(s)]['mean_leak_mAP50_95'] for s in seeds]
agg50 = d['mean_leak_mAP50_across_seeds']
agg95 = d['mean_leak_mAP50_95_across_seeds']

x = np.arange(len(seeds))
w = 0.26
gap = 0.02

# Final on-page width = 8 cm; keep original aspect ratio (6.4 : 3.6).
CM = 1.0 / 2.54
FW = 8.0 * CM                       # 3.1496 in
FH = FW * (3.6 / 6.4)               # 1.7717 in  (preserves original proportions)
fig, ax = plt.subplots(figsize=(FW, FH))

ax.bar(x - w / 2 - gap / 2, m50, w, color=C_B, label='mAP50')
ax.bar(x + w / 2 + gap / 2, m95, w, color=C_C, label='mAP50-95')
for xi, a, b in zip(x, m50, m95):
    ax.text(xi - w / 2 - gap / 2, a + 0.004, '+%.4f' % a, ha='center',
            fontsize=6, color=C_B)
    ax.text(xi + w / 2 + gap / 2, b + 0.004, '+%.4f' % b, ha='center',
            fontsize=6, color=C_C)
    ax.text(xi, b + 0.024, '%.1f\u00d7' % (b / a), ha='center',
            fontsize=6, color='0.35')
# across-seed means with their spread
ax.axhline(agg50['mean'], color=C_B, ls='--', lw=0.9, alpha=0.65)
ax.axhline(agg95['mean'], color=C_C, ls='--', lw=0.9, alpha=0.65)
ax.text(2.60, agg50['mean'] + 0.004, 'mean 0.0109 \u00b1 0.0048',
        color=C_B, fontsize=6, va='bottom')
ax.text(2.60, agg95['mean'] + 0.004, 'mean 0.0922 \u00b1 0.0634',
        color=C_C, fontsize=6, va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(['seed %d' % s for s in seeds])
ax.set_ylabel('leakage  B \u2212 C  (mean over 8 counties)')
ax.set_xlim(-0.55, 3.80)
ax.set_ylim(0, 0.190)
ax.legend(fontsize=6, framealpha=0.9, edgecolor='0.6', loc='upper right')
fig.tight_layout()
out = os.path.join(FIGS, 'fig11_seed_robustness_v2.png')
fig.savefig(out)
print('saved', out)
