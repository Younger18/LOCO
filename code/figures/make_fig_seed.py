#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness of the leakage to the training seed.

Grouped bars give the leakage
(B - C, mean over the eight counties) at the three training seeds:
mAP50 (blue) against mAP50-95 (red), following the colour convention of
Figure 2. Labels above each pair give the mAP50-95 / mAP50 ratio; dashed
lines mark the across-seed means.

Reads multiseed_leakage.json. Times New Roman throughout; no in-image title.
Output: paper/figs/fig11_seed_robustness.png (300 dpi).
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
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(x - w / 2 - gap / 2, m50, w, color=C_B, label='mAP50')
ax.bar(x + w / 2 + gap / 2, m95, w, color=C_C, label='mAP50-95')
for xi, a, b in zip(x, m50, m95):
    ax.text(xi - w / 2 - gap / 2, a + 0.004, '+%.4f' % a, ha='center',
            fontsize=8.5, color=C_B)
    ax.text(xi + w / 2 + gap / 2, b + 0.004, '+%.4f' % b, ha='center',
            fontsize=8.5, color=C_C)
    ax.text(xi, b + 0.024, '%.1f\u00d7' % (b / a), ha='center',
            fontsize=9, color='0.35')
# across-seed means with their spread
ax.axhline(agg50['mean'], color=C_B, ls='--', lw=0.9, alpha=0.65)
ax.axhline(agg95['mean'], color=C_C, ls='--', lw=0.9, alpha=0.65)
ax.text(2.60, agg50['mean'] + 0.004, 'mean 0.0109 \u00b1 0.0048',
        color=C_B, fontsize=8.5, va='bottom')
ax.text(2.60, agg95['mean'] + 0.004, 'mean 0.0922 \u00b1 0.0634',
        color=C_C, fontsize=8.5, va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(['seed %d' % s for s in seeds])
ax.set_ylabel('leakage  B \u2212 C  (mean over 8 counties)')
ax.set_xlim(-0.55, 3.80)
ax.set_ylim(0, 0.190)
ax.legend(fontsize=9, framealpha=0.9, edgecolor='0.6', loc='upper left')
fig.tight_layout()
out = os.path.join(FIGS, 'fig11_seed_robustness.png')
fig.savefig(out)
print('saved', out)
