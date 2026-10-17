#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The saturation mechanism measured by an IoU sweep.

  (a) pooled AP vs the IoU matching threshold, B (seen, blue) and C (unseen, red);
  (b) the B - C leakage vs the same threshold, highlighting that the gap is tiny
      at IoU 0.50 and roughly triples by IoU 0.75.

Reads iou_sweep_n_256.json. Times New Roman throughout; no in-image title.
Output: paper/figs/fig8_iou_sweep.png (300 dpi).
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
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white',
})
C_B, C_C, C_LEAK = '#1f77b4', '#d62728', '#2a6f4e'

d = json.load(open(os.path.join(HERE, 'iou_sweep_n_256.json')))
ious = [c['iou'] for c in d['curve']]
B = np.array([c['B'] for c in d['curve']])
C = np.array([c['C'] for c in d['curve']])
leak = B - C

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))

# (a) AP vs IoU
ax1.plot(ious, B, '-o', color=C_B, ms=4, lw=1.4, label='B (county seen)')
ax1.plot(ious, C, '-s', color=C_C, ms=4, lw=1.4, label='C (county unseen)')
ax1.set_xlabel('IoU matching threshold')
ax1.set_ylabel('Pooled AP (770 instances)')
ax1.set_ylim(-0.02, 1.02)
ax1.legend(frameon=False, loc='upper right', fontsize=8)
ax1.annotate('gap +0.081', xy=(0.50, (B[0] + C[0]) / 2), xytext=(0.56, 0.97),
             fontsize=7.5, color='0.35',
             arrowprops=dict(arrowstyle='-', color='0.6', lw=0.6))
ax1.annotate('gap +0.232', xy=(0.75, (B[6] + C[6]) / 2), xytext=(0.70, 0.30),
             fontsize=7.5, color='0.35',
             arrowprops=dict(arrowstyle='-', color='0.6', lw=0.6))
ax1.text(0.02, 0.03, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold')

# (b) leakage vs IoU
bars = ax2.bar(ious, leak, width=0.035, color=C_LEAK, edgecolor='none', alpha=0.85)
imax = int(np.argmax(leak))
bars[imax].set_color('#8c2d04')
ax2.axhline(leak[0], color='0.5', ls='--', lw=0.8)
ax2.text(0.955, leak[0] + 0.006, 'IoU 0.50 level', fontsize=7, color='0.4', ha='right')
ax2.set_xlabel('IoU matching threshold')
ax2.set_ylabel('Leakage (B \u2212 C)')
ax2.set_ylim(0, max(leak) * 1.22)
ax2.set_xticks(ious)
ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold')

fig.tight_layout()
out = os.path.join(FIGS, 'fig8_iou_sweep.png')
fig.savefig(out)
print('saved', out)
