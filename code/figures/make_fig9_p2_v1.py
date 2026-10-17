#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate Figure 9 (P2 branch under LOCO) as v1: remove the legend in
panel (a) only. Panel (b) keeps its legend. All other styling is a byte-for-byte
replica of make_figures.fig_p2() (figsize 9.6x3.5, 300 dpi, TNR, titles 9pt,
ticks 8pt, bar labels 7pt, bbox_inches='tight'). Output -> paper/figs/fig9_p2_under_loco_v1.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import make_figures as MF  # reuse rcParams + constants (figure fns are guarded by __main__)

D = json.load(open(os.path.join(MF.HERE, 'p2_comparison.json')))
rows = D['per_county']
x = np.arange(len(rows))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
w = 0.38

# ---- panel (a): per-county mAP50-95 under LOCO; LEGEND REMOVED ----
axes[0].bar(x - w / 2, [r['n_mAP50_95'] for r in rows], w, color=MF.C_B, label='YOLO26n')
axes[0].bar(x + w / 2, [r['p2_mAP50_95'] for r in rows], w, color='#ff7f0e', label='YOLO26n + P2')
axes[0].set_title('(a) mAP50-95 under LOCO (C models)', fontsize=9)
axes[0].set_ylabel('score')
# axes[0].legend(fontsize=8)   # <-- removed per request (legend in subplot a)
axes[0].set_xticks(x)
axes[0].set_xticklabels([MF.COUNTY_LABEL[c] for c in MF.COUNTIES], rotation=35, ha='right', fontsize=8)
axes[0].set_xlim(-0.55, 7.55)

# ---- panel (b): leakage (B - C) unchanged by P2; legend KEPT ----
L = D['leakage_by_arch']
lab = ['mAP50', 'mAP50-95']
xs = np.arange(2)
wb = w * 2.0 / 8.1
axes[1].bar(xs - wb / 2, [L['n']['mAP50_mean'], L['n']['mAP50_95_mean']], wb,
            color=MF.C_B, label='YOLO26n')
axes[1].bar(xs + wb / 2, [L['p2']['mAP50_mean'], L['p2']['mAP50_95_mean']], wb,
            color='#ff7f0e', label='YOLO26n + P2')
for i, (a, b) in enumerate(zip([L['n']['mAP50_mean'], L['n']['mAP50_95_mean']],
                               [L['p2']['mAP50_mean'], L['p2']['mAP50_95_mean']])):
    axes[1].text(i - wb / 2, a + 0.004, '%.4f' % a, ha='center', fontsize=7)
    axes[1].text(i + wb / 2, b + 0.016, '%.4f' % b, ha='center', fontsize=7)
axes[1].set_xticks(xs)
axes[1].set_xticklabels(lab)
axes[1].set_title('(b) Leakage (B \u2212 C) is unchanged by P2', fontsize=9)
axes[1].set_ylabel('leakage')
axes[1].legend(fontsize=8)
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(0, 0.150)

fig.tight_layout()
out = os.path.join(MF.FIGS, 'fig9_p2_under_loco_v1.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('wrote', out)

# ---- QA: confirm legend removed from (a), kept in (b) ----
leg0 = axes[0].get_legend()
leg1 = axes[1].get_legend()
print('panel (a) legend present:', leg0 is not None, '(expect False)')
print('panel (b) legend present:', leg1 is not None, '(expect True)')
print('panel (a) bar count:', len(axes[0].patches), ' panel (b) bar count:', len(axes[1].patches))
from PIL import Image
im = Image.open(out)
print('output size (px):', im.size)
