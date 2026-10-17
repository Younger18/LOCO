#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Figure 1(b): landslide-patch count per county-level unit.

Emitted as its own file so it can be assembled separately (the user is
compositing the panels by hand).  The plotting-area height is set to the
same 2.825 in that the map image (fig1_bijie_anno.png) occupies inside
Figure 1(a), so the two line up when placed side by side.

Output: paper/figs/fig1_panel_b.png  (300 dpi)
"""
import os
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')

COUNTIES = [('qx', 'Qianxi', 168), ('qxg', 'Qixingguan', 125), ('zj', 'Zhijin', 122),
            ('hz', 'Hezhang', 102), ('js', 'Jinsha', 75), ('ny', 'Nayong', 72),
            ('wn', 'Weining', 70), ('df', 'Dafang', 36)]

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False,
    'savefig.facecolor': 'white',
})


def main():
    # match the map's displayed height in Figure 1(a)
    with Image.open(os.path.join(FIGS, 'fig1_bijie_anno.png')) as im:
        W, H = im.size
    axa_w_in = 0.46 * 10.4                 # same (a)-box width as Figure 1
    map_disp_h_in = axa_w_in / (W / H)     # 2.825 in
    fig_w, fig_h = 4.6, map_disp_h_in + 0.70
    hfrac = map_disp_h_in / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.16, 0.10, 0.80, hfrac])

    labels = [c[1] for c in COUNTIES]
    vals = [c[2] for c in COUNTIES]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color='#c44e52', height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 3, yy, '%d' % v, va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('landslide patches (= LOCO test-set size)')
    ax.set_xlim(0, 195)
    ax.set_title('(b) Patch count per county-level unit', fontsize=9.5, loc='left')
    ax.grid(axis='y', visible=False)

    out = os.path.join(FIGS, 'fig1_panel_b.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', out, '(panel height %.3f in)' % map_disp_h_in)


if __name__ == '__main__':
    main()
