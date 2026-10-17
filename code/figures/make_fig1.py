#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 (revised): composite location figure.

Panel (a) uses the annotated overview map (fig1_bijie_anno.png): the published
Bijie landslide locations with the names of the eight county-level units
lettered on the map. It depicts a municipal region only and contains no
national boundary, so no territorial-compliance question arises.

Panel (b) is our own contribution: landslide-patch count per county-level unit,
i.e. the LOCO test-set size, which is the grouping variable of the paper.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)
SRC = os.path.join(FIGS, 'fig1_bijie_anno.png')

COUNTIES = [('qx', 'Qianxi', 168), ('qxg', 'Qixingguan', 125), ('zj', 'Zhijin', 122),
            ('hz', 'Hezhang', 102), ('js', 'Jinsha', 75), ('ny', 'Nayong', 72),
            ('wn', 'Weining', 70), ('df', 'Dafang', 36)]
COUNTY_LABEL = {'qx': 'Qianxi', 'qxg': 'Qixingguan', 'zj': 'Zhijin', 'hz': 'Hezhang',
                'js': 'Jinsha', 'ny': 'Nayong', 'wn': 'Weining', 'df': 'Dafang'}

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
    # flatten the RGBA source onto white so the composite prints cleanly
    im = Image.open(SRC).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, im).convert('RGB')
    w, h = flat.size

    # Make panel (b)'s height EXACTLY equal to the map image's displayed height
    # in panel (a).  Panel (a) is width-constrained (the 0.46-wide axes box holds
    # the 879x519 map at 4.784 in wide -> 2.825 in tall, no letterboxing), so the
    # bar chart must use that same height so the two panels line up.
    fig_w, fig_h = 10.4, 3.6
    axa_w_in = 0.46 * fig_w
    map_disp_h_in = axa_w_in / (w / h)      # map height as shown in (a)
    hfrac = map_disp_h_in / fig_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    # both panels share the same bottom and height so the two subplots are of
    # equal height; their widths differ (0.46 vs 0.42 of the figure)
    axa = fig.add_axes([0.02, 0.10, 0.46, hfrac])
    axb = fig.add_axes([0.56, 0.10, 0.42, hfrac])

    axa.imshow(np.asarray(flat))
    axa.set_xticks([])
    axa.set_yticks([])
    for s in axa.spines.values():
        s.set_color('0.7')
    axa.set_title('(a) Study area and landslide locations', fontsize=9.5, loc='left')

    labels = [COUNTY_LABEL[c] for c, _, _ in COUNTIES]
    vals = [v for _, _, v in COUNTIES]
    y = np.arange(len(labels))[::-1]
    axb.barh(y, vals, color='#c44e52', height=0.62)
    for yy, v in zip(y, vals):
        axb.text(v + 3, yy, '%d' % v, va='center', fontsize=8)
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=8.5)
    axb.set_xlabel('landslide patches (= LOCO test-set size)')
    axb.set_xlim(0, 195)
    axb.set_title('(b) Patch count per county-level unit', fontsize=9.5, loc='left')
    axb.grid(axis='y', visible=False)

    # the provenance note lives in the figure caption, not inside the image
    out = os.path.join(FIGS, 'fig1_study_area.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out, '(%d x %d source)' % (w, h))


if __name__ == '__main__':
    main()
