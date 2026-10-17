#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Methodological workflow flowchart (publication-grade, per-box icons).

Built from the study workflow (method section + overview figure): audited Bijie
benchmark -> the two split protocols (B random, C LOCO) -> identical training
-> evaluation on identical test images -> paired B - C contrast, then the lower
band with the two controls, the sensitivity analyses and the conclusion.

Each box carries a content-relevant line icon; the start box uses a rounded
rectangle (rounded corners).  Saved as a JPG, NOT embedded in the text.

Output: paper/figs/fig_methodology_flow.jpg  (300 dpi)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False,
    'savefig.facecolor': 'white',
})

INK = '#1a1a1a'
BODY = '#2a2a2a'
GRAY = '#5a5a5a'
C_B, C_C = '#1f6fb2', '#c0392b'
C_START = '#2e8b57'


# ----------------------------- icon primitives -----------------------------
def _ellipse(ax, cx, cy, rx, ry, color, lw, n=64):
    t = np.linspace(0, 2 * np.pi, n)
    ax.plot(cx + rx * np.cos(t), cy + ry * np.sin(t), color=color, lw=lw)


def _circle(ax, cx, cy, r, color, lw, n=72):
    t = np.linspace(0, 2 * np.pi, n)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=color, lw=lw)


def icon_db(ax, cx, cy, s, color, lw):        # stacked disks -> dataset
    rx, ry = 0.62, 0.30
    _ellipse(ax, cx, cy - 0.72, rx, ry, color, lw)
    _ellipse(ax, cx, cy + 0.72, rx, ry, color, lw)
    ax.plot([cx - rx, cx - rx], [cy - 0.72, cy + 0.72], color=color, lw=lw)
    ax.plot([cx + rx, cx + rx], [cy - 0.72, cy + 0.72], color=color, lw=lw)
    _ellipse(ax, cx, cy, rx, ry, color, lw)


def icon_shuffle(ax, cx, cy, s, color, lw):    # crossing arrows -> random split
    ax.annotate('', xy=(cx + 0.78, cy + 0.52), xytext=(cx - 0.78, cy - 0.42),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=14))
    ax.annotate('', xy=(cx + 0.78, cy - 0.52), xytext=(cx - 0.78, cy + 0.42),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=14))


def icon_grid(ax, cx, cy, s, color, lw):       # grid, one cell out -> LOCO
    x0, x1 = cx - 0.70, cx + 0.70
    y0, y1 = cy - 0.70, cy + 0.70
    ax.plot([x0, x1], [y0, y0], color=color, lw=lw)
    ax.plot([x0, x1], [y1, y1], color=color, lw=lw)
    ax.plot([x0, x0], [y0, y1], color=color, lw=lw)
    ax.plot([x1, x1], [y0, y1], color=color, lw=lw)
    ax.plot([x0, x1], [cy - 0.23, cy - 0.23], color=color, lw=lw * 0.7)
    ax.plot([x0, x1], [cy + 0.23, cy + 0.23], color=color, lw=lw * 0.7)
    ax.plot([cx - 0.23, cx - 0.23], [y0, y1], color=color, lw=lw * 0.7)
    ax.plot([cx + 0.23, cx + 0.23], [y0, y1], color=color, lw=lw * 0.7)
    ax.add_patch(Rectangle((cx - 0.23, cy - 0.23), 0.46, 0.46,
                           fill=False, ec=color, lw=lw * 1.9))


def icon_chip(ax, cx, cy, s, color, lw):       # processor -> model / training
    r = 0.70
    ax.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, ec=color, lw=lw))
    ax.add_patch(Rectangle((cx - 0.34, cy - 0.34), 0.68, 0.68, fill=False, ec=color, lw=lw * 0.8))
    for yy in np.linspace(cy - 0.38, cy + 0.38, 3):
        ax.plot([cx - r, cx - r - 0.20], [yy, yy], color=color, lw=lw)
        ax.plot([cx + r, cx + r + 0.20], [yy, yy], color=color, lw=lw)
    for xx in np.linspace(cx - 0.38, cx + 0.38, 3):
        ax.plot([xx, xx], [cy - r, cy - r - 0.20], color=color, lw=lw)
        ax.plot([xx, xx], [cy + r, cy + r + 0.20], color=color, lw=lw)


def icon_bars(ax, cx, cy, s, color, lw):       # bar chart -> evaluation metrics
    ax.plot([cx - 0.90, cx + 0.90], [cy - 0.80, cy - 0.80], color=color, lw=lw)
    for x, hh in zip([-0.70, -0.23, 0.23, 0.70], [0.62, 1.00, 0.78, 0.50]):
        ax.plot([cx + x, cx + x], [cy - 0.80, cy - 0.80 + hh], color=color, lw=lw * 1.7)
        ax.plot([cx + x - 0.11, cx + x + 0.11], [cy - 0.80 + hh, cy - 0.80 + hh],
                color=color, lw=lw * 1.7)


def icon_venn(ax, cx, cy, s, color, lw):        # two circles -> B vs C contrast
    _circle(ax, cx - 0.36, cy, 0.64, color, lw)
    _circle(ax, cx + 0.36, cy, 0.64, color, lw)


def icon_err(ax, cx, cy, s, color, lw):         # error bars -> uncertainty
    ax.plot([cx, cx], [cy - 0.82, cy + 0.82], color=color, lw=lw * 1.9)
    ax.plot([cx - 0.30, cx + 0.30], [cy - 0.82, cy - 0.82], color=color, lw=lw * 1.9)
    ax.plot([cx - 0.30, cx + 0.30], [cy + 0.82, cy + 0.82], color=color, lw=lw * 1.9)
    ax.plot(cx, cy, 'o', color=color, ms=7, mew=lw * 1.2)


def icon_shield(ax, cx, cy, s, color, lw):      # shield -> controls
    pts = [(cx - 0.60, cy + 0.55), (cx + 0.60, cy + 0.55), (cx + 0.60, cy - 0.05),
           (cx, cy - 0.80), (cx - 0.60, cy - 0.05)]
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    ax.plot(xs, ys, color=color, lw=lw)
    ax.plot([cx - 0.42, cx + 0.42], [cy + 0.12, cy + 0.12], color=color, lw=lw * 0.8)


def icon_sliders(ax, cx, cy, s, color, lw):     # sliders -> sensitivity analyses
    for y, kx in zip([0.62, 0.0, -0.62], [-0.62, 0.20, -0.30]):
        ax.plot([cx - 0.72, cx + 0.72], [cy + y, cy + y], color=color, lw=lw)
        ax.plot(cx + kx, cy + y, 'o', color=color, ms=8, mew=lw)


def icon_flag(ax, cx, cy, s, color, lw):        # flag -> conclusion
    ax.plot([cx - 0.40, cx - 0.40], [cy - 0.82, cy + 0.82], color=color, lw=lw * 1.4)
    ax.plot([cx - 0.40, cx + 0.62], [cy + 0.82, cy + 0.55], color=color, lw=lw)
    ax.plot([cx + 0.62, cx - 0.40], [cy + 0.55, cy + 0.28], color=color, lw=lw)
    ax.plot([cx - 0.40, cx - 0.40], [cy + 0.28, cy + 0.82], color=color, lw=lw)


# ----------------------------- box helpers -----------------------------
def add_box(ax, x, y, w, h, title, lines, fill, edge, accent, icon,
            rounded=1.2, title_size=9, body_size=8, icon_scale=2.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.3,rounding_size=%s' % rounded,
                 linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2))
    T = y + h
    cx = x + w / 2
    icon(ax, cx, T - 6.0, icon_scale, accent, 1.5)
    ty = T - 11.0
    ax.text(cx, ty, title, ha='center', va='top', fontsize=title_size,
            fontweight='bold', color=INK, zorder=3)
    ty -= (title_size * 0.60 + 3.0)
    last_top = None
    for ln in lines:
        if ln:
            ax.text(cx, ty, ln, ha='center', va='top', fontsize=body_size,
                    color=BODY, zorder=3)
            last_top = ty
        ty -= (body_size * 0.60 + 2.8)
    if last_top is not None:
        last_bottom = last_top - body_size * 0.60
        if last_bottom < y + 0.6:
            print('  !! WARN %s: last text bottom %.1f < box bottom %.1f'
                  % (title, last_bottom, y))
    return last_bottom


def add_box_sideicon(ax, x, y, w, h, title, lines, fill, edge, accent, icon,
                     title_size=8.5, body_size=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.3,rounding_size=1.2',
                 linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2))
    icon(ax, x + 3.6, y + h / 2, 2.6, accent, 1.5)
    tx = x + 7.2
    T = y + h
    ty = T - 2.6
    ax.text(tx, ty, title, ha='left', va='top', fontsize=title_size,
            fontweight='bold', color=INK, zorder=3)
    ty -= (title_size * 0.60 + 3.0)
    last_top = ty
    for ln in lines:
        if ln:
            ax.text(tx, ty, ln, ha='left', va='top', fontsize=body_size,
                    color=BODY, zorder=3)
            last_top = ty
        ty -= (body_size * 0.60 + 3.0)
    if last_top is not None:
        last_bottom = last_top - body_size * 0.60
        if last_bottom < y + 0.6:
            print('  !! WARN %s: last text bottom %.1f < box bottom %.1f'
                  % (title, last_bottom, y))
    return last_bottom


def arrow(ax, p0, p1):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='-|>', mutation_scale=12, lw=1.2,
                                color=GRAY, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(10.6, 7.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 76)
    ax.axis('off')

    ax.text(50, 74, 'Methodological workflow', ha='center', va='top',
            fontsize=12.5, fontweight='bold', color='#444444')

    # ---- main pipeline (top band) ----
    add_box(ax, 1.5, 30, 14, 34, 'Bijie benchmark',
            ['770 LS + 2003 NL patches', 'county labels; no coords'],
            fill='#e8f4ee', edge=C_START, accent=C_START, icon=icon_db, rounded=2.4)

    add_box_sideicon(ax, 20.5, 47, 19, 17, 'Random split (B)',
            ['seed 42; county SEEN'],
            fill='#e7f0f8', edge=C_B, accent=C_B, icon=icon_shuffle)

    add_box_sideicon(ax, 20.5, 30, 19, 17, 'LOCO (C)',
            ['8 folds; county UNSEEN'],
            fill='#fbeaea', edge=C_C, accent=C_C, icon=icon_grid)

    add_box(ax, 43.5, 30, 15, 34, 'Identical training',
            ['YOLO26n @ 256 px', 'ResNet-18 replication'],
            fill='#f4f4f4', edge='#555555', accent='#555555', icon=icon_chip)

    add_box(ax, 62, 30, 15, 34, 'Evaluation',
            ['identical test images', 'mAP50; mAP50-95; FPR'],
            fill='#f4f4f4', edge='#555555', accent='#555555', icon=icon_bars)

    add_box(ax, 80.5, 30, 17.5, 34, 'Paired contrast (B \u2212 C)',
            ['leakage \u00b1 bootstrap 95% CI', 'exact Wilcoxon p'],
            fill='#e8eef5', edge='#33517a', accent='#33517a', icon=icon_venn,
            title_size=8.6)

    arrow(ax, (15.5, 55.5), (20.5, 55.5))
    arrow(ax, (15.5, 38.5), (20.5, 38.5))
    arrow(ax, (39.5, 55.5), (43.5, 55.5))
    arrow(ax, (39.5, 38.5), (43.5, 38.5))
    arrow(ax, (58.5, 47.0), (62.0, 47.0))
    arrow(ax, (77.0, 47.0), (80.5, 47.0))

    # ---- lower band: controls, robustness, conclusion ----
    add_box_sideicon(ax, 6, 3, 28, 24, 'Controls',
            ['C1 in-domain parity', 'C2 within-C domain shift'],
            fill='#ffffff', edge='#555555', accent='#555555', icon=icon_shield)

    add_box_sideicon(ax, 38, 3, 28, 24, 'Sensitivity',
            ['IoU sweep 0.50\u20130.95', 'seeds 0 / 7 / 42; P2 branch'],
            fill='#ffffff', edge='#555555', accent='#555555', icon=icon_sliders,
            title_size=8.5, body_size=7.5)

    add_box_sideicon(ax, 70, 3, 28.5, 24, 'Conclusion',
            ['metric choice decides', 'if leakage is visible'],
            fill='#f0f0f0', edge='#333333', accent='#333333', icon=icon_flag)

    # elbow: contrast box -> lower band
    ax.plot([89.25, 89.25], [30 - 0.4, 28], color=GRAY, lw=1.2)
    ax.plot([89.25, 20], [28, 28], color=GRAY, lw=1.2)
    arrow(ax, (20, 28), (20, 27.4))
    arrow(ax, (34, 15), (38, 15))
    arrow(ax, (66, 15), (70, 15))

    out = os.path.join(FIGS, 'fig_methodology_flow.jpg')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.18, facecolor='white',
                format='jpg')
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    main()
