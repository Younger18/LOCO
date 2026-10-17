# -*- coding: utf-8 -*-
"""Render the 4-row qualitative figure (three positive cases + one negative case).

Columns: original patch / ground-truth label / YOLO26n detection.
Author specifications applied here:
  - ground-truth boxes are WHITE (was green)
  - row and column gaps are 4 pt
  - the negative row (no ground truth) is appended as the fourth case
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = r'.'
OUTDIR = os.path.join(ROOT, 'figures', 'supplementary')
DATADIR = os.path.join(ROOT, 'figs')
pos = {r['stem']: r for r in json.load(open(os.path.join(DATADIR, '_qual_cases_raw.json'), encoding='utf-8'))}
neg = {r['stem']: r for r in json.load(open(os.path.join(DATADIR, '_qual_neg_raw.json'), encoding='utf-8'))}

# (record, row label, side text)
ROWS = [
    (pos['qxg038'], 'Case 1\nCorrect detection', 'Qixingguan County\npatch qxg038'),
    (pos['wn060'], 'Case 2\nLocalization error', 'Weining County\npatch wn060'),
    (pos['qxg110'], 'Case 3\nRedundant detection', 'Qixingguan County\npatch qxg110'),
    (neg['fyb423'], 'Case 4\nFalse alarm\n(no ground truth)', 'Negative sample\n(fyb423)'),
]
COLS = ['Original patch', 'Ground-truth label', 'YOLO26n detection']

GAP_COL = 6.0        # pt, distance from column header labels to the image below
GAP_ROW = 6.0        # pt, vertical gap between case rows (all rows identical)
LABEL_PAD = 8.6      # pt, size of the right-hand metric text (+1 pt from 7.6)

# matplotlib's hspace/wspace are multiples of the average axes size, so they
# cannot be set directly to a point value. The image band is therefore laid out
# inside its own gridspec (no collapsed header row), and the 6 pt row/column
# clearance is obtained by scaling the band height by the fraction (gap+cell)/
# (gap+cell) that the gap represents.
fig = plt.figure(figsize=(8.6, 11.6))
FIG_W, FIG_H = 8.6, 11.6
BAND_TOP, BAND_BOTTOM = 0.878, 0.006          # image band vertical extent (figure frac)
band_h = BAND_TOP - BAND_BOTTOM
# one gap per boundary: 3 boundaries between the 4 image rows (top/bottom edges
# carry no gap), and 2 between the 3 image columns
ROW_GAP_F = GAP_ROW / 72 / (band_h * FIG_H)
COL_GAP_F = GAP_COL / 72 / (band_h * FIG_W)
inner_h = band_h * FIG_H - 3 * GAP_ROW / 72     # inches actually used by cells
inner_w = band_h * FIG_W - 2 * GAP_COL / 72
hs = (GAP_ROW / 72) / (inner_h / 4)
ws = (GAP_COL / 72) / (inner_w / 3)
gs = fig.add_gridspec(
    4, 3,
    height_ratios=[1, 1, 1, 1],
    width_ratios=[1, 1, 1],
    left=0.006, right=0.006 + inner_w / FIG_W, top=BAND_TOP, bottom=BAND_BOTTOM,
    hspace=hs, wspace=ws)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
})


def draw(ax, rec, col):
    # Every panel is the same physical square, so each image is scaled by the
    # same factor: the panel covers max(W, H) source pixels, which makes the
    # displayed ground sample distance identical across all panels.
    img = Image.open(rec['path']).convert('RGB')
    W, H = rec['W'], rec['H']
    SPAN = max(W, H)                # common source span -> identical scaling
    x0, y0 = (SPAN - W) / 2.0, (SPAN - H) / 2.0
    ax.imshow(img, extent=(x0, x0 + W, y0 + SPAN, y0), interpolation='nearest')
    if col == 1:
        for g in rec['gt']:
            cx, cy, w, h = g
            # author requested white ground-truth boxes; a dark halo keeps
            # them legible over bright imagery
            ax.add_patch(patches.Rectangle(((cx - w / 2) * W + x0, (cy - h / 2) * H + y0),
                                           w * W, h * H, fill=False,
                                           edgecolor='#1A1A1A', linewidth=3.4))
            ax.add_patch(patches.Rectangle(((cx - w / 2) * W + x0, (cy - h / 2) * H + y0),
                                           w * W, h * H, fill=False,
                                           edgecolor='#FFFFFF', linewidth=1.8))
    elif col == 2:
        preds = sorted(rec['pred'], key=lambda d: -d['conf'])
        # place each confidence label inside its own box (top-left corner) and
        # flip to the bottom when the box touches the image edge, so no label is
        # clipped or hidden behind an overlapping box
        for p in preds:
            px1, py1, px2, py2 = p['xyxy']
            x1, y1, x2, y2 = px1 + x0, py1 + y0, px2 + x0, py2 + y0
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor='#1A1A1A', linewidth=3.4))
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor='#D62728', linewidth=1.8))
            bw, bh = x2 - x1, y2 - y1
            if bh >= 0.05 * SPAN:              # enough room inside the box
                tx, ty, va = x1 + 0.01 * SPAN, y1 + bh - 0.02 * SPAN, 'top'
            else:                              # shallow box: label below the frame
                tx, ty, va = x1 + 0.01 * SPAN, min(max(y2 + 0.02 * SPAN, 0.05 * SPAN),
                                                  SPAN - 0.02 * SPAN), 'bottom'
            ax.text(tx, ty, f"{p['conf']:.2f}", color='#D62728', fontsize=7.2,
                    ha='left', va=va, fontweight='bold',
                    bbox=dict(fc='white', ec='#D62728', lw=0.4, pad=0.7, alpha=0.92))
    ax.set_xlim(0, SPAN)
    ax.set_ylim(SPAN, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color('#B8B8B8')


def metrics_text(rec):
    if rec['gt']:
        return (f"TP {rec['tp']}   FP {rec['fp']}\nFN {rec['fn']}\n"
                f"IoU {rec['best_iou']:.2f}\nconf {rec['top_conf']:.2f}")
    return (f"TP 0   FP {rec['fp']}\nFN 0\nIoU n/a\nconf {rec['top_conf']:.2f}")


# The author specifies exact 6 pt clearances and a 1.8 cm page margin on all four
# sides. GridSpec's hspace/wspace are multiples of the average axes size and
# cannot produce a positive gap reliably here, so the panels are placed
# explicitly in figure coordinates and every panel is a square of identical size.
PT2CM = 2.54 / 72
PAGE_EDGE = 1.8 / PT2CM                # 1.8 cm expressed in points (51.02 pt)
GAP_COL = 6.0                          # pt, horizontal gap between panels
GAP_ROW = 6.0                          # pt, vertical gap between panels (identical)
CAPTION_GAP = 8.0                      # pt, image band -> caption
CAPTION_LINES = 5
CAPTION_PT = CAPTION_LINES * 9.2 * 1.6
HEADER_PT = 10.4 * 1.2                 # one line of 10.4 pt type
N_ROW, N_COL = 4, 3
CELL_PT = 160.0                        # square panel edge (identical for all 12 panels)
LABEL_W = 26.0                         # pt, left gutter for the rotated row labels
METRIC_GAP = 8.0                       # pt, gap between the last panel and the metrics
METRIC_W = 62.0                        # pt, right gutter for the county/metric text

# canvas: page margin + gutter + panels + inter-panel gaps + gutter + page margin
FIG_W = (2 * PAGE_EDGE + LABEL_W + METRIC_GAP + METRIC_W
         + N_COL * CELL_PT + (N_COL - 1) * GAP_COL) / 72
FIG_H = (2 * PAGE_EDGE + HEADER_PT + GAP_COL + N_ROW * CELL_PT
         + (N_ROW - 1) * GAP_ROW + CAPTION_GAP + CAPTION_PT) / 72

MARGIN_L = PAGE_EDGE / 72 / FIG_W
MARGIN_R = PAGE_EDGE / 72 / FIG_W
MARGIN_B = (PAGE_EDGE + CAPTION_PT + CAPTION_GAP) / 72 / FIG_H
MARGIN_T = 1.0 - (PAGE_EDGE + HEADER_PT + GAP_COL) / 72 / FIG_H
GAP_Y = GAP_ROW / 72 / FIG_H
GAP_X = GAP_COL / 72 / FIG_W
cell_h = CELL_PT / 72 / FIG_H
cell_w = CELL_PT / 72 / FIG_W
PANEL_X0 = PAGE_EDGE / 72 / FIG_W + LABEL_W / 72 / FIG_W   # first panel left edge
PANEL_X1 = PANEL_X0 + N_COL * cell_w + (N_COL - 1) * GAP_X  # last panel right edge
band_h_frac = MARGIN_T - MARGIN_B

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Column headers: the top of the label text sits PAGE_EDGE below the top page
# edge, and the label baseline sits GAP_COL (6 pt) above the image band.
band_top_y = MARGIN_T
header_base = band_top_y + GAP_COL / 72 / FIG_H
for j, c in enumerate(COLS):
    cx = PANEL_X0 + j * (cell_w + GAP_X) + cell_w / 2
    fig.text(cx, header_base, c, fontsize=10.4, fontweight='bold',
             ha='center', va='bottom')

for i, (rec, label, side) in enumerate(ROWS):
    y1 = MARGIN_T - i * (cell_h + GAP_Y)
    y0 = y1 - cell_h
    for j in range(3):
        ax = fig.add_axes([PANEL_X0 + j * (cell_w + GAP_X), y0, cell_w, cell_h])
        draw(ax, rec, j)
    mid_y = (y0 + y1) / 2
    # row label in the left gutter
    fig.text(PAGE_EDGE / 72 / FIG_W + LABEL_W / 2 / 72 / FIG_W, mid_y, label,
             fontsize=8.6, fontweight='bold',
             rotation=90, va='center', ha='center', linespacing=1.6)
    # metrics in the right gutter
    rx = PANEL_X1 + METRIC_GAP / 72 / FIG_W
    fig.text(rx, mid_y + cell_h * 0.20, side, fontsize=LABEL_PAD, va='center',
             ha='left', color='#444444', linespacing=1.6)
    fig.text(rx, mid_y - cell_h * 0.22, metrics_text(rec), fontsize=LABEL_PAD,
             va='center', ha='left', color='#333333', linespacing=1.7)

# Figure caption, centred at the bottom of the figure (journal style).
# Its top edge sits CAPTION_GAP (8 pt) below the image band and the text runs
# downward, so va='top' with the anchor at the band bottom plus the gap.
fig.text(0.5, MARGIN_B - CAPTION_GAP / 72 / FIG_H,
         'Fig. S1.  Qualitative comparison of original patches, ground-truth labels and\n'
         'YOLO26n detections. Bijie landslide benchmark, protocol B (random split),\n'
         'seed 42; boxes shown at confidence ≥ 0.25. Rows 1–3 are positive patches and\n'
         'row 4 is a negative sample without ground truth, so every red box in that row\n'
         'is a false alarm.',
         fontsize=9.2, ha='center', va='top', linespacing=1.6)

for ext, dpi in [('png', 300), ('pdf', None)]:
    p = os.path.join(OUTDIR, f'figS1_qualitative_cases.{ext}')
    fig.savefig(p, dpi=dpi, facecolor='white')
    print('saved:', p, os.path.getsize(p), 'bytes')
plt.close(fig)