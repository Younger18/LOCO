# -*- coding: utf-8 -*-
"""
Regenerate Figure 3 with the legend moved INSIDE the blank area of the plot
(no separate legend band below the axes), per Prof. Xiong's request
"Figure 3 中的图例直接移到图中的空白处".

The blank region used is the mid-right of the axes (x ~ 0.13-0.185,
y ~ 3.5-5.5), which no bar reaches: the four classification bars end at
0.065 / 0.11 / 0.10 / 0.035 and the mAP50 bar at 0.030.

Data source: supp_stats.json (paired records, 7 metrics). Bars, whiskers and
value labels are unchanged from v2; only the legend location changes.

Outputs:
    paper/figs/fig3_leakage_by_metric_v3.png
    paper/figs/fig3_leakage_by_metric_v3.pdf
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

for cand in ["Times New Roman", "Times", "Arial"]:
    try:
        plt.rcParams["font.family"] = cand
        plt.rcParams["font.size"] = 10
        break
    except Exception:
        continue
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42

C_B = "#1F77B4"   # classification blue
C_C = "#D62728"   # detection red

S = json.load(open(os.path.join(HERE, "supp_stats.json")))
recs = S["paired"]
order = ["classification AP", "classification F1", "classification recall",
         "classification FPR (rise)", "detection mAP50",
         "detection sizeAP-all@0.5", "detection mAP50-95"]
recs = [r for r in recs if r["label"] in order]
recs = sorted(recs, key=lambda r: order.index(r["label"]))
labels, means, los, his = [], [], [], []
for r in recs:
    labels.append(r["label"].replace("classification ", "classif. ")
                 .replace("detection ", "detect. "))
    means.append(r["mean"])
    los.append(r["mean"] - r["boot95"][0])
    his.append(r["boot95"][1] - r["mean"])
y = np.arange(len(labels))[::-1]
colors = [C_B] * 4 + [C_C] * 3

fig, ax = plt.subplots(figsize=(7.2, 3.5))
ax.barh(y, means, xerr=[los, his], color=colors, height=0.6, zorder=3,
        error_kw=dict(ecolor="0.25", lw=1, capsize=3, zorder=2))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("leakage  B (county seen)  \u2212  C (county withheld)",
              fontsize=10)
for yy, m, h in zip(y, means, his):
    ax.text(m + h + 0.005, yy, "%+.4f" % m, va="center", fontsize=9.5, zorder=5)
xmax = max(m + h for m, h in zip(means, his)) + 0.055
ax.set_xlim(0, xmax)
ax.set_ylim(-0.6, 6.6)

# mAP50 灰带（与题注一致：整个 mAP50 效应 0.0144，CI [0.004, 0.026]）
ax.axvspan(0.0144 - 0.003, 0.0144 + 0.003, color="0.85", zorder=0)
ax.annotate("shaded band: the whole leakage as mAP50 sees it",
            xy=(0.0245, 5.55), fontsize=8.5, color="0.3")

# ---- 图例：放入图中空白处（中右侧，无条形覆盖）----
legend_handles = [
    Patch(facecolor=C_B, edgecolor="none", label="classification"),
    Patch(facecolor=C_C, edgecolor="none", label="detection"),
    Line2D([0], [0], color="0.25", lw=1, marker="|", markersize=6,
           label="bootstrap 95% CI"),
    Patch(facecolor="0.85", edgecolor="0.6", label="mAP50 effect band"),
]
leg = ax.legend(handles=legend_handles,
                loc="center right", bbox_to_anchor=(0.995, 0.60),
                ncol=1, fontsize=8.5, frameon=True, framealpha=0.95,
                edgecolor="0.6", handlelength=1.6, handletextpad=0.6,
                borderpad=0.5, labelspacing=0.45)

fig.tight_layout()
out_png = os.path.join(FIGS, "fig3_leakage_by_metric_v3.png")
out_pdf = os.path.join(FIGS, "fig3_leakage_by_metric_v3.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote:", out_png, os.path.getsize(out_png), "bytes")
print("wrote:", out_pdf, os.path.getsize(out_pdf), "bytes")

from PIL import Image
print("png size:", Image.open(out_png).size)
