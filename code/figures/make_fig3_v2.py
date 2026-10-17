# -*- coding: utf-8 -*-
"""
Regenerate fig3 with an explicit legend showing both family (classification
vs detection) and metric color, in response to Prof. Xiong's request to
"add a legend" to Figure 3.

Source data: supp_stats.json (paired records, 7 metrics, 4 classification
+ 3 detection). Bars and whiskers are unchanged; only the legend is
redesigned. Output:
    paper/figs/fig3_leakage_by_metric_v2.png
The v1 file is kept untouched for comparison.
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

# font
for cand in ["Times New Roman", "Times", "Arial"]:
    try:
        plt.rcParams["font.family"] = cand
        plt.rcParams["font.size"] = 9.5
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

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.barh(y, means, xerr=[los, his], color=colors, height=0.6, zorder=3,
        error_kw=dict(ecolor="0.25", lw=1, capsize=3, zorder=2))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("leakage  B (county seen)  \u2212  C (county withheld)",
              fontsize=10)
for yy, m, h in zip(y, means, his):
    ax.text(m + h + 0.005, yy, "%+.4f" % m, va="center", fontsize=9.5, zorder=5)
ax.set_xlim(0, max(m + h for m, h in zip(means, his)) + 0.060)
ax.axvspan(0.0144 - 0.003, 0.0144 + 0.003, color="0.85", zorder=0)
ax.annotate("shaded band: the whole leakage as mAP50 sees it",
            xy=(0.0245, 5.55), fontsize=8.5, color="0.3")

# ---- 增强的图例：颜色（分类/检测） + 误差线（bootstrap 95% CI） + 灰带（mAP50 区间） ----
legend_handles = [
    Patch(facecolor=C_B, edgecolor="none", label="classification (4 metrics)"),
    Patch(facecolor=C_C, edgecolor="none", label="detection (3 metrics)"),
    Line2D([0], [0], color="0.25", lw=1, marker="|", markersize=6,
           label="bootstrap 95% CI (10,000 resamples)"),
    Patch(facecolor="0.85", edgecolor="0.6", label="mAP50 effect band (0.0114\u20130.0174)"),
]
leg = ax.legend(handles=legend_handles, fontsize=8.5,
                loc="upper center", bbox_to_anchor=(0.55, -0.20),
                ncol=2, frameon=True, framealpha=0.95,
                edgecolor="0.6", handlelength=1.8, columnspacing=1.6)
# 给图例留出底部空间
fig.subplots_adjust(bottom=0.32, left=0.16, right=0.97, top=0.97)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(FIGS, "fig3_leakage_by_metric_v2.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote:", out, os.path.getsize(out), "bytes")

# 同步出 v2 矢量版
out_pdf = os.path.join(FIGS, "fig3_leakage_by_metric_v2.pdf")
fig2 = plt.figure(figsize=(7.2, 3.6))
ax2 = fig2.add_subplot(111)
ax2.barh(y, means, xerr=[los, his], color=colors, height=0.6, zorder=3,
         error_kw=dict(ecolor="0.25", lw=1, capsize=3, zorder=2))
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel("leakage  B (county seen)  \u2212  C (county withheld)", fontsize=10)
for yy, m, h in zip(y, means, his):
    ax2.text(m + h + 0.005, yy, "%+.4f" % m, va="center", fontsize=9.5, zorder=5)
ax2.set_xlim(0, max(m + h for m, h in zip(means, his)) + 0.060)
ax2.axvspan(0.0144 - 0.003, 0.0144 + 0.003, color="0.85", zorder=0)
ax2.annotate("shaded band: the whole leakage as mAP50 sees it",
             xy=(0.0245, 5.55), fontsize=8.5, color="0.3")
leg2 = ax2.legend(handles=legend_handles, fontsize=8.5,
                  loc="upper center", bbox_to_anchor=(0.55, -0.20),
                  ncol=2, frameon=True, framealpha=0.95,
                  edgecolor="0.6", handlelength=1.8, columnspacing=1.6)
fig2.tight_layout(rect=(0, 0.04, 1, 1))
fig2.savefig(out_pdf, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("wrote:", out_pdf, os.path.getsize(out_pdf), "bytes")
