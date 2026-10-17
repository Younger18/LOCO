# -*- coding: utf-8 -*-
"""
fig2_v3: 重新生成 Figure 2，把 "TPR" 改为 "recall"（与正文 §4.4 措辞一致），
并把整图字体统一 Times New Roman；同时把右上角 "Evaluation on identical test images"
盒子里的措辞从 "mAP50, mAP50-95, size-stratified AP50, classification AP / F1 /
TPR / FPR, FPPI and image-level FPR at six thresholds" 改为与正文一致的
"mAP50, mAP50-95, size-stratified AP50, classification AP / F1 / recall / FPR,
FPPI and image-level FPR at six thresholds"。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm

for cand in ["Times New Roman", "Times", "Arial"]:
    try:
        plt.rcParams["font.family"] = cand
        plt.rcParams["font.size"] = 10
        break
    except Exception:
        continue
plt.rcParams["mathtext.fontset"] = "stix"

NAVY  = "#1F3A5F"
INK   = "#1A1A1A"
SOFT  = "#DCE5F0"
LBLUE = "#E2EBF5"
LPEACH= "#FCEAE3"
LGRY  = "#EFEFEF"
ACCENT= "#C23030"

# ---- canvas (same as the original 1239x615 figure) ----
fig, ax = plt.subplots(figsize=(12.39, 6.15), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 50)
ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_facecolor("white")

# ---- 顶部条 ----
ax.add_patch(Rectangle((0.5, 0.5), 99, 49, fc="white", ec="none"))

# ---- 第一行：四个 box ----
# Box 1: Bijie benchmark
ax.add_patch(FancyBboxPatch((0.8, 31), 19, 17,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc=LBLUE, ec=NAVY, lw=1.5))
ax.text(10.3, 45.0, "Bijie benchmark", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=NAVY)
ax.text(10.3, 43.2, "(audited release)", ha="center", va="center",
        fontsize=8.5, color=NAVY, style="italic")
ax.text(10.3, 41.5, "770 landslide patches,\n one instance each",
        ha="center", va="center", fontsize=8.2, color=INK)
ax.text(10.3, 36.0, "2003 non-landslide\n patches; county\n attribution from file\n names; no coordinates,\n no buffer zone",
        ha="center", va="center", fontsize=8.2, color=INK)

# Box 2: Protocol B
ax.add_patch(FancyBboxPatch((23, 37), 24, 11,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc="#FFFFFF", ec=NAVY, lw=1.5))
ax.text(35, 45.5, "Protocol B \u2014 random split",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
ax.text(35, 41.5, "(reference protocol), seed 42\n 70/10/20; test spans all 8\n counties  \u2192  county SEEN",
        ha="center", va="center", fontsize=8.5, color=INK)

# Box 3: Protocol C
ax.add_patch(FancyBboxPatch((23, 23.5), 24, 11,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc=LPEACH, ec=ACCENT, lw=1.5))
ax.text(35, 32.0, "Protocol C \u2014 LOCO",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT)
ax.text(35, 27.5, "8 folds, seed 0, 7, 42; 512\u2013624\n train / 15% val / all of\n county X  \u2192  county UNSEEN",
        ha="center", va="center", fontsize=8.5, color=INK)

# Box 4: Identical training
ax.add_patch(FancyBboxPatch((52, 31), 19, 17,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc="#FFFFFF", ec=NAVY, lw=1.5))
ax.text(61.5, 45.0, "Identical training",
        ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
ax.text(61.5, 40.0,
        "YOLO26n, 256 px\n AdamW, \u2264200\n epochs\n (patience 30);\n plus a ResNet-18\n classifier replication",
        ha="center", va="center", fontsize=8.2, color=INK)

# Box 5: Evaluation
ax.add_patch(FancyBboxPatch((75, 31), 24, 17,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc=LBLUE, ec=NAVY, lw=1.5))
ax.text(87, 45.0, "Evaluation on\n identical test images",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
# 与 §4.4 完全一致的措辞（TPR → recall）
ax.text(87, 38.0,
        "mAP50, mAP50-95,\n size-stratified AP50,\n classification AP / F1 /\n recall / FPR, FPPI and\n image-level FPR at\n six thresholds",
        ha="center", va="center", fontsize=7.7, color=INK)

# 箭头 (Bijie -> B / C)
ax.annotate("", xy=(23, 42.5), xytext=(20, 42.5),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
ax.annotate("", xy=(23, 29.0), xytext=(20, 36.0),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
# 箭头 (B / C -> Identical training)
ax.annotate("", xy=(52, 42.5), xytext=(47, 42.5),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
ax.annotate("", xy=(52, 36.0), xytext=(47, 29.0),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
# 箭头 (Identical -> Evaluation)
ax.annotate("", xy=(75, 39.5), xytext=(71, 39.5),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))

# ---- 第二行：四个 box ----
# Box 6: Paired contrast
ax.add_patch(FancyBboxPatch((75, 13.5), 24, 13,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc=LBLUE, ec=NAVY, lw=1.5))
ax.text(87, 24.0, "Paired contrast",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
ax.text(87, 18.5,
        "B \u2212 C per county on the\n same test images: leakage\n with bootstrap 95% CI\n and exact Wilcoxon\n signed-rank p-value",
        ha="center", va="center", fontsize=7.7, color=INK)

# Box 7: Controls against training-set confound
ax.add_patch(FancyBboxPatch((52, 13.5), 19, 13,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc="#FFFFFF", ec=NAVY, lw=1.5))
ax.text(61.5, 23.0, "Controls against\na training-set\nconfound",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
ax.text(61.5, 17.0,
        "C1: in-domain parity\n of the two families\n C2: within-C,\n in-domain vs\n withheld county",
        ha="center", va="center", fontsize=7.7, color=INK)

# Box 8: Robustness
ax.add_patch(FancyBboxPatch((26, 13.5), 24, 13,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc="#FFFFFF", ec=NAVY, lw=1.5))
ax.text(38, 23.0, "Robustness of the account",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
ax.text(38, 17.5,
        "IoU sweep 0.50\u20130.95;\n training seeds 0 / 7 / 42;\n P2 branch replication\n under LOCO",
        ha="center", va="center", fontsize=8.0, color=INK)

# Box 9: Conclusion
ax.add_patch(FancyBboxPatch((0.8, 13.5), 19, 13,
                             boxstyle="round,pad=0.2,rounding_size=0.7",
                             fc=LPEACH, ec=ACCENT, lw=1.5))
ax.text(10.3, 23.0, "Conclusion",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color=ACCENT)
ax.text(10.3, 17.5,
        "the reported metric\n decides whether\n cross-county leakage\n is visible at all",
        ha="center", va="center", fontsize=8.2, color=INK)

# 箭头: Evaluation -> Paired
ax.annotate("", xy=(87, 26.5), xytext=(87, 31),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
# 箭头: Paired -> Controls
ax.annotate("", xy=(71, 20.0), xytext=(75, 20.0),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
# 箭头: Controls -> Robustness
ax.annotate("", xy=(50, 20.0), xytext=(52, 20.0),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))
# 箭头: Robustness -> Conclusion
ax.annotate("", xy=(20, 20.0), xytext=(26, 20.0),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2,
                            mutation_scale=14))

# ---- 保存 ----
BASE = r"figures"
out_pdf = os.path.join(BASE, "fig2_flow3.pdf")
out_png = os.path.join(BASE, "fig2_flow3.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
plt.close(fig)
print("wrote:", out_pdf, os.path.getsize(out_pdf), "bytes")
print("wrote:", out_png, os.path.getsize(out_png), "bytes")
from PIL import Image
im = Image.open(out_png)
print("size:", im.size)
