#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figures for the study.

All text is rendered in Times New Roman throughout.
No figure-level title is drawn inside the image (the figure name/caption
lives in the caption below the figure); multi-panel figures carry letter
labels (a), (b), ... so panels can be cited from the text.

Fig 2  per-fold training curves (panels a-i)
Fig 3  leakage by metric, with bootstrap CI (the paper's core result)
Fig 4  B vs C per county, mAP50 vs mAP50-95 (panels a-b)
Fig 5  object-size distribution vs training input size (not used in v2)
Fig 6  false-alarm behaviour vs operating threshold (panels a-b)
Fig 7  the P2 branch under LOCO (panels a-b)

Outputs: paper/figs/*.png (300 dpi)
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'paper', 'figs')
os.makedirs(FIGS, exist_ok=True)

COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
COUNTY_LABEL = {'qx': 'Qianxi', 'qxg': 'Qixingguan', 'zj': 'Zhijin', 'hz': 'Hezhang',
                'js': 'Jinsha', 'ny': 'Nayong', 'wn': 'Weining', 'df': 'Dafang'}

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 10,
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white',
})
C_B, C_C = '#1f77b4', '#d62728'


def read_curve(name):
    p = os.path.join(HERE, 'runs', 'detect', name, 'results.csv')
    if not os.path.exists(p):
        return None
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(p))]
    ep = np.array([float(r['epoch']) for r in rows])
    m50 = np.array([float(r['metrics/mAP50(B)']) for r in rows])
    m95 = np.array([float(r['metrics/mAP50-95(B)']) for r in rows])
    return ep, m50, m95


# ---------------------------------------------------------------- Fig 2
def fig_convergence():
    names = [('random_split', 'random split (B)')] + [('fold_' + c, 'LOCO %s' % COUNTY_LABEL[c]) for c in COUNTIES]
    letters = 'abcdefghi'
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.2), sharex=False)
    for k, (ax, (nm, title)) in enumerate(zip(axes.ravel(), names)):
        if nm.startswith('random_split'):
            d = read_curve('n_random_split_256')
        else:
            d = read_curve('n_fold_%s_256' % nm.split('_')[1])
        if d is None:
            ax.set_visible(False)
            continue
        ep, m50, m95 = d
        ax.plot(ep, m50, color=C_B, lw=1.2, label='mAP50')
        ax.plot(ep, m95, color=C_C, lw=1.2, label='mAP50-95')
        bi = int(np.argmax(m95))
        ax.axvline(ep[bi], color='0.5', ls=':', lw=0.8)
        ax.scatter([ep[bi]], [m95[bi]], s=12, color=C_C, zorder=5)
        ax.set_title('(%s) %s\n(best@%d, ran %d epochs)' % (letters[k], title, int(ep[bi]), int(ep[-1])),
                     fontsize=7.5)
        ax.set_ylim(0.5, 1.02)
        ax.set_xlim(0, max(200, ep[-1] + 5))
    for ax in axes[-1]:
        ax.set_xlabel('epoch')
    for ax in axes[:, 0]:
        ax.set_ylabel('validation metric')
    axes[0, 0].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig2_convergence.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------- Fig 3
def fig_leakage():
    S = json.load(open(os.path.join(HERE, 'supp_stats.json')))
    recs = S['paired']
    order = ['classification AP', 'classification F1', 'classification recall',
             'classification FPR (rise)', 'detection mAP50', 'detection sizeAP-all@0.5',
             'detection mAP50-95']
    recs = [r for r in recs if r['label'] in order]
    recs = sorted(recs, key=lambda r: order.index(r['label']))
    labels, means, los, his = [], [], [], []
    for r in recs:
        labels.append(r['label'].replace('classification ', 'classif. ').replace('detection ', 'detect. '))
        means.append(r['mean'])
        los.append(r['mean'] - r['boot95'][0])
        his.append(r['boot95'][1] - r['mean'])
    y = np.arange(len(labels))[::-1]
    colors = [C_B] * 4 + [C_C] * 3
    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    # error bars drawn behind the bars; value labels placed clear of the whiskers
    ax.barh(y, means, xerr=[los, his], color=colors, height=0.6, zorder=3,
            error_kw=dict(ecolor='0.25', lw=1, capsize=3, zorder=2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel('leakage  B (county seen)  \u2212  C (county withheld)')
    for yy, m, h in zip(y, means, his):
        ax.text(m + h + 0.005, yy, '%+.4f' % m, va='center', fontsize=9, zorder=5)
    ax.set_xlim(0, max(m + h for m, h in zip(means, his)) + 0.055)
    ax.axvspan(0.0144 - 0.003, 0.0144 + 0.003, color='0.85', zorder=0)
    ax.annotate('shaded band: the whole leakage as mAP50 sees it', xy=(0.0245, 5.55),
                fontsize=8.5, color='0.3')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_B, label='classification'),
                       Patch(color=C_C, label='detection')],
              fontsize=8.5, loc='center right', bbox_to_anchor=(0.99, 0.5),
              framealpha=0.92, edgecolor='0.6')
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig3_leakage_by_metric.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------- Fig 4
def fig_per_county():
    D = json.load(open(os.path.join(HERE, 'loco_results_n_256.json')))['counties']
    x = np.arange(len(COUNTIES))
    b50 = [D[c]['B_seen']['official']['mAP50'] for c in COUNTIES]
    c50 = [D[c]['C_unseen']['official']['mAP50'] for c in COUNTIES]
    b95 = [D[c]['B_seen']['official']['mAP50-95'] for c in COUNTIES]
    c95 = [D[c]['C_unseen']['official']['mAP50-95'] for c in COUNTIES]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=False)
    specs = [(b50, c50, 'mAP50 (saturated)'), (b95, c95, 'mAP50-95 (sensitive)')]
    for k, (ax, (bb, cc, ttl)) in enumerate(zip(axes, specs)):
        w = 0.38
        ax.bar(x - w / 2, bb, w, color=C_B, label='B (county seen)')
        ax.bar(x + w / 2, cc, w, color=C_C, label='C (county withheld)')
        ax.set_xticks(x)
        ax.set_xticklabels([COUNTY_LABEL[c] for c in COUNTIES], rotation=35, ha='right', fontsize=8)
        ax.set_title('(%s) %s' % ('ab'[k], ttl), fontsize=9)
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel('score')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig4_per_county.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------- Fig 5 (unused in v2)
def fig_size_dist():
    S = json.load(open(os.path.join(HERE, 'scale_analysis.json')))
    rows = [{'imgsz': 'native', 'small': 82, 'medium': 393, 'large': 295, 'scale': 1.00}] + [
        {'imgsz': str(r['imgsz']), 'small': r['small'], 'medium': r['medium'],
         'large': r['large'], 'scale': r['mean_scale']} for r in S['imgsz_effect']]
    tot = 770
    labels = ['%s\n%.2f\u00d7' % (r['imgsz'], r['scale']) for r in rows]
    small = np.array([r['small'] / tot * 100 for r in rows])
    med = np.array([r['medium'] / tot * 100 for r in rows])
    lar = np.array([r['large'] / tot * 100 for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x, small, 0.62, color='#8ecae6', label='small (<32\u00b2)')
    ax.bar(x, med, 0.62, bottom=small, color='#f6bd60', label='medium (32\u00b2\u201396\u00b2)')
    ax.bar(x, lar, 0.62, bottom=small + med, color='#e76f51', label='large (>96\u00b2)')
    for i, r in enumerate(rows):
        if r['small']:
            ax.text(i, r['small'] / tot * 100 / 2, str(r['small']), ha='center', va='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('% of the 770 instances')
    ax.set_xlabel('training input size (px) and mean linear rescale factor')
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig_unused_size_distribution.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------- Fig 6
def fig_fppi():
    p = os.path.join(HERE, 'fppi_thresholds_n_256.json')
    if not os.path.exists(p):
        print('  [skip] fppi_thresholds_n_256.json not found')
        return
    D = json.load(open(p))
    confs = ['0.001', '0.010', '0.050', '0.100', '0.250', '0.500']
    style = {'random_split': (C_B, '-', 'o'), 'fold_qx': (C_C, '--', 's'),
             'fold_qxg': ('#2ca02c', '-.', '^'), 'fold_zj': ('#9467bd', ':', 'D')}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for name, rec in D.items():
        col, ls, mk = style.get(name, ('0.5', '-', 'o'))
        fppi = [rec[c]['FPPI'] for c in confs]
        fpr = [rec[c]['FPR_images'] * 100 for c in confs]
        lab = name.replace('random_split', 'random split').replace('fold_', 'LOCO ')
        axes[0].plot(range(len(confs)), fppi, color=col, ls=ls, marker=mk, ms=4, lw=1.2, label=lab)
        axes[1].plot(range(len(confs)), fpr, color=col, ls=ls, marker=mk, ms=4, lw=1.2, label=lab)
    axes[0].set_yscale('log')
    axes[1].set_ylim(0, 105)
    ttl = ['(a) False positives per image', '(b) Negative images with \u22651 detection']
    ylab = ['FPPI (log scale)', '% of the 500 negative patches']
    for ax, t, yl in zip(axes, ttl, ylab):
        ax.set_xticks(range(len(confs)))
        ax.set_xticklabels(confs, fontsize=8)
        ax.set_xlabel('detection confidence threshold')
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=9)
    axes[1].axhline(1, color='0.4', ls=':', lw=0.9)
    axes[1].annotate('practically useful level', xy=(0.05, 3.2), fontsize=7, color='0.35')
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig6_fppi_thresholds.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


# ---------------------------------------------------------------- Fig 7
def fig_p2():
    p = os.path.join(HERE, 'p2_comparison.json')
    if not os.path.exists(p):
        print('  [skip] p2_comparison.json not found')
        return
    D = json.load(open(p))
    rows = D['per_county']
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    w = 0.38
    axes[0].bar(x - w / 2, [r['n_mAP50_95'] for r in rows], w, color=C_B, label='YOLO26n')
    axes[0].bar(x + w / 2, [r['p2_mAP50_95'] for r in rows], w, color='#ff7f0e', label='YOLO26n + P2')
    axes[0].set_title('(a) mAP50-95 under LOCO (C models)', fontsize=9)
    axes[0].set_ylabel('score')
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([COUNTY_LABEL[c] for c in COUNTIES], rotation=35, ha='right', fontsize=8)
    axes[0].set_xlim(-0.55, 7.55)          # span 8.1
    # leakage by architecture
    L = D['leakage_by_arch']
    lab = ['mAP50', 'mAP50-95']
    xs = np.arange(2)
    # panel (b) spans 2.0 x-units against 8.1 in (a) with equal axes widths, so
    # the bar width is scaled by the span ratio: the bars then have exactly the
    # same visual thickness as the bars in panel (a)
    wb = w * 2.0 / 8.1
    axes[1].bar(xs - wb / 2, [L['n']['mAP50_mean'], L['n']['mAP50_95_mean']], wb,
                color=C_B, label='YOLO26n')
    axes[1].bar(xs + wb / 2, [L['p2']['mAP50_mean'], L['p2']['mAP50_95_mean']], wb,
                color='#ff7f0e', label='YOLO26n + P2')
    # labels staggered vertically so they do not collide at the reduced width
    for i, (a, b) in enumerate(zip([L['n']['mAP50_mean'], L['n']['mAP50_95_mean']],
                                   [L['p2']['mAP50_mean'], L['p2']['mAP50_95_mean']])):
        axes[1].text(i - wb / 2, a + 0.004, '%.4f' % a, ha='center', fontsize=7)
        axes[1].text(i + wb / 2, b + 0.016, '%.4f' % b, ha='center', fontsize=7)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(lab)
    axes[1].set_title('(b) Leakage (B \u2212 C) is unchanged by P2', fontsize=9)
    axes[1].set_ylabel('leakage')
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(-0.5, 1.5)            # span 2.0
    axes[1].set_ylim(0, 0.150)
    fig.tight_layout()
    out = os.path.join(FIGS, 'fig7_p2_under_loco.png')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  wrote', out)


if __name__ == '__main__':
    fig_convergence()
    fig_leakage()
    fig_per_county()
    fig_size_dist()
    fig_fppi()
    fig_p2()
    print('[done] figures in', FIGS)
