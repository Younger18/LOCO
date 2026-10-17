#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classification leakage measurement.

For every county X we evaluate TWO models on the IDENTICAL test set
  test = (all landslide patches of county X)  +  (the shared 401 non-landslide patches)
  B_seen  : the random-split classifier (trained on 70 % of ALL counties' positives,
            therefore it HAS seen county X)
  C_unseen: the LOCO classifier for X (trained with county X positives excluded)
leakage = B - C

Because the negative test set is shared and disjoint from training, and the positive
test set differs only by county, the B/C difference isolates spatial leakage.

Usage: python eval_cls.py
"""
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
IMGSZ = 224
BATCH = 128


def loader_for(test_dir):
    """ImageFolder over <test_dir>/{landslide,non_landslide}; label 1 == landslide (positive).

    ImageFolder sorts classes alphabetically -> 'landslide'=0, 'non_landslide'=1.
    We invert so that y == 1 denotes the POSITIVE (landslide) class.
    """
    tf = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(test_dir, transform=tf)
    assert ds.class_to_idx.get('landslide', None) == 0, ds.class_to_idx
    ds.samples = [(p, 1 - l) for p, l in ds.samples]
    ds.targets = [1 - t for t in ds.targets]
    return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4), len(ds)


@torch.no_grad()
def probs_of(model, loader, dev):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(dev)), 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def labels_of(loader):
    ys = []
    for _, y in loader:
        ys.append(y)
    return torch.cat(ys).numpy()


def metrics(p, y, thr):
    tp = float(((p >= thr) & (y == 1)).sum())
    fp = float(((p >= thr) & (y == 0)).sum())
    fn = float(((p < thr) & (y == 1)).sum())
    tn = float(((p < thr) & (y == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return {'P': P, 'R': R, 'F1': F1, 'TPR': R, 'FPR': fp / (fp + tn) if fp + tn else 0.0,
            'AP': ap(p, y)}


def ap(p, y):
    o = p.argsort()[::-1]
    y = y[o]
    tp = (y == 1).astype(float)
    fp = (y == 0).astype(float)
    ctp, cfp = tp.cumsum(), fp.cumsum()
    rec = ctp / max(tp.sum(), 1)
    prec = ctp / (ctp + cfp + 1e-8)
    return float(sum(prec[rec >= t].max() if (rec >= t).any() else 0.0
                     for t in np.linspace(0, 1, 101)) / 101.0)


def load_model(ckpt, dev):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    c = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(c['state_dict'])
    return m.to(dev), float(c.get('thr', 0.5))


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    R, thrR = load_model(os.path.join(HERE, 'cls_results', 'cls_random.pt'), dev)
    out = {'counties': {}}
    for c in COUNTIES:
        ck = os.path.join(HERE, 'cls_results', 'cls_fold_%s.pt' % c)
        if not os.path.exists(ck):
            print('[skip]', ck)
            continue
        L, thrL = load_model(ck, dev)
        pos_dir = os.path.join(HERE, 'cls_fold_' + c, 'test')
        # positives only within pos_dir/landslide ; negatives within pos_dir/non_landslide
        ld, n = loader_for(pos_dir)
        y = labels_of(ld)
        pB = probs_of(R, ld, dev)
        pC = probs_of(L, ld, dev)
        mB = metrics(pB, y, thrR)
        mC = metrics(pC, y, thrL)
        out['counties'][c] = {
            'n_test': int(n), 'n_pos': int((y == 1).sum()), 'n_neg': int((y == 0).sum()),
            'B_seen': mB, 'C_unseen': mC,
            'leak': {k: mB[k] - mC[k] for k in mB}}
        print('  %-4s n_pos=%3d n_neg=%3d | B: AP=%.4f F1=%.4f TPR=%.4f FPR=%.4f | '
              'C: AP=%.4f F1=%.4f TPR=%.4f FPR=%.4f | dAP=%+.4f'
              % (c, int((y == 1).sum()), int((y == 0).sum()),
                 mB['AP'], mB['F1'], mB['TPR'], mB['FPR'],
                 mC['AP'], mC['F1'], mC['TPR'], mC['FPR'], mB['AP'] - mC['AP']), flush=True)
        del L

    ks = ['AP', 'F1', 'TPR', 'FPR']
    means = {}
    for k in ks:
        b = np.mean([v['B_seen'][k] for v in out['counties'].values()])
        cc = np.mean([v['C_unseen'][k] for v in out['counties'].values()])
        means[k] = {'B_mean': float(b), 'C_mean': float(cc), 'leak_mean': float(b - cc),
                    'C_std': float(np.std([v['C_unseen'][k] for v in out['counties'].values()]))}
    out['mean_over_counties'] = means
    print('\n=== mean over the 8 counties ===')
    for k in ks:
        m = means[k]
        print('  %-4s B=%.4f  C=%.4f +- %.4f  leakage=%+.4f' % (k, m['B_mean'], m['C_mean'], m['C_std'], m['leak_mean']))
    json.dump(out, open(os.path.join(HERE, 'cls_leakage.json'), 'w'), indent=2)
    print('[done] wrote cls_leakage.json')


if __name__ == '__main__':
    main()
