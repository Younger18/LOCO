#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Binary classification: landslide vs non-landslide (the task framing of Ji et al., 2020).

ResNet-18 (torchvision, ImageNet-pretrained) fine-tuned on Bijie patches.
Reported: AP (average precision, threshold-free), F1 at the operating point that
maximises F1 on validation, plus TPR on landslide patches and FPR on non-landslide
patches reported SEPARATELY (because test-set class balance varies across LOCO folds,
a single F1 is not comparable between folds).

Usage: python train_cls.py <datadir> [imgsz] [epochs] [batch]
"""
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))


def _to_pos_landslide(ds):
    """ImageFolder sorts alphabetically -> 'landslide'=0, 'non_landslide'=1.
    Remap so that label 1 == landslide, i.e. the POSITIVE class of this study."""
    assert ds.class_to_idx.get('landslide', None) == 0, ds.class_to_idx
    ds.samples = [(p, 1 - l) for p, l in ds.samples]
    ds.targets = [1 - t for t in ds.targets]
    return ds


def build_loaders(root, imgsz, batch):
    tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr = _to_pos_landslide(datasets.ImageFolder(os.path.join(root, 'train'), transform=tf))
    va = _to_pos_landslide(datasets.ImageFolder(os.path.join(root, 'val'), transform=tf))
    te = _to_pos_landslide(datasets.ImageFolder(os.path.join(root, 'test'), transform=tf))
    # num_workers=0: on Windows the worker spawn per epoch is very slow and starves the GPU
    dl = lambda d, sh: DataLoader(d, batch_size=batch, shuffle=sh, num_workers=0,
                                  pin_memory=torch.cuda.is_available())
    return dl(tr, True), dl(va, False), dl(te, False), (tr, va, te)


@torch.no_grad()
def collect(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(dev)
        p = torch.softmax(model(x), 1)[:, 1]
        probs.append(p.cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def prf(p, y, thr):
    tp = float(((p >= thr) & (y == 1)).sum())
    fp = float(((p >= thr) & (y == 0)).sum())
    fn = float(((p < thr) & (y == 1)).sum())
    tn = float(((p < thr) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {'P': prec, 'R': rec, 'F1': f1,
            'TPR': rec,
            'FPR': fp / (fp + tn) if fp + tn > 0 else 0.0}


def ap_score(p, y):
    order = p.argsort()[::-1]
    y = y[order]
    tp = (y == 1).astype(float)
    fp = (y == 0).astype(float)
    ctp, cfp = tp.cumsum(), fp.cumsum()
    rec = ctp / max(tp.sum(), 1)
    prec = ctp / (ctp + cfp + 1e-8)
    ap = 0.0
    for t in __import__('numpy').linspace(0, 1, 101):
        m = rec >= t
        ap += (prec[m].max() if m.any() else 0.0) / 101.0
    return float(ap)


def main():
    datadir = sys.argv[1] if len(sys.argv) > 1 else 'cls_random'
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 224
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 64

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(HERE, datadir)
    tr_l, va_l, te_l, (tr, va, te) = build_loaders(root, imgsz, batch)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(dev)

    n_pos = max(sum(1 for _, c in tr.samples if c == 1), 1)      # landslide
    n_neg = max(len(tr.samples) - n_pos, 1)                      # non-landslide
    # label 1 (landslide) is the minority -> up-weight it
    w = torch.tensor([1.0, float(n_neg) / float(n_pos)]).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_l:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        sch.step()
        pv, yv = collect(model, va_l, dev)
        thrs = __import__('numpy').unique(
            __import__('numpy').quantile(pv, __import__('numpy').linspace(0, 1, 101)))
        bestv = max((prf(pv, yv, t)['F1'], t) for t in thrs)
        if bestv[0] > best_f1:
            best_f1, best_thr, bad = bestv[0], bestv[1], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print('  %s ep%02d loss=%.4f valF1=%.4f thr=%.3f'
              % (datadir, ep + 1, tot / n, best_f1, best_thr), flush=True)
        if bad >= 8:
            break

    if best_state:
        model.load_state_dict(best_state)
        torch.save({'state_dict': best_state, 'val_f1': best_f1, 'thr': float(best_thr)},
                   os.path.join(HERE, 'cls_results', datadir + '.pt'))
    pt, yt = collect(model, te_l, dev)
    res = prf(pt, yt, best_thr)
    res['AP'] = ap_score(pt, yt)
    res.update({'thr': float(best_thr), 'epochs_run': ep + 1,
                'n_test': len(yt), 'n_pos_test': int((yt == 1).sum()),
                'n_neg_test': int((yt == 0).sum())})
    os.makedirs(os.path.join(HERE, 'cls_results'), exist_ok=True)
    __import__('json').dump(res, open(os.path.join(HERE, 'cls_results', datadir + '.json'), 'w'), indent=2)
    print('  [RESULT] %-14s AP=%.4f F1=%.4f P=%.4f R=%.4f TPR=%.4f FPR=%.4f (n_pos=%d n_neg=%d)'
          % (datadir, res['AP'], res['F1'], res['P'], res['R'], res['TPR'], res['FPR'],
             res['n_pos_test'], res['n_neg_test']), flush=True)


if __name__ == '__main__':
    main()
