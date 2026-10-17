#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E0b — build the CLASSIFICATION splits (landslide vs non-landslide), which is the task
framing of the reference paper (Ji et al., 2020) and therefore the one that permits a
direct comparison with their reported F1 = 96.62 %.

Design
------
Positives : 770 landslide patches (county-coded)
Negatives : 2003 non-landslide patches (85 % carry no county code -> cannot be held out
            by county; we therefore use ONE shared negative test set for every fold, so
            that folds remain mutually comparable)

Random split (reference protocol): stratified 70/10/20 over positives AND negatives.
LOCO folds:  test positives = all patches of county X; test negatives = the shared
             negative test set (identical for all folds);
             train/val = remaining counties' positives + the shared negative train/val.
"""
import os
import re
import json
import random
import shutil
import collections

ROOT = r'bijie-landslide-dataset'
OUT = r'.'
SEED = 42

POS = os.path.join(ROOT, 'landslide', 'image')
NEG = os.path.join(ROOT, 'non-landslide', 'image')
COUNTIES = ['qx', 'qxg', 'zj', 'hz', 'js', 'ny', 'wn', 'df']
MERGE = {'zdf': 'df'}


def prefix_of(stem):
    m = re.match(r'^([a-zA-Z]+)', stem)
    return m.group(1).lower() if m else '?'


def emit(dst, pos, neg):
    for sp in ('train', 'val', 'test'):
        for c in ('landslide', 'non_landslide'):
            os.makedirs(os.path.join(dst, sp, c), exist_ok=True)
    for sp, names in pos.items():
        for s in names:
            shutil.copy2(os.path.join(POS, s + '.png'), os.path.join(dst, sp, 'landslide', s + '.png'))
    for sp, names in neg.items():
        for s in names:
            shutil.copy2(os.path.join(NEG, s + '.png'), os.path.join(dst, sp, 'non_landslide', s + '.png'))


def main():
    pos_all = sorted(os.path.splitext(f)[0] for f in os.listdir(POS) if f.endswith('.png'))
    neg_all = sorted(os.path.splitext(f)[0] for f in os.listdir(NEG) if f.endswith('.png'))

    county = {}
    for s in pos_all:
        p = prefix_of(s)
        county[s] = MERGE.get(p, p)
    by_county = collections.defaultdict(list)
    for s in pos_all:
        by_county[county[s]].append(s)

    # ---- shared negative partitions (identical for every fold) ----
    r0 = random.Random(SEED)
    neg_shuf = sorted(neg_all)
    r0.shuffle(neg_shuf)
    n_ntest = int(round(0.20 * len(neg_shuf)))
    n_nval = int(round(0.10 * len(neg_shuf)))
    neg_test = neg_shuf[:n_ntest]
    neg_val = neg_shuf[n_ntest:n_ntest + n_nval]
    neg_train = neg_shuf[n_ntest + n_nval:]

    manifest = {'n_pos': len(pos_all), 'n_neg': len(neg_all), 'folds': {}, 'random': {}}

    # ---- LOCO folds ----
    for c in COUNTIES:
        ptest = sorted(by_county[c])
        rest = sorted(s for s in pos_all if county[s] != c)
        rng = random.Random(SEED)
        rng.shuffle(rest)
        n_val = max(int(round(0.15 * len(rest))), 10)
        pval = rest[:n_val]
        ptrain = rest[n_val:]
        dst = os.path.join(OUT, 'cls_fold_' + c)
        emit(dst, {'train': ptrain, 'val': pval, 'test': ptest},
             {'train': neg_train, 'val': neg_val, 'test': neg_test})
        manifest['folds'][c] = {'n_pos_test': len(ptest), 'n_pos_val': len(pval),
                                'n_pos_train': len(ptrain),
                                'n_neg_test': len(neg_test), 'n_neg_train': len(neg_train),
                                'n_neg_val': len(neg_val)}

    # ---- random split (reference protocol) ----
    rng2 = random.Random(SEED)
    ps = sorted(pos_all)
    rng2.shuffle(ps)
    ns = sorted(neg_all)
    rng2.shuffle(ns)
    p_te, p_va = int(round(0.2 * len(ps))), int(round(0.1 * len(ps)))
    n_te, n_va = int(round(0.2 * len(ns))), int(round(0.1 * len(ns)))
    dst = os.path.join(OUT, 'cls_random')
    emit(dst, {'train': ps[p_te + p_va:], 'val': ps[p_te:p_te + p_va], 'test': ps[:p_te]},
         {'train': ns[n_te + n_va:], 'val': ns[n_te:n_te + n_va], 'test': ns[:n_te]})
    manifest['random'] = {'n_pos_test': p_te, 'n_pos_train': len(ps) - p_te - p_va,
                          'n_neg_test': n_te, 'n_neg_train': len(ns) - n_te - n_va,
                          'test_pos_counties': dict(collections.Counter(county[s] for s in ps[:p_te]))}

    json.dump(manifest, open(os.path.join(OUT, 'cls_manifest.json'), 'w'), indent=2, ensure_ascii=False)
    print('pos=%d neg=%d' % (len(pos_all), len(neg_all)))
    for c in COUNTIES:
        d = manifest['folds'][c]
        print('  cls_fold_%-4s pos: train=%3d val=%3d test=%3d | neg: train=%d test=%d'
              % (c, d['n_pos_train'], d['n_pos_val'], d['n_pos_test'], d['n_neg_train'], d['n_neg_test']))
    print('random: pos test=%d train=%d | neg test=%d train=%d'
          % (manifest['random']['n_pos_test'], manifest['random']['n_pos_train'],
             manifest['random']['n_neg_test'], manifest['random']['n_neg_train']))
    print('random test positives by county:', manifest['random']['test_pos_counties'])
    print('[done] wrote cls_manifest.json')


if __name__ == '__main__':
    main()
