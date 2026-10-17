#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print every number needed for the Results section, in compact tables.

Usage: python summarize.py [tag] [imgsz]
"""
import os
import json
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else 'n'
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 256

CN = {'qx': 'Qianxi', 'qxg': 'Qixingguan', 'zj': 'Zhijin', 'hz': 'Hezhang',
      'js': 'Jinsha', 'ny': 'Nayong', 'wn': 'Weining', 'df': 'Dafang'}


def main():
    p = os.path.join(HERE, 'loco_results_%s_%d.json' % (tag, imgsz))
    if not os.path.exists(p):
        print('missing', p)
        return
    d = json.load(open(p))

    print('### Per-county results (tag=%s, imgsz=%d)' % (tag, imgsz))
    print('| County | n | B_AP50_all | C_AP50_all | leak_all | B_med | C_med | B_lrg | C_lrg | B_mAP50 | C_mAP50 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for c, v in d['counties'].items():
        B, C = v['B_seen']['sizeAP'], v['C_unseen']['sizeAP']
        oB, oC = v['B_seen']['official'], v['C_unseen']['official']
        print('| %s (%s) | %d | %.4f | %.4f | %+.4f | %.4f | %.4f | %.4f | %.4f | %s | %s |'
              % (c, CN.get(c, c), v['n_test'], B['all'], C['all'], B['all'] - C['all'],
                 B['medium'], C['medium'], B['large'], C['large'],
                 '%.4f' % oB['mAP50'] if oB['mAP50'] is not None else 'n/a',
                 '%.4f' % oC['mAP50'] if oC['mAP50'] is not None else 'n/a'))

    pl = d['pooled']
    print('\n### Pooled (each county judged by its own LOCO model)')
    print('| bin | B_seen | C_unseen | leakage | GT |')
    print('|---|---:|---:|---:|---:|')
    for b in ('all', 'small', 'medium', 'large'):
        print('| %s | %.4f | %.4f | %+.4f | %d |'
              % (b, pl['B_seen'][b], pl['C_unseen'][b], pl['leakage'][b], pl['gt_effective'][b]))
    print('\nGT native-area bins:', pl['gt_native'])
    print('GT effective bins  :', pl['gt_effective'])

    # mean over counties
    import numpy as np
    bl = [v['B_seen']['sizeAP']['all'] for v in d['counties'].values()]
    cl = [v['C_unseen']['sizeAP']['all'] for v in d['counties'].values()]
    bm = [v['B_seen']['sizeAP']['medium'] for v in d['counties'].values()]
    cm = [v['C_unseen']['sizeAP']['medium'] for v in d['counties'].values()]
    br = [v['B_seen']['sizeAP']['large'] for v in d['counties'].values()]
    cr = [v['C_unseen']['sizeAP']['large'] for v in d['counties'].values()]
    print('\n### Mean +- std over the 8 counties')
    print('  all    B = %.4f +- %.4f   C = %.4f +- %.4f   leakage = %+.4f'
          % (np.mean(bl), np.std(bl), np.mean(cl), np.std(cl), np.mean(bl) - np.mean(cl)))
    print('  medium B = %.4f +- %.4f   C = %.4f +- %.4f' % (np.mean(bm), np.std(bm), np.mean(cm), np.std(cm)))
    print('  large  B = %.4f +- %.4f   C = %.4f +- %.4f' % (np.mean(br), np.std(br), np.mean(cr), np.std(cr)))

    fp = os.path.join(HERE, 'fppi_%s_%d.json' % (tag, imgsz))
    if os.path.exists(fp):
        f = json.load(open(fp))
        print('\n### False positives on %d non-landslide patches' % list(f.values())[0]['n_images'])
        print('| model | FPPI | FPR(images) | fp_boxes |')
        print('|---|---:|---:|---:|')
        for k, v in sorted(f.items()):
            print('| %s | %.4f | %.4f | %d |' % (k, v['FPPI'], v['FPR_images'], v['fp_boxes']))


if __name__ == '__main__':
    main()
