#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Serial orchestrator (single GPU — one subprocess at a time).

Usage: python run_loco.py <tag> <imgsz> <epochs> <dirs_csv>
  python run_loco.py n 256 200 random_split,fold_qx,fold_qxg,fold_zj,fold_hz,fold_js,fold_ny,fold_wn,fold_df
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'python'

tag = sys.argv[1] if len(sys.argv) > 1 else 'n'
imgsz = sys.argv[2] if len(sys.argv) > 2 else '256'
epochs = sys.argv[3] if len(sys.argv) > 3 else '200'
dirs = (sys.argv[4] if len(sys.argv) > 4 else
        'random_split,fold_qx,fold_qxg,fold_zj,fold_hz,fold_js,fold_ny,fold_wn,fold_df').split(',')

log = open(os.path.join(HERE, 'run_loco_%s_%s.log' % (tag, imgsz)), 'a', encoding='utf-8')
t0 = time.time()
for d in dirs:
    cmd = [PY, os.path.join(HERE, 'train_loco.py'), tag, d, imgsz, epochs]
    log.write('\n===== [%s] START %s =====\n' % (time.strftime('%H:%M:%S'), ' '.join(cmd[1:])))
    log.flush()
    r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    log.write('===== [%s] END rc=%d =====\n' % (time.strftime('%H:%M:%S'), r.returncode))
    log.flush()
log.write('\nALL_DONE_%s_%s elapsed=%.0fs\n' % (tag, imgsz, time.time() - t0))
log.flush()
log.close()
print('ALL_DONE_%s_%s' % (tag, imgsz))
