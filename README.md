# Spatial Leakage in Object-Centric Landslide Detection

Reproducibility package for a leave-one-county-out (LOCO) evaluation of spatial leakage in object-centric (bounding-box) landslide detection on the Bijie landslide benchmark.

The question this package addresses: **how large is the apparent accuracy advantage obtained when training and testing patches are drawn randomly from the same region, and how much of that advantage depends on which evaluation metric is reported?**

## Leakage quantification

Two evaluation protocols share the **identical test set** (all patches of a held-out county):

| Protocol | Training data | Interpretation |
|---|---|---|
| **B — seen** | Random split over all counties (the model has seen the held-out county during training) | Optimistic |
| **C — unseen** | Leave-one-county-out (the model never saw the held-out county) | Unbiased |

Leakage is defined as `B − C`, computed per county and pooled.

The eight counties are the units of cross-validation; for fold X, test = all patches of county X, and the remaining seven counties are split train/val = 85/15 (seed 42). A single random split over all patches (train 70 / val 10 / test 20, seed 42) reproduces the optimistic protocol for comparison.

### Metric-dependent magnitude

Because the detector letterboxes every patch to a fixed input size, the effect is strongly metric-dependent:

- **mAP50** is near-saturated on this benchmark, so the leakage signal is weak.
- **mAP50-95** and **F1** separate the two protocols much more clearly.
- The gap widens with object size, when stratified by the *effective* area the network actually processes (`area_native × s²`, `s = min(imgsz/W, imgsz/H)`).

### False-positive behaviour on non-landslide patches

A set of 2,003 non-landslide patches is evaluated under the same settings. Two quantities are reported:

```
FPPI = false-positive boxes / number of negative patches
FPR  = fraction of negative patches with >= 1 detection
```

Both are strongly confidence-threshold dependent, which is why thresholds are swept rather than fixed at a single value.

## Layout

```
data/canonical result JSON — the single source of truth for every number
code/
  pipeline/   splitting, training, evaluation, statistics
  figures/    scripts that render each figure from data/
figures/     rendered figures (PNG, plus one vector PDF)
```

## Requirements

Python 3.9 with `ultralytics` and `torch` (CUDA build). The scripts were run with Python 3.9.23, `ultralytics` 8.4.173 and `torch` 2.8.0 (CUDA 12.8) on an NVIDIA GPU.

All dataset paths are configured at the top of each script (`ROOT`, `OUT`, `DATADIR`) as plain relative or dataset-root names; adjust them to point at the local dataset copy.

## Pipeline

| Script | Purpose |
|---|---|
| `verify_data.py` | Integrity check of the dataset and split manifests |
| `make_split.py` | Build YOLO-format datasets: LOCO folds, random split, negative-patch list. Polygons become axis-aligned boxes, one per file, normalised to [0, 1] |
| `train_loco.py` | Train one model per LOCO fold |
| `train_multiseed.py` | Repeat the protocol under several random seeds for paired contrasts |
| `run_loco.py` | Drive a full cross-validation sweep |
| `eval_loco.py` | Evaluate a trained model — `python eval_loco.py loco <tag> <imgsz>` for detection metrics and size stratification, `python eval_loco.py fppi <tag> <imgsz>` for the false-positive study |
| `eval_fppi_thresholds.py` | FPPI and FPR across confidence thresholds |
| `eval_iou_sweep.py` | IoU threshold sweep from 0.50 to 0.95 |
| `eval_multiseed.py` | Paired per-seed contrasts |
| `compare_p2.py` | Comparison of the high-resolution P2 branch under LOCO |
| `train_cls.py`, `eval_cls.py`, `make_cls_split.py` | Classification controls on the same folds |
| `scale_analysis.py`, `stats_revision.py` | Size-stratified analysis and statistics |
| `summarize.py` | Aggregate per-fold JSONs into the canonical result files |

Figures are drawn with `matplotlib` from the JSON results in `data/`. `code/figures/make_figS1_qualitative_inference.py` runs the detector over the test and negative patches; `make_figS1_qualitative_cases.py` renders the qualitative comparison panel.

## Key numbers (seed 42)

- Paired leakage B − C: **mAP50 +0.0144**, **mAP50-95 +0.1147** (bootstrap 95% CI [0.090, 0.139]), **F1 +0.0915**. Across three seeds the mAP50-95 effect spans +0.0205 to +0.1413 (mean 0.0922 ± 0.0634), i.e. 3.8×–11.1× the mAP50 effect.
- mAP50 saturation: 0.9891 in-domain versus 0.9784 cross-county; in-domain mAP50-95 0.7089.
- P2 high-resolution branch under LOCO: −0.0258 mAP50-95 (one-sided p = 0.055, CI [−0.048, −0.002]).
- Negative patches: 52% fire at confidence 0.50; FPPI 11–22; negative hit rate 99.7–100.0% over all 2,003 patches at confidence 0.001.
- Training cost 281–614 s per model (mean 492 s); six of nine models early-stopped before the 200-epoch budget.

## Citation

Yongzhu Xiong, Mingyong Zhu, Youcun Liu, Haifeng Huang.

```
(a) School of Geography and Tourism, Jiaying University, Meizhou, Guangdong, China
(b) National Field Observation and Research Station of Landslides in the Three Gorges
    Reservoir Area of the Yangtze River, China Three Gorges University, Yichang 443002, Hubei, China
```

The Bijie landslide dataset is used under courtesy of its original providers.

---

*Repository prepared with the assistance of the WorkBuddy AI assistant; all content reviewed by the authors.*