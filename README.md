# Scopium — WSI Classification Framework

A modular, YAML-configurable framework for Whole Slide Image (WSI) analysis — from raw slide ingestion through tissue segmentation, feature extraction, Multiple Instance Learning (MIL) classification, hyperparameter optimisation (HPO), and cross-validation — for both **single-label** (binary/multiclass) and **multi-label** tasks.

---

## Quick Start

```bash
conda activate dl_py39
cd wsi_framework

# ── Shared preprocessing ────────────────────────────────────────────
python main.py stats    --config config/config.yaml   # scan dataset
python main.py process  --config config/config.yaml   # thumbnails + metadata
python main.py segment  --config config/config.yaml   # tissue segmentation + patches
python main.py extract  --config config/config.yaml   # GPU feature extraction → .pt files

# ── Single-label classification ─────────────────────────────────────
python main.py analyse  --config config/config.yaml
python main.py split    --config config/config.yaml
python main.py train    --config config/config.yaml
python main.py evaluate --config config/config.yaml
python main.py heatmap  --config config/config.yaml

# ── Hyperparameter optimisation ─────────────────────────────────────
python main.py hpo      --config config/config.yaml
python main.py train    --config config/config.yaml --use_best_config

# ── Cross-validation ────────────────────────────────────────────────
python main.py crossval --config config/config.yaml

# ── Multi-label classification ──────────────────────────────────────
python main.py multilabel-split    --config config/config.yaml --csv dataset/annotations/labels.csv
python main.py multilabel-train    --config config/config.yaml
python main.py multilabel-evaluate --config config/config.yaml
python main.py multilabel-hpo      --config config/config.yaml
python main.py multilabel-crossval --config config/config.yaml
```

---

## CLI Command Reference

### Shared Preprocessing

| Command | Description |
|---|---|
| `stats` | Scan `slides_dir` → `dataset_stats.csv` |
| `process` | Thumbnail PNGs + JSON metadata per slide |
| `segment` | Tissue segmentation → patch coordinate `.h5` files |
| `debug-segmentation` | Tile overlay on WSI thumbnail for visual QC |
| `extract-tile` | Single high-res tile extraction |
| `extract` | GPU feature extraction → `.pt` + `.h5` per slide |

### Single-Label Pipeline

| Command | Description |
|---|---|
| `analyse` | Annotation CSV validation + class distribution report |
| `split` | Train/val/test CSV split generation |
| `train` | MIL model training with early stopping + LR scheduling |
| `evaluate` | Metrics, ROC curves, confusion matrix on test set |
| `heatmap` | Attention heatmap overlay + top-20 tile extraction |
| `classify` | Patch-level classifier inference → CSV + filtered features |
| `classify-heatmap` | Tile-level prediction heatmaps |
| `hpo` | Optuna HPO study — searches model arch, LR, regularisation |
| `crossval` | Stratified K-fold cross-validation |

### Multi-Label Pipeline

| Command | Description |
|---|---|
| `multilabel-validate` | Validate label CSV and feature file availability |
| `multilabel-split` | Train/val/test split with iterative stratification |
| `multilabel-train` | Multi-label MIL training (BCE / Focal loss) |
| `multilabel-evaluate` | Per-label metrics, ROC curves, Hamming loss |
| `multilabel-hpo` | Optuna HPO with 19-parameter search space |
| `multilabel-crossval` | K-fold CV with iterative stratification |

### Key CLI Flags

| Flag | Applies to | Description |
|---|---|---|
| `--features <dir>` | all MIL commands | **Exact** feature directory (model suffix not appended) |
| `--csv <path>` | `analyse`, `split`, `multilabel-split`, `multilabel-validate` | Label CSV path |
| `--experiment <dir>` | `evaluate`, `heatmap`, `multilabel-evaluate` | Use a specific experiment directory |
| `--use_best_config` | `train`, `crossval`, `multilabel-train`, `multilabel-crossval` | Merge best HPO config before running |
| `--split <val\|test>` | `multilabel-evaluate` | Which split to evaluate (default: `test`) |

---

## Feature Directory Selection

Feature extraction produces one directory per (patch-config, model) combination:

```
results/features/
    patch512_step512_level0__optimus__TUM/   ← TUM-filtered features
    patch512_step512_level0__optimus/        ← full-slide features
    patch256_step256_level0__uni/
```

**Resolution priority (highest → lowest):**

1. CLI `--features <dir>` — exact dir, no suffix appended
2. `feature_extraction.features_dir_override` in config — exact dir
3. `feature_extraction.features_subfolder_override` + auto-appended model suffix
4. Auto-derived from tiling params + model name

```bash
python main.py train           --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py hpo             --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py crossval        --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-hpo  --config config/config.yaml --features patch512_step512_level0__optimus__TUM
```

---

## Implemented Modules

| Module | Status | Key File |
|---|---|---|
| WSI Reading & Metadata | ✅ | `core/wsi_reader.py` |
| Tissue Segmentation | ✅ | `core/segmenter.py` |
| Patch Extraction | ✅ | `core/patcher.py` |
| Debugging Utilities | ✅ | `pipelines/debug.py` |
| Feature Extraction | ✅ | `pipelines/extract.py` |
| Annotation Analysis | ✅ | `pipelines/analyse_annotations.py` |
| Dataset Splitting (single-label) | ✅ | `pipelines/split.py` |
| MIL Training (single-label) | ✅ | `pipelines/train.py` |
| MIL Evaluation (single-label) | ✅ | `pipelines/evaluate.py` |
| Attention Heatmaps | ✅ | `pipelines/visualize.py` |
| Patch Classification | ✅ | `pipelines/classify.py` |
| Prediction Heatmaps | ✅ | `pipelines/classify_heatmap.py` |
| Hyperparameter Optimisation (single-label) | ✅ | `pipelines/hpo.py` |
| K-Fold Cross-Validation (single-label) | ✅ | `pipelines/crossval.py` |
| **Multi-Label Dataset Splitting** | ✅ | `pipelines/multilabel_split.py` |
| **Multi-Label Dataset** | ✅ | `datasets/mil_multilabel_dataset.py` |
| **Multi-Label MIL Models** | ✅ | `models/mil_multilabel_models.py` |
| **Multi-Label Training** | ✅ | `pipelines/multilabel_train.py` |
| **Multi-Label Evaluation** | ✅ | `pipelines/multilabel_evaluate.py` |
| **Multi-Label HPO** | ✅ | `pipelines/multilabel_hpo.py` |
| **Multi-Label Cross-Validation** | ✅ | `pipelines/multilabel_crossval.py` |
| **Multi-Label CSV Validator** | ✅ | `utils/multilabel_validator.py` |

---

## Supported MIL Models

Set via `mil.model` in `config.yaml` (shared by both pipelines):

| Key | Architecture | Attention |
|---|---|---|
| `mean_pool` | Global Average Pool | No |
| `max_pool` | Global Max Pool | No |
| `abmil` | Gated Attention MIL (Ilse 2018) | Yes |
| `clam_sb` | CLAM Single-Branch (Lu 2021) | Yes |
| `clam_mb` | CLAM Multi-Branch (Lu 2021) | Yes |
| `transmil` | Transformer MIL (Shao 2021) | No |
| `dsmil` | Dual-Stream MIL (Li 2021) | Yes |

> Heatmaps are only generated for attention-based models: `abmil`, `clam_sb`, `clam_mb`, `dsmil`.

---

## Supported Feature Extractors

| Key | Architecture | Embedding Dim |
|---|---|---|
| `rn18` | ResNet-18 | 512 |
| `rn50` | ResNet-50 | 2048 |
| `vit_l` | ViT-Large | 1024 |
| `uni` | UNI | 1024 |
| `provgigapath` | Prov-GigaPath | 1536 |
| `phikon` | Phikon | 768 |
| `hibou_b` | Hibou-B | 768 |
| `hibou_l` | Hibou-L | 1024 |
| `optimus` | H-Optimus-0 | 1536 |
| `virchow` | Virchow | 2560 |

---

## Key Configuration Options

```yaml
# Single-label task
task:
  name: metastasis
  type: binary          # binary | multiclass
  num_classes: 2
  class_names: [benign, malignant]

mil:
  model: abmil           # see model table above
  encoding_size: 1536

training:
  max_epochs: 100
  learning_rate: 0.0002
  weight_decay: 1e-4
  optimizer: AdamW
  lr_scheduler: plateau   # plateau | cosine | step
  early_stopping: true
  early_stopping_patience: 20
  label_smoothing: 0.0
  patch_dropout: 0.0
  patch_shuffle: false

# Multi-label task
multilabel:
  task_name: multilabel_task
  label_names: [TTN, TP53, MUC16]   # ← required: your actual labels
  binary_columns: true               # true = one-column-per-label CSV format
  threshold: 0.5                     # sigmoid decision threshold

multilabel_training:
  max_epochs: 100
  learning_rate: 2.0e-4
  loss: bce              # bce | focal
  weighted_loss: true    # recommended for imbalanced datasets
  monitor_metric: macro_auc

multilabel_split:
  type: train_val_test
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
  random_seed: 42

crossval:
  n_folds: 5
  seed: 42

multilabel_crossval:
  n_folds: 5
  seed: 42
```

---

## Hyperparameter Optimisation (HPO)

HPO uses [Optuna](https://optuna.org/) to automatically search over model architecture, optimizer, regularization, and scheduler.

```bash
pip install optuna tqdm scikit-multilearn
```

### Single-Label HPO

```bash
python main.py hpo     --config config/config.yaml
python main.py hpo     --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py train   --config config/config.yaml --use_best_config
python main.py crossval --config config/config.yaml --use_best_config
```

### Multi-Label HPO

```bash
python main.py multilabel-hpo     --config config/config.yaml
python main.py multilabel-hpo     --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-train   --config config/config.yaml --use_best_config
python main.py multilabel-crossval --config config/config.yaml --use_best_config
```

### HPO Experiment Tracking

Each HPO run creates an **isolated, timestamped directory**. Re-running with different features or configs never overwrites previous studies:

```
results/hpo/
    mil_hpo__patch512_step512_level0__optimus__TUM__20260329_143022/
        experiment_info.json   ← provenance (features, device, task, timestamp)
        base_config.yaml       ← exact config snapshot
        trial_0000/
            trial_config.yaml
            trial_metrics.json
        best_config.yaml       ← best hyperparameters (merged)
        best_trial.json
        hpo_results.csv
        study.db               ← Optuna SQLite (resumable)
    ml_hpo__patch512_step512_level0__optimus__TUM__20260330_094511/
        ...
```

---

## Cross-Validation

Pools train + val splits and runs stratified K-fold, training a fresh model per fold.

### Single-Label CV

```bash
python main.py crossval --config config/config.yaml
python main.py crossval --config config/config.yaml --use_best_config
python main.py crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py crossval --config config/config.yaml --use_best_config --features patch512_step512_level0__optimus__TUM
```

### Multi-Label CV

```bash
python main.py multilabel-crossval --config config/config.yaml
python main.py multilabel-crossval --config config/config.yaml --use_best_config
python main.py multilabel-crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-crossval --config config/config.yaml --use_best_config --features patch512_step512_level0__optimus__TUM
```

### CV Experiment Tracking

Every CV run is saved in a **unique, structured directory** that encodes task, model, feature folder, and timestamp — so multiple runs never overwrite each other:

```
results/
├── crossval/                              ← single-label CV
│   └── <task_name>/
│       └── <model>__<feat_dir>__<YYYYMMDD_HHMMSS>/
│           ├── experiment_info.json       ← provenance metadata
│           ├── config_snapshot.yaml       ← exact config used
│           ├── combined_pool.csv          ← merged train+val pool
│           ├── fold_01/
│           │   ├── best_model.pt
│           │   └── fold_metrics.json
│           ├── ...
│           ├── cv_summary.json            ← mean ± std across folds
│           └── cv_summary.csv
│
└── multilabel/
    └── crossval/                          ← multi-label CV
        └── <task_name>/
            └── <model>__<feat_dir>__<YYYYMMDD_HHMMSS>/
                ├── experiment_info.json
                ├── config_snapshot.yaml
                ├── combined_pool.csv
                ├── fold_01/
                │   ├── best_model.pt
                │   └── fold_metrics.json
                ├── ...
                ├── cv_summary.json
                └── cv_summary.csv
```

**Example run names:**
```
abmil__patch512_step512_level0__optimus__TUM__20260401_103245/
clam_sb__patch256_step256_level0__uni__20260401_120000/
transmil__patch512_step512_level0__optimus__TUM__20260401_143022/
```

---

## Multi-Label Pipeline

### End-to-End Workflow

```bash
# 1. Validate your label CSV
python main.py multilabel-validate --config config/config.yaml --csv dataset/annotations/labels.csv

# 2. Create stratified splits (70/15/15 by default)
python main.py multilabel-split --config config/config.yaml --csv dataset/annotations/labels.csv

# 3. Train
python main.py multilabel-train --config config/config.yaml --features patch512_step512_level0__optimus__TUM

# 4. Evaluate on test set
python main.py multilabel-evaluate --config config/config.yaml

# 5. HPO (30 trials × 30 epochs by default)
python main.py multilabel-hpo --config config/config.yaml --features patch512_step512_level0__optimus__TUM

# 6. Retrain with best config
python main.py multilabel-train --config config/config.yaml --use_best_config

# 7. Cross-validate
python main.py multilabel-crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
```

### CSV Formats

**Format A — Binary columns (one column per label):**
```csv
slide_id,TTN,TP53,MUC16
TCGA-XX-1234,1,0,1
TCGA-XX-5678,0,1,0
```

**Format B — String column:**
```csv
slide_id,labels
TCGA-XX-1234,"TTN,MUC16"
TCGA-XX-5678,TP53
```
Set `multilabel.binary_columns: false` and `multilabel.labels_string_col: labels` in config for Format B.

### Multi-Label Outputs

```
results/multilabel/
├── splits/<task_name>/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── split_summary.txt
├── experiments/<task_name>/<model>_<timestamp>/
│   ├── best_model.pt
│   ├── train_history.csv
│   ├── config_snapshot.yaml
│   ├── plots/
│   │   ├── loss_curve.png
│   │   └── macro_auc_curve.png
│   └── evaluate/
│       ├── metrics.json          ← macro/micro AUC, F1, Hamming, SubsetAcc
│       ├── per_label_metrics.csv
│       └── roc_curves/
│           └── <label>_roc.png
├── hpo/
│   └── ml_hpo__<feat>__<YYYYMMDD_HHMMSS>/
│       ├── experiment_info.json
│       ├── best_config.yaml
│       └── hpo_results.csv
└── crossval/<task_name>/
    └── <model>__<feat>__<YYYYMMDD_HHMMSS>/
        ├── experiment_info.json
        ├── fold_01/
        │   ├── best_model.pt
        │   └── fold_metrics.json
        ├── cv_summary.json
        └── cv_summary.csv
```

---

## Overfitting Mitigation

| Technique | Single-Label Key | Multi-Label Key | HPO-Tunable |
|---|---|---|---|
| Label smoothing | `training.label_smoothing` | `multilabel_training.label_smoothing` | ✅ |
| Patch-level dropout | `training.patch_dropout` | `multilabel_training.patch_dropout` | ✅ |
| Patch shuffling | `training.patch_shuffle` | `multilabel_training.patch_shuffle` | – |
| Bag size cap | `training.max_patches` | `multilabel_training.max_patches` | ✅ |
| Weight decay | `training.weight_decay` | `multilabel_training.weight_decay` | ✅ |
| LR warmup | `training.warmup_epochs` | `multilabel_training.warmup_epochs` | ✅ |
| LR scheduling | `training.lr_scheduler` | `multilabel_training.lr_scheduler` | ✅ |
| Early stopping | `training.early_stopping` | `multilabel_training.early_stopping` | – |
| Focal loss | – | `multilabel_training.loss: focal` | ✅ |
| Per-label pos_weight | – | `multilabel_training.weighted_loss` | – |

---

## Parameter Validation

HPO and cross-validation validate all required config keys **before** starting:

```
HPO: config missing required key 'task.class_names'
HPO aborted due to missing configuration keys.
```

Required (single-label): `mil.model`, `task.name`, `task.num_classes`, `task.class_names`, `training.max_epochs`, `paths.results_dir`

Required (multi-label): `mil.model`, `mil.encoding_size`, `multilabel.label_names`, `multilabel_training.learning_rate`, `paths.results_dir`

---

## Documentation

| File | Contents |
|---|---|
| `docs/setup_and_usage.md` | Full installation, commands, feature dir resolution, HPO, CV |
| `docs/architecture.md` | Pipeline architecture and module relationships |
| `config/config.yaml` | Fully-annotated configuration (all pipeline parameters) |
