# WSI Framework — Setup and Usage Guide

## Contents

1. [Installation](#installation)
2. [Repository Layout](#repository-layout)
3. [Configuration](#configuration)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Single-Label MIL Pipeline](#single-label-mil-pipeline)
6. [Patch-Level Classifier Pipeline](#patch-level-classifier-pipeline)
7. [Hyperparameter Reference](#hyperparameter-reference)
8. [Hyperparameter Optimisation (HPO)](#hyperparameter-optimisation-hpo)
9. [Cross-Validation](#cross-validation)
10. [Multi-Label MIL Pipeline](#multi-label-mil-pipeline)
11. [Feature Directory Selection](#feature-directory-selection)
12. [Results Directory Structure](#results-directory-structure)

---

## Installation

```bash
# Clone
git clone <repo_url> && cd wsi_framework

# Create environment
conda create -n dl_py39 python=3.9 -y
conda activate dl_py39

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install openslide-python h5py timm transformers
pip install scikit-learn scikit-multilearn pandas numpy matplotlib seaborn
pip install optuna pyyaml tqdm

# OpenSlide system library (Linux)
sudo apt-get install openslide-tools libvips-dev
```

### Verify Setup

```bash
python -c "import torch; print(torch.cuda.is_available())"
python main.py --help
```

---

## Repository Layout

```
wsi_framework/
├── main.py                        CLI entry point
├── config/config.yaml             All pipeline configuration (annotated)
├── core/                          WSI IO, segmentation, tiling, feature extraction
├── models/                        MIL model zoo (7 architectures, single + multi-label)
├── datasets/                      Bag datasets (single-label and multi-label)
├── pipelines/                     One file per pipeline command
├── utils/                         Config loading, logging, transforms
└── docs/                          This file + architecture.md
```

---

## Configuration

All parameters live in `config/config.yaml`. A snapshot is saved automatically alongside every experiment output.

```bash
# Use default config
python main.py train --config config/config.yaml

# Point to another config
python main.py train --config experiments/my_config.yaml
```

### Section Overview

| YAML Section | Purpose |
|---|---|
| `paths` | Slides dir, annotations dir, results root |
| `dataset` | Slide file extension |
| `segmentation` | Tissue masking thresholds |
| `filter` | Contour area + hole filtering |
| `tiling` | Patch size, step size, pyramid level |
| `feature_extraction` | Encoder model, batch size, GPU workers |
| `task` | Task name, type, num_classes, class_names |
| `split` | Train/val/test proportions + seed |
| `mil` | MIL model, encoding_size, proj_dim, dropout |
| `training` | Optimizer, LR, betas, eps, scheduler, early stopping, max_patches |
| `hpo` | Optuna study config + search space |
| `crossval` | K-fold settings |
| `multilabel` | Label names, threshold, CSV format |
| `multilabel_training` | ML training config (mirrors training section) |
| `multilabel_hpo` | ML HPO study config + search space |
| `multilabel_crossval` | ML K-fold settings |
| `multilabel_split` | ML train/val/test proportions |

---

## Preprocessing Pipeline

### 1. Dataset Statistics

```bash
python main.py stats --config config/config.yaml
```

Scans `paths.slides_dir` and writes `dataset_stats.csv` (slide count, sizes, format distribution).

### 2. Slide Processing

```bash
python main.py process --config config/config.yaml
```

Generates per-slide thumbnail PNG and JSON metadata.

### 3. Tissue Segmentation

```bash
python main.py segment --config config/config.yaml
```

Produces tissue contour `.h5` files used by the tiler. Key config:

```yaml
segmentation:
  seg_level: -1        # -1 = auto (~64× downsample)
  sthresh: 8           # HSV saturation threshold
  mthresh: 7           # median blur kernel
  close: 4             # morphological closing kernel
  use_otsu: false

filter:
  a_t: 100             # min tissue area (in patch units)
  a_h: 16              # min hole area to keep
  max_n_holes: 8
```

Visual QC:

```bash
python main.py debug-segmentation --config config/config.yaml
```

### 4. Feature Extraction

```bash
python main.py extract --config config/config.yaml
```

```yaml
feature_extraction:
  model: optimus          # rn50 | uni | optimus | virchow | phikon | ...
  batch_size: 256
  num_extraction_workers: 4   # one process per GPU
  dataloader_workers: 4
  use_amp: true
  target_patch_size: -1       # -1 = no resize before transform
  transforms: auto            # see Transform Pipeline section below
```

**Encoder output dimensions:**

| Model | Dim |
|---|---|
| `rn18` | 512 |
| `rn50` | 2048 |
| `uni`, `vit_l`, `hibou_l` | 1024 |
| `phikon`, `hibou_b` | 768 |
| `optimus`, `provgigapath` | 1536 |
| `virchow` | 2560 |

> Always set `mil.encoding_size` to match the encoder you used.

#### Transform Pipeline

The `transforms` key controls **how raw patches are preprocessed** before being fed to the feature extractor. Two syntaxes are supported:

**Syntax 1 — single preset** (original behaviour):

```yaml
feature_extraction:
  transforms: auto              # canonical preset for the chosen model (recommended)
  # transforms: optimus_default # named preset
  # transforms: reinhard        # stain normalisation only, then auto model norm
  # transforms: none            # minimal: ToTensor + ImageNet normalisation
```

**Syntax 2 — sequential cascade** (new):

```yaml
feature_extraction:
  transforms:
    - reinhard           # Step 1: H&E colour normalisation
    - optimus_default    # Step 2: model-specific crop + channel normalise
```

The cascade applies steps **in order**. The first step converts the raw PIL patch to a `float32` tensor `[0, 1]`. Every subsequent step receives and returns a tensor — spatial ops (Resize, CenterCrop) from step 2+ work because modern torchvision ops accept both PIL and Tensor inputs.

**Tensor data flow:**

```
Raw PIL patch
    │
    ├─ Step 1  (pre_ops on PIL) → ToTensor → (post_ops on Tensor)
    │          e.g. CenterCrop(224)    →   float32 [0,1]   →   Reinhard()
    │
    ├─ Step 2  (pre_ops on Tensor) → (post_ops on Tensor)
    │          e.g. Resize(224)     →   Normalize(optimus_stats)
    │
    └─ → float32 Tensor ready for model
```

**Available steps:**

| Category | Step name | Operations |
|---|---|---|
| **Auto** | `auto` | Canonical preset for chosen model |
| **Standard** | `none` / `imagenet` | ToTensor + ImageNet norm |
| **Stain norm** | `reinhard` | Reinhard H&E colour normalisation |
| | `macenko` | Macenko stain separation + normalisation |
| **Foundation models** | `optimus_default` | CenterCrop(224) + Optimus channel norm |
| | `uni_default` | Resize(224) + ImageNet norm |
| | `gigapath_default` | Resize(256) + CenterCrop(224) + ImageNet norm |
| | `hibou_default` | Resize+CenterCrop(224) + Hibou norm |
| | `kaiko_default` | Resize+CenterCrop(224) + Kaiko norm |
| | `gpfm_default` | Resize(224,224) + ImageNet norm |
| | `resnet50lunit_default` | Lunit norm (no resize) |
| | `vitslunit_default` | Resize(224) + Lunit norm |
| | `histo_resnet18` | Half norm (mean=std=0.5) |
| **Spatial blocks** | `resize_224` | Resize(224) only |
| | `resize_256_crop_224` | Resize(256) + CenterCrop(224) |
| | `centercrop_224` | CenterCrop(224) only |
| **Norm only** | `imagenet_norm` | ImageNet channel normalisation |
| | `optimus_norm` | Optimus channel normalisation |
| | `hibou_norm` | Hibou channel normalisation |
| | `lunit_norm` | Lunit channel normalisation |
| | `half_norm` | Mean/std=0.5 normalisation |
| **Augmentation** | `colourjitter` | ColorJitter |
| | `colourjitternorm` | ColorJitter + ImageNet norm |
| | `spatial` | Flip + Affine + ImageNet norm |

**Common cascade recipes:**

```yaml
# H&E normalisation → Optimus  (recommended for colorectal H&E slides)
transforms:
  - reinhard
  - optimus_default

# Macenko → UNI
transforms:
  - macenko
  - uni_default

# Resize → stain normalise → channel normalise (fine-grained control)
transforms:
  - resize_224        # Step 1: spatial resize on PIL
  - reinhard          # Step 2: stain normalise (tensor)
  - optimus_norm      # Step 3: channel normalise (tensor)

# Augmentation + stain normalise + channel normalise
transforms:
  - colourjitter      # Step 1: colour augmentation (PIL)
  - reinhard          # Step 2: H&E normalisation (tensor)
  - optimus_norm      # Step 3: channel normalise (tensor)
```

> **Note:** `reinhard` and `macenko` require `pip install torchstain`. They are skipped at import time — the error is raised only when the step is actually built.

---


## Single-Label MIL Pipeline

### 1. Annotation Analysis

```bash
python main.py analyse --config config/config.yaml --csv dataset/annotations/labels.csv
```

### 2. Dataset Split

```bash
python main.py split --config config/config.yaml
```

```yaml
split:
  type: train_val_test   # train_test | train_val_test
  train_size: 0.7
  val_size: 0.1
  test_size: 0.2
  stratified: true
  random_seed: 42
```

Output: `results/splits/<task_name>/train.csv`, `val.csv`, `test.csv`

### 3. Train

```bash
python main.py train --config config/config.yaml
python main.py train --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py train --config config/config.yaml --use_best_config   # after HPO
```

Output per run:

```
results/experiments/<task>/<model>_<timestamp>/
    best_model.pt
    final_model.pt
    train_history.csv          ← epoch, phase, loss, accuracy, auc, f1, lr, time
    config_snapshot.yaml
    plots/train_loss_curve.png | val_auc_curve.png | val_acc_curve.png | learning_rate.png
```

### 4. Evaluate

```bash
python main.py evaluate --config config/config.yaml
python main.py evaluate --config config/config.yaml --experiment results/experiments/metastasis/abmil_20260402_063229
```

Generates: classification report, ROC curve, confusion matrix.

### 5. Attention Heatmaps

```bash
python main.py heatmap --config config/config.yaml
```

Overlays per-patch attention weights on the WSI thumbnail (supported: `abmil`, `clam_sb`, `clam_mb`, `dsmil`).

---

## Patch-Level Classifier Pipeline

The `classify` → `classify-heatmap` pipeline performs **patch-level tissue classification** using a pretrained patch classifier checkpoint. It is a prerequisite for supplying tissue-specific feature directories (e.g. containing only tumour patches) to any downstream MIL pipeline.

> **Typical use-case:** Train an NCT-CRC-style 8-class patch classifier independently, then apply it here to isolate `TUM` or `STR` patches, creating `patch512_step512_level0__optimus__TUM/` as a feature directory for MIL.

### Step 6 — Patch Inference (`classify`)

```bash
# Classify all slides in the active feature directory
python main.py classify --config config/config.yaml

# Override feature directory (full pipeline output from extract)
python main.py classify --config config/config.yaml \
    --features patch512_step512_level0__optimus
```

**What it does:**
1. Loads a pretrained classifier from `patch_classifier.checkpoint_path`.
2. Iterates over every feature file (`.h5` or `.pt`) in the active feature directory.
3. Runs batched softmax inference over all N patches in each slide.
4. Writes one **prediction CSV** per slide containing coordinates, predicted label, confidence, and per-class probabilities.
5. For each category listed in `filter_categories`, writes a new **filtered feature directory** (`/features/<subfolder>__<CATEGORY>/`) containing only the patches that belong to that tissue class.

**Config:**

```yaml
patch_classifier:
  checkpoint_path: "outputs/checkpoints/BEST_MODEL.pth"  # required
  batch_size: 512         # GPU batch size for inference
  input_format: h5        # 'h5' (preferred — includes coords) or 'pt'
  filter_categories:      # tissue classes to save as filtered feature dirs
    - TUM                 # colorectal adenocarcinoma
    - STR                 # cancer-associated stroma
                          # omit or set to null to skip filtered outputs
```

> If `input_format: h5` is requested but only `pt_files/` exists (or vice versa), the pipeline auto-detects the available format and logs a notice.

**Prediction CSV columns:**

| Column | Description |
|---|---|
| `coord_x` | Patch x-coordinate at level 0 |
| `coord_y` | Patch y-coordinate at level 0 |
| `predicted_label` | Winning tissue class (e.g. `TUM`) |
| `confidence` | Max softmax probability |
| `ADI`, `DEB`, `LYM`, `MUC`, `MUS`, `NOR`, `STR`, `TUM` | Per-class softmax probability |

**Outputs:**

```
results/
├── patch_predictions/<feat_subfolder>/
│   ├── <slide_id>.csv           ← per-patch predictions (one row per patch)
│   └── <slide_id>.csv ...       ← one file per slide
│
└── features/<feat_subfolder>__TUM/
    ├── h5_files/<slide_id>.h5   ← (M, D) features + (M, 2) coords for TUM patches
    └── pt_files/<slide_id>.pt   ← FloatTensor (M, D) for TUM patches
```

Where `M ≤ N` is the patch count for that tissue category.

**Supported tissue categories (NCT-CRC palette):**

| Code | Tissue Type |
|---|---|
| `ADI` | Adipose tissue |
| `DEB` | Debris / necrosis |
| `LYM` | Lymphocytes / immune infiltration |
| `MUC` | Mucus |
| `MUS` | Smooth muscle |
| `NOR` | Normal colon mucosa |
| `STR` | Cancer-associated stroma |
| `TUM` | Colorectal adenocarcinoma (tumour) |

### Step 7 — Tile Prediction Heatmaps (`classify-heatmap`)

```bash
# All slides, all categories
python main.py classify-heatmap --config config/config.yaml

# Single slide only
python main.py classify-heatmap --config config/config.yaml --slide CMU-1

# Override feature directory
python main.py classify-heatmap --config config/config.yaml \
    --features patch512_step512_level0__optimus
```

> **Prerequisites:** Run `classify` first. The heatmap pipeline reads the prediction CSVs and the original WSI files.

**What it does:**
For every selected slide, opens the WSI, reads its prediction CSV, and alpha-blends a per-patch colour overlay onto the WSI thumbnail at the best available pyramid level.

**Visualisation modes:**

| Mode | `category_map` | `confidence` |
|---|---|---|
| **Tile colour** | Fixed class colour (TUM=red, STR=blue, …) | Jet colour scale: red=high, blue=low confidence |
| **Overlay** | All patches from selected categories | All patches, one image per category |
| **Legend** | Colour-coded legend strip appended below | Jet colour bar appended to the right |
| **Extras** | — | Top-K highest-confidence patch crops saved |

**Config:**

```yaml
patch_classifier:
  heatmap:
    slides: all              # all | "CMU-1" | [CMU-1, CMU-2]
    categories: all          # all | "TUM"   | [TUM, STR, LYM]
    mode: category_map       # category_map | confidence
    alpha: 0.50              # overlay opacity: 0.0 = transparent, 1.0 = opaque
    top_k_tiles: 10          # number of top-K tile crops to save (confidence mode only)
```

**Category colour palette:**

| Category | Colour |
|---|---|
| `ADI` | Tan `(210, 180, 140)` |
| `DEB` | Grey `(160, 160, 160)` |
| `LYM` | Purple `(170, 80, 200)` |
| `MUC` | Teal `(60, 190, 170)` |
| `MUS` | Orange `(220, 130, 50)` |
| `NOR` | Green `(80, 180, 80)` |
| `STR` | Blue `(70, 120, 220)` |
| `TUM` | Red `(210, 50, 50)` |

**Outputs:**

```
results/patch_predictions/<feat_subfolder>/heatmaps/
└── <slide_id>/
    ├── <slide_id>_category_map.png         ← category_map mode: all selected classes
    ├── <slide_id>_confidence_TUM.png        ← confidence mode: one PNG per category
    ├── <slide_id>_confidence_STR.png
    └── top10_TUM/
        ├── 01_<slide_id>_x256_y512_conf0.987.png
        ├── 02_<slide_id>_x768_y1024_conf0.973.png
        └── ...                               ← top-K crops at level-0 resolution
```

### End-to-End Classify Workflow

```bash
# 1. Extract patch features for all tissue
python main.py extract --config config/config.yaml

# 2. Run patch-level classifier inference
#    → creates prediction CSVs and filtered feature directories
python main.py classify --config config/config.yaml \
    --features patch512_step512_level0__optimus

# 3. Visualise tissue predictions on WSI thumbnails
python main.py classify-heatmap --config config/config.yaml \
    --features patch512_step512_level0__optimus

# 4. Train MIL using only tumour-enriched features
python main.py split --config config/config.yaml
python main.py train   --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# 5. HPO on tumour features
python main.py hpo     --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# 6. Cross-validate with best config
python main.py crossval --config config/config.yaml --use_best_config \
    --features patch512_step512_level0__optimus__TUM
```

## Hyperparameter Reference

All parameters below are configurable in `config.yaml` and fully tunable via HPO.

### Model (`mil.*`)

| Parameter | Default | Description |
|---|---|---|
| `model` | `abmil` | MIL architecture (`abmil`, `clam_sb`, `clam_mb`, `mean_pool`, `max_pool`, `transmil`, `dsmil`) |
| `encoding_size` | `1536` | Input feature dimension — **must match extractor** |
| `proj_dim` | `512` | Internal projection dimension (all layers after first projection share this) |
| `hidden_dim` | `256` | Attention network hidden size |
| `dropout` | `0.4` | Dropout applied to projection and classifier layers |
| `k_sample` | `8` | CLAM: top/bottom patch count for instance clustering |
| `bag_weight` | `0.7` | CLAM: weight for bag loss vs instance loss |

### Optimiser (`training.*`)

| Parameter | Default | Reference Name | Description |
|---|---|---|---|
| `optimizer` | `AdamW` | — | `Adam` or `AdamW` |
| `learning_rate` | `2e-3` | `lr` | Learning rate |
| `weight_decay` | `1e-3` | `reg` | L2 regularization strength |
| `beta1` | `0.75` | `beta1` | Adam first moment decay rate |
| `beta2` | `0.95` | `beta2` | Adam second moment decay rate |
| `eps` | `1e-2` | `eps` | Adam numerical stability constant |

### LR Scheduler (`training.*`)

| Parameter | Default | Reference Name | Description |
|---|---|---|---|
| `lr_scheduler` | `plateau` | — | `plateau`, `cosine`, or `step` |
| `lr_scheduler_factor` | `0.75` | `lr_factor` | LR decay multiplier on plateau |
| `lr_scheduler_patience` | `20` | `lr_patience` | Epochs before plateau triggers LR reduction |

### Training Loop (`training.*`)

| Parameter | Default | Reference Name | Description |
|---|---|---|---|
| `max_epochs` | `100` | — | Maximum training epochs |
| `early_stopping` | `true` | — | Enable early stopping |
| `early_stopping_patience` | `20` | — | Consecutive non-improving epochs before stopping |
| `early_stopping_min_epochs` | `10` | — | Minimum epochs before early stopping is checked |
| `label_smoothing` | `0.0` | — | Cross-entropy label smoothing (0 = off) |
| `warmup_epochs` | `0` | — | Linear LR warmup over first N epochs |

### Bag-Level Regularisation (`training.*`)

| Parameter | Default | Reference Name | Description |
|---|---|---|---|
| `max_patches` | `800` | `A_patches` | Max patches per slide per epoch |
| `patch_dropout` | `0.0` | — | Fraction of patches randomly dropped per step |
| `patch_shuffle` | `false` | — | Shuffle patch order each step |

---

## Hyperparameter Optimisation (HPO)

### Single-Label HPO

```bash
# Basic
python main.py hpo --config config/config.yaml

# With specific feature directory
python main.py hpo --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# After HPO: train / crossval with best config
python main.py train    --config config/config.yaml --use_best_config
python main.py crossval --config config/config.yaml --use_best_config
```

### HPO Configuration

```yaml
hpo:
  study_name: mil_hpo
  n_trials: 30
  metric: val_auc         # val_auc | val_acc | val_f1
  direction: maximize
  epochs_per_trial: 30
  pruning: true           # MedianPruner eliminates weak trials early
  timeout_hours: null     # null = run all n_trials

  search_space:
    model:               [abmil]               # add more models to tune jointly
    optimizer:           [AdamW, Adam]
    learning_rate:       [1.0e-4, 2.0e-3]      # log-uniform
    weight_decay:        [1.0e-5, 5.0e-3]      # log-uniform
    beta1:               [0.5, 0.99]            # uniform
    beta2:               [0.9, 0.999]           # uniform
    eps:                 [1.0e-8, 1.0e-2]       # log-uniform
    dropout:             [0.1, 0.6]             # uniform
    dropout_attn:        [0.2, 0.5]             # uniform
    dropout_classifier:  [0.1, 0.4]             # uniform
    attn_hidden_dim:     [32, 64, 128, 256]     # categorical
    feature_proj_dim:    [256, 512]             # categorical
    lr_scheduler:        [cosine, step, plateau]
    lr_factor:           [0.1, 0.9]             # uniform
    lr_patience:         [5, 10, 20]            # categorical
    label_smoothing:     [0.0, 0.15]            # uniform
    early_stop_patience: [10, 15, 20]           # categorical
    warmup_epochs:       [0, 2, 5]              # categorical
    patch_dropout:       [0.0, 0.3]             # uniform
    max_patches:         [600, 700, 800, 900, 1000, 1100, 1200]  # categorical (A_patches)
```

> Setting a key to `null` in `search_space` keeps that parameter fixed at its `config.yaml` value.

### HPO Output

```
results/hpo/mil_hpo__patch512_step512_level0__optimus__TUM__20260402_063229/
    experiment_info.json    ← study name, metric, feature dir, n_trials, device, timestamp
    base_config.yaml        ← exact config.yaml used at launch
    trial_0000/
        trial_config.yaml   ← merged config for this trial
        trial_metrics.json  ← per-epoch val metrics
    ...
    best_config.yaml        ← best hyperparameters merged into full config
    best_trial.json         ← best trial: params + val metric value
    hpo_results.csv         ← all trials sortable summary
    study.db                ← Optuna SQLite (resumable with same path)
```

Each HPO run produces a **unique directory** — multiple runs with different feature directories, models, or configs are never overwritten.

---

## Cross-Validation

### Single-Label Cross-Validation

```bash
# Standard 5-fold CV
python main.py crossval --config config/config.yaml

# With specific feature dir
python main.py crossval --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# Using best HPO configuration
python main.py crossval --config config/config.yaml --use_best_config

# Combined
python main.py crossval --config config/config.yaml --use_best_config \
    --features patch512_step512_level0__optimus__TUM
```

```yaml
crossval:
  study_name: crossval
  n_folds: 5
  seed: 42
```

### Cross-Validation Output

Each run is stored in a **content-addressable directory** encoding task, model, feature dir, and timestamp:

```
results/crossval/<task_name>/<model>__<feat_dir>__<YYYYMMDD_HHMMSS>/
    experiment_info.json    ← task, model, feature_dir, n_folds, seed, timestamp
    config_snapshot.yaml    ← exact config used
    combined_pool.csv       ← all available labelled slides (train + val merged)
    fold_01/
        best_model.pt
        fold_metrics.json   ← per-fold: acc, auc, f1, precision, recall
    fold_02/ ... fold_N/
    cv_summary.json         ← mean ± std per metric across folds
    cv_summary.csv
```

---

## Multi-Label MIL Pipeline

### 1. Validate Label Coverage

```bash
python main.py multilabel-validate \
    --config config/config.yaml \
    --csv dataset/annotations/slide_labels.csv \
    --features patch512_step512_level0__optimus__TUM
```

Checks every slide in the CSV has a matching `.pt` feature file. Reports coverage.

### 2. Split (Iterative Stratification)

```bash
python main.py multilabel-split \
    --config config/config.yaml \
    --csv dataset/annotations/slide_labels.csv
```

```yaml
multilabel_split:
  type: train_val_test
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
  random_seed: 42
```

Uses `skmultilearn.model_selection.IterativeStratification` for balanced multi-label splits. Falls back to random if skmultilearn is unavailable (with a warning).

### 3. Train

```bash
python main.py multilabel-train --config config/config.yaml
python main.py multilabel-train --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM
python main.py multilabel-train --config config/config.yaml --use_best_config
```

```yaml
multilabel_training:
  optimizer: AdamW
  learning_rate: 2.0e-3
  weight_decay: 1.0e-3
  beta1: 0.75
  beta2: 0.95
  eps: 1.0e-2
  lr_scheduler: plateau
  lr_scheduler_factor: 0.75
  lr_scheduler_patience: 20
  early_stopping: true
  early_stopping_patience: 20
  loss: bce           # bce | focal
  weighted_loss: true
  label_smoothing: 0.05
  focal_alpha: 0.25
  focal_gamma: 2.0
  monitor_metric: macro_auc
  max_patches: 800    # A_patches
  patch_dropout: 0.0
  patch_shuffle: false
  max_epochs: 100
```

### 4. Evaluate

```bash
python main.py multilabel-evaluate --config config/config.yaml
python main.py multilabel-evaluate --config config/config.yaml --split val
python main.py multilabel-evaluate --config config/config.yaml \
    --experiment results/multilabel/experiments/multilabel_task/abmil_20260402_063229
```

Produces: per-label AUC, macro/micro AUC, Hamming loss, exact-match accuracy, per-label threshold-tuning curves.

### 5. Multi-Label HPO

```bash
python main.py multilabel-hpo --config config/config.yaml
python main.py multilabel-hpo --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM
```

```yaml
multilabel_hpo:
  study_name: ml_hpo
  n_trials: 30
  epochs_per_trial: 30
  n_folds: 1              # 1 = single train/val; >1 = K-fold within each trial
  metric: val_macro_auc
  direction: maximize
  pruning: true
  timeout_hours: null

  search_space:
    model:               [abmil, clam_sb, clam_mb, mean_pool, transmil, dsmil]
    optimizer:           [AdamW, Adam]
    learning_rate:       [1.0e-4, 2.0e-3]
    weight_decay:        [1.0e-5, 5.0e-3]
    beta1:               [0.5, 0.99]
    beta2:               [0.9, 0.999]
    eps:                 [1.0e-8, 1.0e-2]
    dropout:             [0.1, 0.6]
    attn_hidden_dim:     [32, 64, 128, 256]
    feature_proj_dim:    [256, 512]
    lr_scheduler:        [cosine, step, plateau]
    lr_factor:           [0.1, 0.9]
    lr_patience:         [5, 10, 20]
    label_smoothing:     [0.0, 0.15]
    early_stop_patience: [10, 15, 20]
    warmup_epochs:       [0, 2, 5]
    patch_dropout:       [0.0, 0.3]
    patch_shuffle:       [true, false]
    max_patches:         [600, 700, 800, 900, 1000, 1100, 1200]
    loss:                [bce, focal]
    focal_gamma:         [1.0, 3.0]
    threshold:           [0.3, 0.7]
```

### 6. Multi-Label Cross-Validation

```bash
python main.py multilabel-crossval --config config/config.yaml
python main.py multilabel-crossval --config config/config.yaml --use_best_config
python main.py multilabel-crossval --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM
```

```yaml
multilabel_crossval:
  study_name: ml_crossval
  n_folds: 5
  seed: 42
```

Output: `results/multilabel/crossval/<task>/<model>__<feat_dir>__<ts>/`

---

## Feature Directory Selection

**Priority (highest → lowest):**

1. `--features <name>` CLI flag → `results/features/<name>/` (exact, no model suffix)
2. `feature_extraction.features_dir_override` in YAML → exact dir
3. `feature_extraction.features_subfolder_override` + model suffix auto-appended
4. Auto-derived: `patch{size}_step{step}_level{lvl}__{model}/`

```bash
# Select by exact directory name
python main.py train  --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# Or override in YAML
feature_extraction:
  features_dir_override: patch512_step512_level0__optimus__TUM
```

---

## Results Directory Structure

```
results/
│
├── features/
│   └── patch512_step512_level0__optimus__TUM/
│       ├── pt_files/*.pt              ← (N_patches, 1536) embeddings per slide
│       └── h5_files/*.h5              ← patch coords + embeddings
│
├── splits/<task_name>/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── experiments/<task_name>/<model>_<timestamp>/
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── train_history.csv
│   ├── config_snapshot.yaml
│   └── plots/
│
├── hpo/<run_id>/                      ← unique per run
│   ├── experiment_info.json
│   ├── best_config.yaml
│   ├── hpo_results.csv
│   └── study.db
│
├── patch_predictions/<feat_subfolder>/       ← classify output
│   ├── <slide_id>.csv                        ← per-patch predictions
│   └── heatmaps/<slide_id>/
│       ├── <slide_id>_category_map.png       ← category_map mode
│       ├── <slide_id>_confidence_TUM.png     ← confidence mode
│       └── top10_TUM/
│           └── 01_<slide_id>_x256_y512_conf0.987.png
│
├── crossval/<task>/<model>__<feat>__<ts>/   ← unique per run
│   ├── experiment_info.json
│   ├── fold_*/
│   └── cv_summary.json
│
└── multilabel/
    ├── splits/<task>/
    │   ├── train.csv | val.csv | test.csv
    ├── experiments/<task>/<model>_<ts>/
    ├── hpo/<run_id>/
    └── crossval/<task>/<model>__<feat>__<ts>/
```

---

## Overfitting Mitigation Reference

| Technique | Config Key | HPO-Tunable | Notes |
|---|---|---|---|
| Dropout | `mil.dropout` | ✅ | Applied to projection + classifier |
| L2 regularization | `training.weight_decay` | ✅ | Controls weight magnitude |
| Label smoothing | `training.label_smoothing` | ✅ | Reduces overconfidence |
| Patch dropout | `training.patch_dropout` | ✅ | Randomly drops patch fraction per step |
| Patch shuffling | `training.patch_shuffle` | ✅ | Breaks order bias |
| Bag size cap | `training.max_patches` | ✅ | A_patches: 600–1200 |
| LR warmup | `training.warmup_epochs` | ✅ | Linear ramp over first N epochs |
| LR scheduling | `training.lr_scheduler` | ✅ | Plateau / cosine / step |
| Early stopping | `training.early_stopping_patience` | ✅ | Stops after N non-improving epochs |
| Focal loss (ML) | `multilabel_training.loss` | ✅ | Downweights easy examples |

---

## Common Issues

### No `.pt` files found

```
Training dataset has 0 valid bags.
Expected features in: results/features/<dir>/pt_files/
```

**Fix:** Run `python main.py extract --config config/config.yaml` first, or point `--features` to the correct directory.

### HPO shape mismatch during training

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Fix:** Ensure `mil.encoding_size` matches the actual embedding dimension of your `.pt` files. HPO varies `proj_dim` (internal projection), not `encoding_size`.

### Iterative stratification warning

```
skmultilearn not installed — falling back to random split
```

**Fix:** `pip install scikit-multilearn` for correct multi-label stratified splits.

### HPO resumes an old study

Optuna reuses `study.db` if the path already exists. Each run creates a new directory with a unique timestamp, so this should not occur unless you manually re-use an old directory.
