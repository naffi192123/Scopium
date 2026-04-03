# Scopium — WSI Classification Framework

A modular, YAML-configurable framework for Whole Slide Image (WSI) analysis — from raw slide ingestion through tissue segmentation, feature extraction, Multiple Instance Learning (MIL) classification, hyperparameter optimisation (HPO), and cross-validation — for both **single-label** (binary/multiclass) and **multi-label** tasks.

---

## Quick Start

```bash
conda activate dl_py39
cd wsi_framework

# ── Shared preprocessing ─────────────────────────────────────────────────────
python main.py stats    --config config/config.yaml
python main.py process  --config config/config.yaml
python main.py segment  --config config/config.yaml
python main.py extract  --config config/config.yaml

# ── Single-label pipeline ─────────────────────────────────────────────────────
python main.py analyse  --config config/config.yaml
python main.py split    --config config/config.yaml
python main.py train    --config config/config.yaml
python main.py evaluate --config config/config.yaml
python main.py heatmap  --config config/config.yaml

# ── HPO ──────────────────────────────────────────────────────────────────────
python main.py hpo    --config config/config.yaml
python main.py hpo    --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py train  --config config/config.yaml --use_best_config

# ── Cross-validation ─────────────────────────────────────────────────────────
python main.py crossval --config config/config.yaml
python main.py crossval --config config/config.yaml --use_best_config
python main.py crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM

# ── Multi-label pipeline ──────────────────────────────────────────────────────
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
| `classify` | Patch-level classifier inference → per-patch CSV + tissue-filtered `.pt`/`.h5` files |
| `classify-heatmap` | Tile-level prediction heatmaps overlaid on WSI thumbnails |
| `hpo` | Optuna HPO — 22-parameter search space |
| `crossval` | Stratified K-fold cross-validation |

### Multi-Label Pipeline

| Command | Description |
|---|---|
| `multilabel-validate` | Validate label CSV and feature file availability |
| `multilabel-split` | Train/val/test split with iterative stratification |
| `multilabel-train` | Multi-label MIL training (BCE / Focal loss) |
| `multilabel-evaluate` | Per-label metrics, ROC curves, Hamming loss |
| `multilabel-hpo` | Optuna HPO with 25-parameter search space |
| `multilabel-crossval` | K-fold CV with iterative stratification |

### Key CLI Flags

| Flag | Applies to | Description |
|---|---|---|
| `--features <dir>` | all MIL commands | Exact feature directory (model suffix not appended) |
| `--csv <path>` | `analyse`, `split`, `multilabel-split`, `multilabel-validate` | Label CSV path |
| `--experiment <dir>` | `evaluate`, `heatmap`, `multilabel-evaluate` | Use a specific experiment directory |
| `--slide <id>` | `classify-heatmap` | Process a single slide by ID (e.g. `--slide CMU-1`). Overrides `patch_classifier.heatmap.slides` in config |
| `--use_best_config` | `train`, `crossval`, `multilabel-train`, `multilabel-crossval` | Merge best HPO config before running |
| `--split <val\|test>` | `multilabel-evaluate` | Which split to evaluate (default: `test`) |

---

## Feature Directory Selection

Feature extraction produces one directory per (patch-config, model) combination:

```
results/features/
    patch512_step512_level0__optimus__TUM/
    patch512_step512_level0__optimus/
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

## Supported MIL Models

Set via `mil.model` in `config.yaml`:

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

### Single-Label

```yaml
task:
  name: metastasis
  type: binary          # binary | multiclass
  num_classes: 2
  class_names: [benign, malignant]

mil:
  model: abmil          # model architecture
  encoding_size: 1536   # MUST match feature extractor output dim
  proj_dim: 512         # internal projection dim (tunable by HPO)
  hidden_dim: 256       # attention hidden size
  dropout: 0.4          # dropout rate (default 0.4)

training:
  optimizer: AdamW
  learning_rate: 2.0e-3    # lr (default 2e-3)
  weight_decay: 1.0e-3     # reg / L2 regularization (default 1e-3)
  beta1: 0.75              # Adam β₁ (default 0.75)
  beta2: 0.95              # Adam β₂ (default 0.95)
  eps: 1.0e-2              # Adam ε (default 1e-2)
  lr_scheduler: plateau
  lr_scheduler_factor: 0.75   # decay factor (default 0.75)
  lr_scheduler_patience: 20   # plateau patience (default 20)
  early_stopping: true
  early_stopping_patience: 20
  max_patches: 800         # A_patches — max patches per slide (default 800)
  max_epochs: 100
```

### Multi-Label

```yaml
multilabel:
  task_name: multilabel_task
  label_names: [TTN, TP53, MUC16]
  binary_columns: true
  threshold: 0.5

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
  loss: bce              # bce | focal
  weighted_loss: true
  monitor_metric: macro_auc
  max_patches: 800       # A_patches
  max_epochs: 100

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

## Hyperparameter Reference

All hyperparameters are configurable in `config.yaml` and tunable via HPO.

| Parameter | Default | Key in config | Description |
|---|---|---|---|
| `dropout` | 0.4 | `mil.dropout` | Dropout applied to projection and classifier layers |
| `proj_dim` | 512 | `mil.proj_dim` | Internal projection size (all post-projection layers) |
| `hidden_dim` | 256 | `mil.hidden_dim` | Attention network hidden dimension |
| `learning_rate` (lr) | 2e-3 | `training.learning_rate` | Adam/AdamW learning rate |
| `weight_decay` (reg) | 1e-3 | `training.weight_decay` | L2 regularization strength |
| `beta1` | 0.75 | `training.beta1` | Adam first moment decay rate |
| `beta2` | 0.95 | `training.beta2` | Adam second moment decay rate |
| `eps` | 1e-2 | `training.eps` | Adam numerical stability constant |
| `lr_scheduler_factor` | 0.75 | `training.lr_scheduler_factor` | LR decay multiplier (plateau mode) |
| `lr_scheduler_patience` | 20 | `training.lr_scheduler_patience` | Epochs before LR reduction (plateau) |
| `max_patches` (A_patches) | 800 | `training.max_patches` | Max patches per slide per epoch |
| `early_stopping_patience` | 20 | `training.early_stopping_patience` | Epochs without improvement before stopping |

---

## Hyperparameter Optimisation (HPO)

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

### HPO Search Space (both pipelines)

| Parameter | Type | Range / Choices |
|---|---|---|
| `model` | categorical | `[abmil, clam_sb, clam_mb, mean_pool, transmil, dsmil]` |
| `optimizer` | categorical | `[AdamW, Adam]` |
| `learning_rate` | log-uniform | `[1e-4, 2e-3]` |
| `weight_decay` | log-uniform | `[1e-5, 5e-3]` |
| `beta1` | uniform | `[0.5, 0.99]` |
| `beta2` | uniform | `[0.9, 0.999]` |
| `eps` | log-uniform | `[1e-8, 1e-2]` |
| `dropout` | uniform | `[0.1, 0.6]` |
| `dropout_attn` | uniform | `[0.2, 0.5]` |
| `dropout_classifier` | uniform | `[0.1, 0.4]` |
| `attn_hidden_dim` | categorical | `[32, 64, 128, 256]` |
| `feature_proj_dim` (`proj_dim`) | categorical | `[256, 512]` |
| `lr_scheduler` | categorical | `[plateau, cosine, step]` |
| `lr_factor` | uniform | `[0.1, 0.9]` |
| `lr_patience` | categorical | `[5, 10, 20]` |
| `label_smoothing` | uniform | `[0.0, 0.15]` |
| `early_stop_patience` | categorical | `[10, 15, 20]` |
| `warmup_epochs` | categorical | `[0, 2, 5]` |
| `patch_dropout` | uniform | `[0.0, 0.3]` |
| `max_patches` (A_patches) | categorical | `[600, 700, 800, 900, 1000, 1100, 1200]` |

Multi-label HPO additionally tunes: `loss`, `focal_gamma`, `threshold`, `patch_shuffle`.

### HPO Experiment Tracking

Each HPO run creates an **isolated, timestamped directory**:

```
results/hpo/
    mil_hpo__patch512_step512_level0__optimus__TUM__20260402_063229/
        experiment_info.json   ← provenance (task, features, device, timestamp)
        base_config.yaml       ← exact config snapshot
        trial_0000/
            trial_config.yaml
            trial_metrics.json
        best_config.yaml       ← best hyperparameters (merged config)
        best_trial.json
        hpo_results.csv
        study.db               ← Optuna SQLite (resumable)
```

---

## Cross-Validation

### Single-Label CV

```bash
python main.py crossval --config config/config.yaml
python main.py crossval --config config/config.yaml --use_best_config
python main.py crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
```

### Multi-Label CV

```bash
python main.py multilabel-crossval --config config/config.yaml
python main.py multilabel-crossval --config config/config.yaml --use_best_config
python main.py multilabel-crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
```

### CV Experiment Tracking

Every run is saved to a **unique, structured directory** encoding task → model → features → timestamp:

```
results/crossval/<task>/<model>__<feat>__<ts>/
    experiment_info.json
    fold_01/ ... fold_N/
        best_model.pt
        fold_metrics.json
    cv_summary.json
    cv_summary.csv
```

---

## Patch-Level Classifier Pipeline

The `classify` → `classify-heatmap` pipeline provides a **patch-level tissue-classification** workflow on top of an already-extracted feature set.  It requires a separately trained patch classifier (e.g. trained on the [NCT-CRC-HE](https://zenodo.org/record/1214456) dataset).

> **Use-case:** Isolate tumour (TUM) or stromal (STR) patches before MIL training to supply tissue-specific feature directories (e.g. `patch512_step512_level0__optimus__TUM`).

### Step 1 — Run Inference (`classify`)

```bash
python main.py classify --config config/config.yaml
python main.py classify --config config/config.yaml --features patch512_step512_level0__optimus
```

**What it does:**
1. Loads the pretrained patch classifier from `patch_classifier.checkpoint_path`.
2. Iterates over every `.h5` or `.pt` feature file in the active feature directory.
3. Runs batched softmax inference over all patches in each slide.
4. Writes one **prediction CSV** per slide.
5. For each requested tissue category (`filter_categories`), writes filtered `.h5` and `.pt` files containing only the patches belonging to that tissue class.

**Config:**

```yaml
patch_classifier:
  checkpoint_path: "outputs/checkpoints/BEST_MODEL.pth"  # pretrained patch classifier
  batch_size: 512         # patches per GPU batch
  input_format: h5        # 'h5' (preferred, includes coords) or 'pt'
  filter_categories:      # tissue classes to save as separate feature dirs
    - TUM                 # tumour
    - STR                 # stroma
```

**Outputs:**

```
results/
├── patch_predictions/<feat_subfolder>/
│   ├── <slide_id>.csv          ← one row per patch:
│   │   coord_x, coord_y, predicted_label, confidence,
│   │   ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM
│   └── ...
│
└── features/<feat_subfolder>__TUM/     ← filtered feature dirs, one per category
    ├── h5_files/<slide_id>.h5          ← (M, D) features + (M, 2) coords
    └── pt_files/<slide_id>.pt          ← FloatTensor (M, D)
```

Where `M ≤ N` is the number of patches classified as that tissue category.

**Tissue categories** (NCT-CRC palette):

| Code | Tissue |
|---|---|
| `ADI` | Adipose |
| `DEB` | Debris |
| `LYM` | Lymphocytes |
| `MUC` | Mucus |
| `MUS` | Smooth muscle |
| `NOR` | Normal colon |
| `STR` | Cancer-associated stroma |
| `TUM` | Colorectal adenocarcinoma |

### Step 2 — Heatmaps (`classify-heatmap`)

```bash
# All slides, default mode
python main.py classify-heatmap --config config/config.yaml

# Single slide
python main.py classify-heatmap --config config/config.yaml --slide CMU-1

# Specify feature dir
python main.py classify-heatmap --config config/config.yaml \
    --features patch512_step512_level0__optimus
```

**What it does:**
Reads the prediction CSVs generated by `classify` and overlays per-patch classification colours or confidence scores onto the WSI thumbnail.

**Two visualisation modes:**

| Mode | Description |
|---|---|
| `category_map` | Each tile coloured by its predicted tissue class. Colour-coded legend appended below. |
| `confidence` | Each tile coloured by the jet colour scale (blue = low, red = high confidence). Jet colorbar appended. Top-K highest-confidence tiles saved as individual patch crops. |

**Config:**

```yaml
patch_classifier:
  heatmap:
    slides: all           # all | "CMU-1" | [CMU-1, CMU-2]
    categories: all       # all | "TUM"   | [TUM, STR, LYM]
    mode: category_map    # category_map | confidence
    alpha: 0.50           # overlay opacity [0, 1]
    top_k_tiles: 10       # number of top-confidence tiles to crop and save
```

**Outputs:**

```
results/patch_predictions/<feat_subfolder>/heatmaps/
└── <slide_id>/
    ├── <slide_id>_category_map.png           ← category_map mode
    ├── <slide_id>_confidence_TUM.png         ← confidence mode, per category
    ├── <slide_id>_confidence_STR.png
    └── top10_TUM/
        ├── 01_<slide_id>_x<x>_y<y>_conf0.987.png
        └── ...                               ← top-K highest confidence patch crops
```

**Category colour palette:**

| Category | Colour |
|---|---|
| `ADI` | Tan |
| `DEB` | Grey |
| `LYM` | Purple |
| `MUC` | Teal |
| `MUS` | Orange |
| `NOR` | Green |
| `STR` | Blue |
| `TUM` | Red |

### End-to-End Classify Workflow

```bash
# 1. Extract features (all tissue)
python main.py extract --config config/config.yaml

# 2. Classify every patch → CSVs + filtered feature dirs
python main.py classify --config config/config.yaml \
    --features patch512_step512_level0__optimus

# 3. Visualise patch predictions
python main.py classify-heatmap --config config/config.yaml \
    --features patch512_step512_level0__optimus

# 4. Use tumour-only features for MIL training
python main.py train --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

python main.py hpo --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM
```

---

## Multi-Label Pipeline — End-to-End

```bash
# 1. Validate CSV coverage
python main.py multilabel-validate --config config/config.yaml \
    --csv dataset/annotations/labels.csv \
    --features patch512_step512_level0__optimus__TUM

# 2. Stratified splits (70/15/15)
python main.py multilabel-split --config config/config.yaml \
    --csv dataset/annotations/labels.csv

# 3. Train
python main.py multilabel-train --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# 4. Evaluate
python main.py multilabel-evaluate --config config/config.yaml

# 5. HPO (30 trials)
python main.py multilabel-hpo --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# 6. Retrain with best config
python main.py multilabel-train --config config/config.yaml --use_best_config

# 7. Cross-validate
python main.py multilabel-crossval --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM
```

---

## Overfitting Mitigation

| Technique | Single-Label Key | Multi-Label Key | HPO-Tunable |
|---|---|---|---|
| Dropout | `mil.dropout` | `mil.dropout` | ✅ |
| L2 regularization | `training.weight_decay` | `multilabel_training.weight_decay` | ✅ |
| Label smoothing | `training.label_smoothing` | `multilabel_training.label_smoothing` | ✅ |
| Patch dropout | `training.patch_dropout` | `multilabel_training.patch_dropout` | ✅ |
| Patch shuffling | `training.patch_shuffle` | `multilabel_training.patch_shuffle` | ✅ |
| Bag size cap (A_patches) | `training.max_patches` | `multilabel_training.max_patches` | ✅ |
| LR warmup | `training.warmup_epochs` | `multilabel_training.warmup_epochs` | ✅ |
| LR scheduling | `training.lr_scheduler` | `multilabel_training.lr_scheduler` | ✅ |
| Early stopping | `training.early_stopping` | `multilabel_training.early_stopping` | — |
| Focal loss | — | `multilabel_training.loss: focal` | ✅ |
| Per-label pos_weight | — | `multilabel_training.weighted_loss` | — |

---

## Parameter Validation

HPO and cross-validation validate all required config keys **before** starting:

```
HPO: config missing required key 'task.class_names'
HPO aborted due to missing configuration keys.
```

**Required (single-label):** `mil.model`, `task.name`, `task.num_classes`, `task.class_names`, `training.max_epochs`, `paths.results_dir`

**Required (multi-label):** `mil.model`, `mil.encoding_size`, `multilabel.label_names`, `multilabel_training.learning_rate`, `paths.results_dir`

---

## Documentation

| File | Contents |
|---|---|
| `docs/setup_and_usage.md` | Full installation, all commands, HP reference, HPO, CV |
| `docs/architecture.md` | Pipeline architecture and module relationships |
| `config/config.yaml` | Fully-annotated configuration (all pipeline parameters) |
