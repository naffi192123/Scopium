# WSI Framework — Setup & Usage Guide

---

## Environment Setup

### Linux / HPC Cluster (recommended)

```bash
module load anaconda3/2023.03 cuda/12.2   # load modules if on HPC
conda create -y -n dl_py39 python=3.9
conda activate dl_py39
cd path/to/wsi_framework
pip install -r requirements.txt
pip install scikit-learn optuna tqdm scikit-multilearn
```

### Windows (Local Development)

```bash
conda create -y -n dl_py39 python=3.9
conda activate dl_py39
cd path\to\wsi_framework
pip install -r requirements.txt
pip install scikit-learn optuna tqdm scikit-multilearn
```

> **Windows note:** Install OpenSlide binaries from [openslide.org](https://openslide.org/download/) and add `bin\` to your `PATH`.

---

## Dataset Structure

```text
wsi_framework/
└── dataset/
    ├── slides/           ← .svs / .tif / .ndpi files go here
    └── annotations/      ← (optional) CSV label files
```

---

## Configuration (`config/config.yaml`)

The entire pipeline is driven by a single YAML file.

### Paths
```yaml
paths:
  slides_dir: "dataset/slides"
  results_dir: "results"
```

### Segmentation
```yaml
segmentation:
  seg_level: -1       # -1 = auto (~64x downsample)
  sthresh: 8          # HSV saturation threshold
  mthresh: 7          # Median blur kernel size
  use_otsu: false     # Use Otsu's method instead of fixed threshold
```

### Tiling
```yaml
tiling:
  patch_level: 0      # Pyramid level (0 = full resolution)
  patch_size: 512     # Tile width/height in pixels
  step_size: 512      # Step between tiles (= patch_size for no overlap)
  contour_fn: four_pt # Tissue containment check strategy
  filter_blank: true  # Skip background patches
  mode: sequential    # sequential | parallel
  num_workers: 8      # Workers for parallel mode

  # Optional: override the auto-derived patch subfolder name.
  patches_subfolder_override: null   # e.g. "patch256_step256_level0_otsu"
```

### Feature Extraction
```yaml
feature_extraction:
  model: rn50         # See supported models table below
  batch_size: 64      # Reduce if CUDA out of memory
  transforms: auto    # 'auto' | 'none' | 'reinhard' | 'macenko' | 'uni_default' | …
  weights_path: null  # Path to local weights (required for some models)

  features_subfolder_override: null  # base name, model auto-appended
  features_dir_override: null        # exact name, model NOT appended
```

---

## Default Output Directory Naming

Every pipeline stage saves to a subfolder whose name encodes the active parameters — **different tiling or extraction runs never overwrite each other**.

| Stage | Default subfolder path | Key parameters encoded |
|---|---|---|
| `segment` — patches | `results/patches/patch{size}_step{step}_level{lvl}/` | `patch_size`, `step_size`, `patch_level` |
| `segment` — masks | `results/masks/patch{size}_step{step}_level{lvl}/` | same as patches |
| `extract` — features | `results/features/patch{size}_step{step}_level{lvl}__{model}/` | patch config + model |
| `crossval` | `results/crossval/{task}/{model}__{feat}__{YYYYMMDD_HHMMSS}/` | task, model, features, timestamp |
| `multilabel-crossval` | `results/multilabel/crossval/{task}/{model}__{feat}__{YYYYMMDD_HHMMSS}/` | task, model, features, timestamp |
| `hpo` | `results/hpo/{study}__{feat}__{YYYYMMDD_HHMMSS}/` | study name, features, timestamp |
| `multilabel-hpo` | `results/multilabel/hpo/{study}__{feat}__{YYYYMMDD_HHMMSS}/` | study name, features, timestamp |

---

## Selecting a Specific Feature Subfolder

Four resolution tiers, applied in priority order:

| Priority | Mechanism | Model auto-appended? |
|---|---|---|
| 1 (highest) | `--features <dir>` CLI | **No — exact name verbatim** |
| 2 | `feature_extraction.features_dir_override` (YAML) | **No** |
| 3 | `feature_extraction.features_subfolder_override` (YAML) | **Yes** |
| 4 (lowest) | Auto-derived from config params | **Yes** |

```bash
# All downstream commands accept --features:
python main.py train            --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py evaluate         --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py hpo              --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py crossval         --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-train --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-hpo   --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
```

---

## Command Reference

### 1. Dataset Scanning (`stats`)

```bash
python main.py stats --config config/config.yaml
```
**Output:** `results/dataset_stats.csv`

---

### 2. Thumbnail & Metadata Extraction (`process`)

```bash
python main.py process --config config/config.yaml
```
**Output:** `results/thumbnails/<slide>.png`, `results/metadata/<slide>.json`

---

### 3. Tissue Segmentation & Patching (`segment`)

```bash
python main.py segment --config config/config.yaml
# Custom subfolder:
python main.py segment --config config/config.yaml --patches my_512px_run
```
**Output:** `results/masks/.../<slide>_mask.png`, `results/patches/.../<slide>.h5`

---

### 4. Segmentation Debugging

```bash
python main.py debug-segmentation --config config/config.yaml   # tile overlay
python main.py extract-tile       --config config/config.yaml   # single high-res tile
```

---

### 5. Feature Extraction (`extract`)

```bash
python main.py extract --config config/config.yaml
python main.py extract --config config/config.yaml \
    --patches my_512px_run \
    --features my_512px_run
```
**Output:** `results/features/{patch_cfg}__{model}/pt_files/*.pt` + `h5_files/*.h5`

---

### 6. Annotation Analysis (`analyse`)

```bash
python main.py analyse --config config/config.yaml
python main.py analyse --config config/config.yaml --csv path/to/labels.csv
```

---

### 7. Single-Label Dataset Splitting (`split`)

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
**Output:** `results/splits/<task_name>/{train,val,test}.csv` + `split_summary.txt`

---

### 8. Single-Label MIL Training (`train`)

```bash
python main.py train --config config/config.yaml
python main.py train --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py train --config config/config.yaml --use_best_config   # after hpo
```
**Output:** `results/experiments/<task>/<model>_<timestamp>/`

---

### 9. Evaluation (`evaluate`)

```bash
python main.py evaluate --config config/config.yaml
python main.py evaluate --config config/config.yaml \
    --experiment results/experiments/metastasis/abmil_20260401_103245
```
**Output:** `<experiment_dir>/evaluate/` — predictions.csv, roc_curve.png, confusion_matrix.png, metrics.json

---

### 10. Attention Heatmaps (`heatmap`)

Only for: `abmil`, `clam_sb`, `clam_mb`, `dsmil`

```bash
python main.py heatmap --config config/config.yaml
python main.py heatmap --config config/config.yaml \
    --experiment results/experiments/metastasis/abmil_...
```
**Output:** `<experiment_dir>/heatmaps/<slide_id>/` — heatmap PNG, attention CSV, top20_tiles/

---

### 11. Patch Classification & Heatmaps (`classify`, `classify-heatmap`)

```bash
python main.py classify        --config config/config.yaml
python main.py classify-heatmap --config config/config.yaml
python main.py classify-heatmap --config config/config.yaml --slide CMU-1
```

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

## Supported MIL Models

| Key | Architecture | Attention |
|---|---|---|
| `mean_pool` | Global Average Pool | No |
| `max_pool` | Global Max Pool | No |
| `abmil` | Gated Attention MIL | Yes |
| `clam_sb` | CLAM Single-Branch | Yes |
| `clam_mb` | CLAM Multi-Branch | Yes |
| `transmil` | Transformer MIL | No |
| `dsmil` | Dual-Stream MIL | Yes |

---

## Hyperparameter Optimisation (HPO)

### Single-Label HPO

```bash
python main.py hpo --config config/config.yaml
python main.py hpo --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py train --config config/config.yaml --use_best_config
```

### Multi-Label HPO

```bash
python main.py multilabel-hpo --config config/config.yaml
python main.py multilabel-hpo --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py multilabel-train --config config/config.yaml --use_best_config
```

### HPO Experiment Tracking

Each HPO run creates an **isolated, timestamped directory**:

```
results/hpo/
    mil_hpo__patch512_step512_level0__optimus__TUM__20260401_103245/
        experiment_info.json   ← provenance (task, features, device, timestamp)
        base_config.yaml       ← exact config used
        trial_0000/
            trial_config.yaml
            trial_metrics.json
        best_config.yaml       ← best hyperparameters (merged config)
        best_trial.json
        hpo_results.csv
        study.db               ← Optuna SQLite (resumable)
```

To pin a specific HPO run for `--use_best_config`:
```yaml
hpo:
  best_run_path: results/hpo/mil_hpo__patch512_step512_level0__optimus__TUM__20260401_103245
```

```yaml
multilabel_hpo:
  best_run_path: results/multilabel/hpo/ml_hpo__patch512_step512_level0__optimus__TUM__20260401_103245
```

---

## K-Fold Cross-Validation

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

Every CV invocation generates a **unique, structured run directory** that encodes four dimensions — so re-running with different settings never overwrites previous results:

```
results/crossval/<task_name>/<model>__<feat_dir>__<YYYYMMDD_HHMMSS>/
```

**Example:**
```
results/crossval/
    metastasis/
        abmil__patch512_step512_level0__optimus__TUM__20260401_103245/
            experiment_info.json    ← task, model, features, n_folds, timestamp
            config_snapshot.yaml    ← exact config.yaml used for this run
            combined_pool.csv       ← merged train+val pool
            fold_01/
                best_model.pt
                fold_metrics.json
            fold_02/ ... fold_05/
            cv_summary.json         ← mean ± std across all folds
            cv_summary.csv
        abmil__patch256_step256_level0__uni__20260402_090010/   ← second run
            ...

results/multilabel/crossval/
    multilabel_task/
        abmil__patch512_step512_level0__optimus__TUM__20260402_143022/
            experiment_info.json    ← includes label_names, stratification method
            config_snapshot.yaml
            fold_01/ ... fold_05/
            cv_summary.json
            cv_summary.csv
```

### CV Configuration

```yaml
crossval:
  n_folds: 5
  seed: 42

multilabel_crossval:
  n_folds: 5
  seed: 42
```

---

## Multi-Label Pipeline

### Step-by-Step

```bash
# 0. Install extra dependencies
pip install scikit-multilearn optuna tqdm

# 1. Set label_names in config.yaml:
#    multilabel:
#      label_names: [TTN, TP53, MUC16]
#      binary_columns: true
#      threshold: 0.5

# 2. Validate CSV and feature file coverage
python main.py multilabel-validate --config config/config.yaml \
    --csv dataset/annotations/labels.csv \
    --features patch512_step512_level0__optimus__TUM

# 3. Create splits (70/15/15 by default)
python main.py multilabel-split --config config/config.yaml \
    --csv dataset/annotations/labels.csv

# 4. Train
python main.py multilabel-train --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# 5. Evaluate on test split
python main.py multilabel-evaluate --config config/config.yaml

# 6. Evaluate on val split
python main.py multilabel-evaluate --config config/config.yaml --split val

# 7. Run HPO (30 trials × 30 epochs by default)
python main.py multilabel-hpo --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM

# 8. Train with best HPO config
python main.py multilabel-train --config config/config.yaml --use_best_config

# 9. Cross-validate
python main.py multilabel-crossval --config config/config.yaml \
    --features patch512_step512_level0__optimus__TUM
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
Config for Format B:
```yaml
multilabel:
  binary_columns: false
  labels_string_col: labels
```

### Multi-Label Configuration

```yaml
multilabel:
  task_name: multilabel_task
  label_names: [TTN, TP53, MUC16]   # ← required, in order
  binary_columns: true
  threshold: 0.5                     # sigmoid decision threshold

multilabel_training:
  max_epochs: 100
  learning_rate: 2.0e-4
  weight_decay: 1.0e-4
  optimizer: AdamW
  lr_scheduler: plateau              # plateau | cosine | step
  early_stopping: true
  early_stopping_patience: 20
  loss: bce                          # bce | focal
  weighted_loss: true                # per-label pos_weight (recommended)
  label_smoothing: 0.05
  focal_gamma: 2.0
  monitor_metric: macro_auc
  patch_dropout: 0.0
  max_patches: null

multilabel_split:
  type: train_val_test
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
  random_seed: 42

multilabel_hpo:
  study_name: ml_hpo
  n_trials: 30
  epochs_per_trial: 30
  metric: val_macro_auc
  direction: maximize

multilabel_crossval:
  n_folds: 5
  seed: 42
```

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
│   └── evaluate/
│       ├── metrics.json              ← macro/micro AUC+F1, Hamming, SubsetAcc
│       ├── per_label_metrics.csv
│       └── roc_curves/<label>_roc.png
├── hpo/ml_hpo__<feat>__<YYYYMMDD_HHMMSS>/
│   ├── experiment_info.json
│   ├── best_config.yaml
│   └── hpo_results.csv
└── crossval/<task_name>/<model>__<feat>__<YYYYMMDD_HHMMSS>/
    ├── experiment_info.json
    ├── config_snapshot.yaml
    ├── fold_01/ ... fold_N/
    ├── cv_summary.json
    └── cv_summary.csv
```

---

## GPU Memory Guide

| GPU VRAM | Recommended `batch_size` |
|---|---|
| 8 GB | 64 |
| 16 GB | 128 |
| 40 GB | 256+ |

---

## Logging

All commands write a structured log to `results/wsi_framework.log` (rotating, 10 MB max). Log level and format are configured in `utils/logger.py`.

---

## Parameter Validation

HPO and cross-validation check all required config keys **before** starting:

```
HPO: config missing required key 'task.class_names'
HPO aborted due to missing configuration keys.
```

**Required (single-label):** `mil.model`, `task.name`, `task.num_classes`, `task.class_names`, `training.max_epochs`, `paths.results_dir`

**Required (multi-label):** `mil.model`, `mil.encoding_size`, `multilabel.label_names`, `multilabel_training.learning_rate`, `multilabel_training.weight_decay`, `multilabel_training.max_epochs`, `paths.results_dir`
