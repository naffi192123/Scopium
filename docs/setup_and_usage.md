# WSI Framework — Setup & Usage Guide

---

## Environment Setup

### Windows (Local Development)

```bash
conda create -y -n dl_py39 python=3.9
conda activate dl_py39
cd path\to\wsi_framework
pip install -r requirements.txt
pip install scikit-learn
```

> **Windows note:** Install OpenSlide binaries from [openslide.org](https://openslide.org/download/) and add `bin\` to your `PATH`.

### HPC Cluster (CentOS / SLURM)

```bash
module load anaconda3/2023.03 cuda/12.2
conda create -y -n dl_py39 python=3.9
conda activate dl_py39
pip install -r requirements.txt
pip install scikit-learn
```

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
  # Default auto-name: patches/patch{size}_step{step}_level{lvl}/
  # CLI flag --patches takes priority over this key.
  patches_subfolder_override: null   # e.g. "patch256_step256_level0_otsu"
```

### Feature Extraction
```yaml
feature_extraction:
  model: rn50         # See supported models table below
  batch_size: 64      # Reduce if CUDA out of memory
  transforms: auto    # 'auto' | 'none' | 'reinhard' | 'macenko' | 'uni_default' | …
  weights_path: null  # Path to local weights (required for some models)

  # Optional: override the base name of the feature subfolder.
  # The model key is ALWAYS auto-appended.
  # Default auto-name: features/patch{size}_step{step}_level{lvl}__{model}/
  # Example: features_subfolder_override: "patch512_step512_level0"
  #   → resolves to: features/patch512_step512_level0__rn50/
  # CLI flag --features takes priority over this key.
  features_subfolder_override: null
```

---

## Default Output Directory Naming

By default every pipeline stage saves its outputs to a subfolder whose name encodes the active configuration parameters. This ensures **different tiling or extraction runs never overwrite each other**.

| Stage | Default subfolder path | Key parameters encoded |
|---|---|---|
| `segment` — patches | `results/patches/patch{size}_step{step}_level{lvl}/` | `patch_size`, `step_size`, `patch_level` |
| `segment` — masks | `results/masks/patch{size}_step{step}_level{lvl}/` | same as patches |
| `extract` — features | `results/features/patch{size}_step{step}_level{lvl}__{model}/` | patch config + `feature_extraction.model` |
| `classify` — filtered | `results/features/{feature_subfolder}__{CATEGORY}/` | tissue classes kept |
| `classify` — CSVs | `results/patch_predictions/{feature_subfolder}/` | slide predictions |
| `classify-heatmap` | `results/patch_predictions/{feature_subfolder}/heatmaps/` | prediction heatmaps |

**Example** with the default config (`patch_size=512`, `step_size=512`, `patch_level=0`, `model=rn50`):

```
results/patches/patch512_step512_level0/         ← .h5 patch coordinate files
results/masks/patch512_step512_level0/           ← tissue mask PNGs
results/features/patch512_step512_level0__rn50/  ← extracted feature tensors
```

---

## Selecting a Specific Patch or Feature Subfolder

Two independent override mechanisms let you direct any command at a specific pre-run configuration.

### Patch subfolder (affects `segment` + `extract`)

**YAML (persists across runs):**
```yaml
tiling:
  patches_subfolder_override: "patch256_step256_level0_otsu"
```

**CLI flag (one-shot, highest priority):**
```bash
python main.py segment --config config/config.yaml --patches patch256_step256_level0_otsu
python main.py extract --config config/config.yaml --patches patch256_step256_level0_otsu
```

### Feature subfolder (affects `extract`, `train`, `evaluate`, `heatmap`, `classify`, `classify-heatmap`)

Five resolution tiers, applied in priority order:

| Priority | Mechanism | Model auto-appended? | Example resolves to |
|---|---|---|---|
| 1 (highest) | `--feature_dir` CLI | **No** | `features/patch512_step512_level0__uni/` |
| 2 | `--features` CLI | **Yes** | `features/patch512_step512_level0__rn50/` |
| 3 | `feature_extraction.features_subfolder_override` | **Yes** | same pattern |
| 4 | `feature_extraction.features_dir_override` | **No** | exact name as given |
| 5 (lowest) | Auto-derived | **Yes** | `features/patch{sz}_step{sz}_level{lvl}__{model}/` |

**When to use which:**
- Use **`--feature_dir`** (or `features_dir_override`) when pointing to a specific existing directory whose name already includes the model suffix.
- Use **`--features`** (or `features_subfolder_override`) when you only want to override the base name; the current model key is still auto-appended.

**YAML:**
```yaml
feature_extraction:
  # BASE name — model always appended:
  features_subfolder_override: "patch512_step512_level0"
  # → resolves to: features/patch512_step512_level0__rn50/

  # EXACT name — model NOT appended:
  features_dir_override: "patch512_step512_level0__uni"
  # → resolves to: features/patch512_step512_level0__uni/
```

**CLI:**
```bash
# Base name (model auto-appended)
python main.py train --config config/config.yaml --features patch512_step512_level0
# reads from: results/features/patch512_step512_level0__rn50/pt_files/

# Exact name (model NOT appended — highest priority)
python main.py train --config config/config.yaml --feature_dir patch512_step512_level0__uni
# reads from: results/features/patch512_step512_level0__uni/pt_files/
```

### Override priority (highest → lowest)

1. CLI flag `--feature_dir` (exact, no model appended)
2. CLI flag `--features` (base name, model auto-appended)
3. YAML `features_subfolder_override` (base, model auto-appended)
4. YAML `features_dir_override` (exact, no model appended)
5. Auto-derived from config parameters

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

**Output:**
- `results/thumbnails/<slide>.png`
- `results/metadata/<slide>.json`

---

### 3. Tissue Segmentation & Patching (`segment`)

Detects tissue regions using HSV colour-space segmentation and saves patch coordinates.
Supports `sequential` (default) and `parallel` execution via `tiling.mode`.

```bash
python main.py segment --config config/config.yaml

# Write to a custom-named subfolder:
python main.py segment --config config/config.yaml --patches my_512px_run
```

**Output:**
- `results/masks/patch512_step512_level0/<slide>_mask.png` — tissue mask
- `results/patches/patch512_step512_level0/<slide>.h5` — patch coordinates

Each `.h5` file contains:
```
coords       (N, 2)  int64   — (x, y) top-left corner of each valid patch
attrs:
  patch_level         int     — pyramid level used
  patch_size          int     — tile width/height in pixels
```

---

### 4. Segmentation Debugging

#### Visualise Tiles on Thumbnail (`debug-segmentation`)

```bash
python main.py debug-segmentation --config config/config.yaml
```

**Output:** `results/debug/segmented_tiles_thumbnail_<slide>.png`

#### Extract a Single Tile (`extract-tile`)

```bash
python main.py extract-tile --config config/config.yaml
```

**Output:** `results/debug/tile_<slide>.png`

---

### 5. Feature Extraction (`extract`)

Extracts deep learning embeddings from patch coordinates using a GPU-accelerated backbone.

```bash
# Default: reads from patches/patch512_step512_level0/, writes to features/patch512_step512_level0__rn50/
python main.py extract --config config/config.yaml

# Read from a specific patch subfolder, write features to a named base (model auto-appended):
python main.py extract --config config/config.yaml \
    --patches my_512px_run \
    --features my_512px_run
# writes to: results/features/my_512px_run__rn50/
```

**Output:**
```
results/features/patch512_step512_level0__rn50/
    ├── h5_files/<slide>.h5    ← HDF5: 'features' (N, D) + 'coords' (N, 2)
    └── pt_files/<slide>.pt    ← PyTorch FloatTensor (N, D)
```

---

## Supported Feature Extractors

Set via `feature_extraction.model` in `config.yaml`:

| Model Key | Architecture | Embedding Dim | Source |
|---|---|---|---|
| `rn18` | ResNet-18 | 512 | torchvision (ImageNet) |
| `rn50` | ResNet-50 | 2048 | torchvision (ImageNet) |
| `vit_l` | ViT-Large/16 | 1024 | timm (ImageNet) |
| `uni` | UNI ViT-L | 1024 | MahmoodLab / HF Hub |
| `provgigapath` | Prov-GigaPath | 1536 | Microsoft / HF Hub |
| `phikon` | Phikon ViT-B | 768 | Owkin / HF Hub |
| `hibou_b` | Hibou-B | 768 | HistAI / HF Hub |
| `hibou_l` | Hibou-L | 1024 | HistAI / HF Hub |
| `optimus` | H-Optimus-0 | 1536 | BioOptimus / HF Hub |
| `virchow` | Virchow | 2560 | Paige / HF Hub |
| `virchow2cls` | Virchow-2 (CLS) | 1280 | Paige / HF Hub |

---

## Supported Transforms

Set via `feature_extraction.transforms` in `config.yaml`:

| Key | Description |
|---|---|
| `auto` | Selects the canonical pipeline for the chosen model **(recommended)** |
| `none` | `ToTensor` + ImageNet normalisation |
| `reinhard` | Reinhard H&E stain normalisation → ImageNet range |
| `macenko` | Macenko stain normalisation (requires `torchstain`) |
| `uni_default` | Resize 224 + ImageNet normalisation |
| `gigapath_default` | Resize 256 → CenterCrop 224 + ImageNet normalisation |
| `hibou_default` | Resize/CenterCrop 224 + Hibou-specific statistics |
| `kaiko_default` | Resize/CenterCrop 224 + 0.5/0.5/0.5 normalisation |
| `optimus_default` | CenterCrop 224 + Optimus statistics |

---

## GPU Memory Guide

| GPU VRAM | Recommended `batch_size` |
|---|---|
| 8 GB | 64 |
| 16 GB | 128 |
| 40 GB | 256+ |

---

### 6. Annotation CSV Analysis (`analyse`)

```bash
python main.py analyse --config config/config.yaml
# Override CSV:
python main.py analyse --config config/config.yaml --csv path/to/labels.csv
```

Reports: CSV format, class distribution, duplicates, missing labels, slide file coverage, feature file coverage.

---

### 7. Dataset Splitting (`split`)

```bash
python main.py split --config config/config.yaml
```

Config options (`split` block):

```yaml
split:
  type: train_val_test   # train_test | train_val_test
  train_size: 0.7
  val_size: 0.1          # only for train_val_test
  test_size: 0.2
  stratified: true
  random_seed: 42
```

**Output:** `results/splits/<task_name>/{train,val,test}.csv` + `split_summary.txt`

---

### 8. MIL Training (`train`)

```bash
python main.py train --config config/config.yaml

# Train using a specific feature set (model auto-appended):
python main.py train --config config/config.yaml --features patch512_step512_level0
```

The **default** feature directory is derived automatically from the active `tiling` and `feature_extraction` config keys:
```
results/features/patch{size}_step{step}_level{lvl}__{model}/pt_files/
```

Config options:

```yaml
task:
  name: metastasis
  type: binary          # binary | multiclass
  num_classes: 2
  class_names: [benign, malignant]

mil:
  model: abmil          # abmil | clam_sb | clam_mb | mean_pool | max_pool | transmil | dsmil
  encoding_size: 2048   # MUST match feature extractor output dim
  hidden_dim: 256
  dropout: 0.25
  k_sample: 8           # CLAM only
  bag_weight: 0.7       # CLAM only

training:
  max_epochs: 100
  learning_rate: 0.0002
  early_stopping: true
  early_stopping_patience: 20
  early_stopping_min_epochs: 10
  weighted_loss: false
```

**Outputs:** `results/experiments/<task>/<model>_<timestamp>/`
- `best_model.pt` — weights, optimizer, config, class map, metrics
- `final_model.pt`
- `train_history.csv` — loss, acc, auc, f1, lr, time per epoch
- `config_snapshot.yaml`
- `plots/` — `train_loss_curve.png`, `val_auc_curve.png`, `val_acc_curve.png`, `learning_rate.png`

---

### 9. Evaluation (`evaluate`)

```bash
# Auto-detect latest experiment:
python main.py evaluate --config config/config.yaml

# Specify experiment directory:
python main.py evaluate --config config/config.yaml \
    --experiment results/experiments/metastasis/abmil_20260318_010000
```

**Outputs:** `<experiment_dir>/evaluate/`
- `predictions.csv` — slide_id, true_label, pred_label, prob_class0, prob_class1, …
- `roc_data.csv` — fpr, tpr, thresholds
- `confusion_matrix.csv`
- `classification_report.txt`
- `metrics.json` — accuracy, balanced_accuracy, f1, precision, recall, roc_auc
- `roc_curve.png`, `confusion_matrix.png`

---

### 10. Attention Heatmaps (`heatmap`)

Only supported for attention-based models: `abmil`, `clam_sb`, `clam_mb`, `dsmil`.

```bash
python main.py heatmap --config config/config.yaml
# Or specify experiment:
python main.py heatmap --config config/config.yaml \
    --experiment results/experiments/metastasis/abmil_...
```

**Outputs:** `<experiment_dir>/heatmaps/<slide_id>/`
- `<slide_id>_heatmap.png` — jet colormap overlay on WSI thumbnail
- `<slide_id>_attention_scores.csv` — coord_x, coord_y, attention per patch
- `top20_tiles/` — top 20 highest-attention tiles extracted from the raw WSI

---

### 11. Patch-Level Classification (`classify`)

Runs batched inference on `.h5` or `.pt` features using a pretrained patch classifier.
Saves CSV predictions and creates filtered feature directories per category.

```bash
python main.py classify --config config/config.yaml

# Read features from a specific subfolder:
python main.py classify --config config/config.yaml --features my_features
```

Config (`patch_classifier` block):
```yaml
patch_classifier:
  checkpoint_path: "outputs/classifier/best.pth"
  batch_size: 512
  input_format: h5        # h5 or pt
  filter_categories:      # categories to save as newly filtered .h5/.pt files
    - TUM
    - STR
```

**Outputs:**
- `results/patch_predictions/{features_subfolder}/<slide>.csv`
- `results/features/{features_subfolder}__{CATEGORY}/` (filtered subset files)

---

### 12. Prediction Heatmaps (`classify-heatmap`)

Overlays categorical patch predictions or confidence scores onto the WSI thumbnail.

```bash
python main.py classify-heatmap --config config/config.yaml
# Only for a single WSI:
python main.py classify-heatmap --config config/config.yaml --slide CMU-1
```

Config (`patch_classifier.heatmap` block):
```yaml
patch_classifier:
  heatmap:
    slides: all               # or single slide ID, or list of IDs
    categories: [TUM, STR]    # only visualise these categories
    mode: category_map        # 'category_map' (all classes at once) or 'confidence' (jet map)
    top_k_tiles: 10           # exports top 10 highest confidence patches (confidence mode only)
```

**Outputs:**
- `results/patch_predictions/{subfolder}/heatmaps/<slide>/<slide>_category_map.png`
- `results/patch_predictions/{subfolder}/heatmaps/<slide>/<slide>_confidence_<cat>.png`

---

## Supported MIL Models

| Key | Architecture | Attention |
|---|---|---|
| `mean_pool` | Global Average Pool | No |
| `max_pool` | Global Max Pool | No |
| `abmil` | Gated Attention MIL (Ilse 2018) | Yes |
| `clam_sb` | CLAM Single-Branch (Lu 2021) | Yes |
| `clam_mb` | CLAM Multi-Branch (Lu 2021) | Yes |
| `transmil` | Transformer MIL (Shao 2021) | No |
| `dsmil` | Dual-Stream MIL (Li 2021) | Yes |

> Heatmaps are only generated for attention-based models.

---

## Full Multi-config Workflow Example

```bash
# 1. Tile at 512 px (default auto-name)
python main.py segment --config config/config.yaml
# → results/patches/patch512_step512_level0/
# → results/masks/patch512_step512_level0/

# 2. Tile at 256 px into a named subfolder
python main.py segment --config config/config.yaml --patches patch256_step256_level0
# → results/patches/patch256_step256_level0/
# → results/masks/patch256_step256_level0/

# 3. Extract features from the 512 px patches with ResNet-50 (default)
python main.py extract --config config/config.yaml
# → results/features/patch512_step512_level0__rn50/

# 4. Extract features from the 256 px patches with UNI
#    (change model in config first, then override the patch subfolder)
python main.py extract --config config/config.yaml \
    --patches patch256_step256_level0 \
    --features patch256_step256_level0
# → results/features/patch256_step256_level0__uni/

# 5. Train MIL using default (512 px + rn50) features
python main.py train --config config/config.yaml

# 6. Train MIL using 256 px + UNI features
python main.py train --config config/config.yaml --features patch256_step256_level0
# reads from: results/features/patch256_step256_level0__uni/pt_files/
```

---

## Logging

All commands write a structured log to `results/wsi_framework.log` (rotating, 10 MB max).
