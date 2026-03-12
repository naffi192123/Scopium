# WSI Framework — Setup & Usage Guide

---

## Environment Setup

### Windows (Local Development)

```bash
conda create -y -n dl_py39 python=3.9
conda activate dl_py39
cd path\to\wsi_framework
pip install -r requirements.txt
pip install scikit-learn   # for dataset splitting + metrics
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

Place your raw WSI files and optional annotations as follows:

```text
wsi_framework/
└── dataset/
    ├── slides/           ← .svs / .tif / .ndpi files go here
    └── annotations/      ← (optional) manual annotation files
```

---

## Configuration (`config/config.yaml`)

The entire pipeline is driven by a single YAML file. Key sections:

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
```

### Feature Extraction
```yaml
feature_extraction:
  model: rn50         # See supported models table below
  batch_size: 64      # Reduce if CUDA out of memory
  transforms: auto    # 'auto' | 'none' | 'reinhard' | 'macenko' | 'uni_default' | …
  weights_path: null  # Path to local weights (required for some models)
```

---

## Command Reference

### 1. Dataset Scanning (`stats`)

Scan `slides_dir` and generate a metadata CSV.

```bash
python main.py stats --config config/config.yaml
```

**Output:** `results/dataset_stats.csv`

---

### 2. Thumbnail & Metadata Extraction (`process`)

Generate low-resolution thumbnail PNGs and full-resolution metadata JSONs.

```bash
python main.py process --config config/config.yaml
```

**Output:**
- `results/thumbnails/<slide>.png`
- `results/metadata/<slide>.json`

---

### 3. Tissue Segmentation & Patching (`segment`)

Detect tissue regions using HSV colour-space segmentation and extract patch coordinates.
Supports both `sequential` (default) and `parallel` execution via `tiling.mode`.

```bash
python main.py segment --config config/config.yaml
```

**Output:**
- `results/masks/<slide>_mask.png`
- `results/patches/patch512_step512_level0/<slide>.h5`

Each `.h5` file contains:
```
coords       (N, 2)  int64   — (x, y) top-left corner of each valid patch
attrs:
  patch_level         int     — pyramid level
  patch_size          int     — tile width/height in pixels
```

---

### 4. Segmentation Debugging

#### Visualise Tiles on Thumbnail (`debug-segmentation`)

Overlay every extracted patch coordinate onto the WSI thumbnail as a green bounding box. Ideal for verifying that tissue segmentation is capturing the right regions.

```bash
python main.py debug-segmentation --config config/config.yaml
```

**Output:** `results/debug/segmented_tiles_thumbnail_<slide>.png`

#### Extract a Single Tile (`extract-tile`)

Pick one coordinate from the `.h5` file and extract the corresponding high-resolution tile for manual visual inspection.

```bash
python main.py extract-tile --config config/config.yaml
```

**Output:** `results/debug/tile_<slide>.png`

---

### 5. Feature Extraction (`extract`)

Extracts deep learning embeddings from the patch coordinates using a GPU-accelerated backbone.

```bash
python main.py extract --config config/config.yaml
```

**How it works:**
1. Loads the configured model (lazy imports — only the required backbone loads).
2. Builds the preprocessing transform pipeline from the `transforms` key.
3. For each slide: opens the WSI in the main thread → streams batches of tiles via PyTorch `DataLoader` → forwards through the model → writes results batch-by-batch.
4. Saves two output files per slide.

**Output:**
```
results/features/patch512_step512_level0__rn50/
    ├── h5_files/<slide>.h5    ← HDF5: 'features' (N, D) + 'coords' (N, 2)
    └── pt_files/<slide>.pt    ← PyTorch FloatTensor (N, D)
```

Folder names encode patch parameters and model name, so different configurations never overwrite each other.

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
| `none` | `ToTensor` + ImageNet normalisation `(0.485/0.456/0.406)` |
| `reinhard` | Reinhard H&E stain normalisation → ImageNet range |
| `macenko` | Macenko stain normalisation (requires `torchstain`) |
| `uni_default` | Resize 224 + ImageNet normalisation |
| `gigapath_default` | Resize 256 → CenterCrop 224 + ImageNet normalisation |
| `hibou_default` | Resize/CenterCrop 224 + Hibou-specific statistics |
| `kaiko_default` | Resize/CenterCrop 224 + 0.5/0.5/0.5 normalisation |
| `optimus_default` | CenterCrop 224 + Optimus statistics |
| `colourjitter` | Random colour jitter (no normalisation) |
| `colourjitternorm` | Random colour jitter + ImageNet normalisation |

**Stain normalisation note:**  
`reinhard` and `macenko` use the pattern:
```
PIL → ToTensor → ×255 → NormClass() → [0,1] (C,H,W)
```
This matches what `torchstain`'s PyTorch backend expects internally.

---

## GPU Memory Guide

Batch size affects GPU memory linearly. Reference values for a **512×512 tile pipeline**:

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

Reports: CSV format, class distribution, duplicates, missing labels, SVS file coverage, feature file coverage.

---

### 7. Dataset Splitting (`split`)

```bash
python main.py split --config config/config.yaml
```

Config options (`split` block in `config.yaml`):

```yaml
split:
  type: train_test       # train_test | train_val_test
  train_size: 0.8
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
  encoding_size: 1536   # MUST match feature extractor output dim
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
- `best_model.pt` — dict: weights, optimizer, scheduler, config, class_map, metrics, timestamp
- `final_model.pt`
- `training_history.csv` — loss, acc, auc, f1, lr, time per epoch
- `config_snapshot.yaml`

---

### 9. Evaluation (`evaluate`)

```bash
# Auto-detect latest experiment:
python main.py evaluate --config config/config.yaml

# Specify experiment directory:
python main.py evaluate --config config/config.yaml --experiment results/experiments/metastasis/abmil_20260313_010000
```

**Outputs:** `<experiment_dir>/evaluate/`
- `predictions.csv` — slide_id, true_label, pred_label, prob_class0, prob_class1, ...
- `roc_data.csv` — fpr, tpr, thresholds (replot without rerunning)
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
python main.py heatmap --config config/config.yaml --experiment results/experiments/metastasis/abmil_...
```

**Outputs:** `<experiment_dir>/heatmaps/<slide_id>/`
- `<slide_id>_heatmap.png` — jet colormap overlay on WSI thumbnail
- `<slide_id>_attention_scores.csv` — coord_x, coord_y, attention per patch
- `top20_tiles/` — top 20 highest-attention tiles extracted from the raw WSI

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

---

## Logging

All commands write a structured log to `results/wsi_framework.log` (rotating, 10 MB).

