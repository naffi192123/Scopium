# WSI Framework — Setup & Usage Guide

---

## Environment Setup

### Windows (Local Development)

```bash
# 1. Create the conda environment
conda create -y -n dl_py39 python=3.9
conda activate dl_py39

# 2. Navigate to the project and install dependencies
cd path\to\wsi_framework
pip install -r requirements.txt
```

> **Windows note:** OpenSlide binaries must be installed manually.
> Download from [openslide.org](https://openslide.org/download/) and add the `bin\` folder to your `PATH`.
> Install `torchstain` separately if using stain normalisation:
> ```bash
> pip install torchstain
> ```

### HPC Cluster (CentOS / SLURM)

```bash
ssh gpu4
module load anaconda3/2023.03
module load cuda/12.2
eval "$(conda shell.bash hook)"

conda create -y -n dl_py39 python=3.9
conda activate dl_py39

cd /path/to/wsi_framework
pip install -r requirements.txt
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

## Logging

All commands write a structured log to `results/wsi_framework.log` (rotating, 10 MB). Console output mirrors the log at the same level.
