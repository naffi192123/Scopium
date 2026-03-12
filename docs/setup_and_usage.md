# WSI Classification Framework Documentation

## System Architecture
This framework is designed as a modular, experiment-friendly environment for Whole Slide Image (WSI) analysis tasks, including preprocessing, feature extraction, and MIL model training.

The core pipeline operates over:
- WSI reading and metadata extraction
- Tissue segmentation
- Patch extraction
- Feature encoding using CNNs or Foundation Models
- Weakly supervised learning with MIL algorithms

## Directory Structure
Dynamic dataset loading and results structuring driven by `config.yaml`.
- Dataset components:
  - `dataset/slides/`: Put raw `.svs`, `.tif`, etc. here.
  - `dataset/annotations/`: Manual annotations if any.
- Results components (configured in config.yaml):
  - `results/thumbnails/`
  - `results/metadata/`
  - `results/segmentation/`
  - `results/patches/`
  - `results/features/`
  - `results/models/`
  - `results/heatmaps/`
  - `results/logs/`

## Project Architecture
```text
wsi_framework/
├── config/                 # YAML configuration files
│   └── config.yaml         # Main pipeline configuration execution
├── core/                   # Core pipeline modules
│   ├── __init__.py
│   ├── wsi_reader.py       # Reads WSI, extracts metadata, thumbnails
│   ├── segmenter.py        # Tissue segmentation (Otsu, HSV, adaptive)
│   ├── patcher.py          # Patch coordinate extraction from segments
│   └── extractor.py        # Feature extraction (CNNs/Foundation Models)
├── datasets/               # Dataset handling
│   ├── __init__.py
│   ├── slide_dataset.py    # PyTorch dataset for slide coordinate loading
│   └── split_manager.py    # Cross-validation and split generation
├── models/                 # Deep learning models
│   ├── __init__.py
│   ├── feature_models.py   # Implementations of ResNet, HIPT, UNI, etc.
│   └── mil_models.py       # ABMIL, CLAM, TransMIL, etc.
├── pipelines/              # Orchestration scripts for CLI commands
│   ├── __init__.py
│   ├── preprocess.py       # Thumbnails, Metadata, Segmentation, Patching
│   ├── train.py            # Model training and validation loops
│   ├── evaluate.py         # Evaluation and metrics
│   └── visualize.py        # Heatmap reconstruction
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── config.py           # YAML parsing and directory initialization
│   └── metrics.py          # AUC, F1, Accuracy tracking
├── docs/                   # Documentation
│   ├── setup_and_usage.md
│   └── architecture.md     # This file
├── main.py                 # Unified Command-Line Interface (CLI)
└── requirements.txt        # Python dependencies
```

## Environment Setup

### Environment 1 — Local Development (Windows)
```bash
# 1. Open Anaconda Prompt
# 2. Navigate to the project directory
cd /path/to/wsi_framework

# 3. Create the conda environment
conda create -y -n dl_py39 python=3.9
conda activate dl_py39

# 4. Install requirements
pip install -r requirements.txt
```
*Note: For Windows, you may need to install the OpenSlide binaries manually and add them to your `PATH` if `openslide-python` fails to locate the library.*

### Environment 2 — HPC Cluster (CentOS)
```bash
# 1. Connect to GPU node and load modules
ssh gpu4
module load anaconda3/2023.03
module load cuda/12.2

# 2. Initialize conda and activate environment
eval "$(conda shell.bash hook)"

# 3. Create and activate the conda environment
conda create -y -n dl_py39 python=3.9
conda activate dl_py39

# 4. Navigate to project and install requirements
cd /path/to/wsi_framework
pip install -r requirements.txt
```

## Command-Line Usage

The `main.py` entrypoint serves as the primary way to interact with the framework. It relies on the configurations set in `config/config.yaml`.

### 1. Dataset Scanning (`stats`)
Scan the dataset directory and extract high-level metadata for all WSIs (dimensions, file size, pyramid levels, microns per pixel).
```bash
python main.py stats --config config/config.yaml
```
**Output:** `results/dataset_stats.csv`

### 2. Thumbnail & Metadata Extraction (`process`)
Verify WSI reading by generating thumbnails and extracting complete metadata dictionaries.
```bash
python main.py process --config config/config.yaml
```
**Output:** `results/thumbnails/*.png` and `results/metadata/*.json`

### 3. Tissue Segmentation & Patching (`segment`)
Process the WSIs to detect valid tissue regions (using HSV color space and morphological filtering) and extract valid patch coordinates.
It supports both `sequential` and `parallel` execution via the config `tiling.mode` setting.
```bash
python main.py segment --config config/config.yaml
```
**Output:** `results/masks/*_mask.png` and `results/patches/patch512_step512_level0/*.h5`

### 4. Debugging Patch Extraction
Because tiles are extracted only from segmented tissue regions and stored as coordinates in `.h5` files, it is crucial to verify that the extraction logic is capturing the right amount of tissue. You can use the following commands to debug your extraction pipeline.

**Visualize Segmented Tiles on Thumbnail (`debug-segmentation`)**
This reconstructs an approximate tissue coverage map directly on the WSI thumbnail by plotting every single extracted patch coordinate as a green bounding box.
```bash
python main.py debug-segmentation --config config/config.yaml
```
**Output:** `results/debug/segmented_tiles_thumbnail_*.png`

**Single Tile Extraction (`extract-tile`)**
This command selects one tile coordinate from the `.h5` dataset, physically extracts that specific high-resolution tile from the `.svs` file using the tiling parameters, and saves it to disk for manual visual inspection.
```bash
python main.py extract-tile --config config/config.yaml
```
**Output:** `results/debug/tile_*.png`

### 5. Feature Extraction (`extract`)
This module maps the previously extracted `.h5` patch coordinates directly to a PyTorch `Dataset` that natively reads high-resolution tiles directly from the WSI memory stream using OpenSlide, allowing deep learning embeddings to be generated without unpacking millions of `.png` files to disk.

It supports configuring multiple sequential data augmentation/normalization transforms entirely from `config.yaml` (`totensor`, `resize_224`, `macenko`, `imagenet`, etc.).
It also supports over a dozen Baseline CNNs and Pathology Foundation Models (`rn50`, `uni`, `hibou_b`, `virchow`, `provgigapath`, etc.) operating automatically on detected CUDA GPUs.

**To Run Feature Extraction:**
```bash
python main.py extract --config config/config.yaml
```

**Testing Output**
By default, the framework will create robust subdirectories named after your pipeline (`results/features/patch512_step512_level0__rn50_reinhard-imagenet/...`) to prevent colliding different datasets.
Features for each slide are saved as PyTorch `.pt` tensors inside the respective `pt_files/` subdirectory.
