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
wsi_classification/
├── config.yaml             # Main configuration execution
├── main.py                 # Unified Command-Line Interface (CLI)
├── README.md               # Quick start documentation
├── requirements.txt        # Dependencies
├── docs/                   # Full documentation suite
├── experiments/            # Storage for experiment sweeps
└── wsi_lib/                # Core pipeline modules
    ├── config.py           # Config loader and dir validator
    ├── logger.py           # Logging utility
    ├── utils.py            # General utilities
    ├── preprocessing/      # WSI reader, thumbnails, global metadata
    │   ├── thumbnail.py
    │   └── metadata.py
    ├── tiling/             # Patch extraction and tissue segmentation
    │   └── tiler.py
    ├── features/           # Inference encoders (CNNs/PfMs)
    │   ├── extractor.py
    │   ├── models.py
    │   └── transforms.py
    ├── mil/                # Multiple Instance Learning backend
    │   ├── dataset.py      
    │   ├── evaluation.py
    │   └── trainer.py
    ├── models/             # PyTorch MIL model architectures
    └── visualization/      # Attention heatmaps and ROCs
        └── heatmap.py
```

## Environment Setup

### Environment 1 — Local Development (Windows)
```bash
# 1. Open Anaconda Prompt
# 2. Navigate to the project directory
cd /path/to/wsi_classification

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
cd /path/to/wsi_classification
pip install -r requirements.txt
```

## First Task: Thumbnail & Metadata Extraction
Run the following testing commands to verify WSI reading and metadata extraction.

**1. Copy sample WSIs**
Copy 1-2 Sample `.svs` files into `wsi_classification/dataset/slides/`.

**2. Run the process command**
```bash
python main.py process
```

**3. Verify Output**
Check `wsi_classification/results/thumbnails/` and `wsi_classification/results/metadata/` for generated `.png` thumbnails and `.json` metadata files.
