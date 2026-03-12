# Scopium — WSI Classification Framework

A clean, modular, and YAML-configurable framework for Whole Slide Image (WSI) analysis — from raw slide ingestion through tissue segmentation, deep learning feature extraction, and Multiple Instance Learning (MIL) classification.

---

## Quick Start

```bash
conda activate dl_py39
cd wsi_framework

# 1. Scan dataset and view slide metadata
python main.py stats --config config/config.yaml

# 2. Generate thumbnails and JSON metadata
python main.py process --config config/config.yaml

# 3. Tissue segmentation + patch coordinate extraction
python main.py segment --config config/config.yaml

# 4. (Optional) Verify segmentation visually
python main.py debug-segmentation --config config/config.yaml
python main.py extract-tile --config config/config.yaml

# 5. Feature extraction (GPU)
python main.py extract --config config/config.yaml
```

---

## CLI Command Reference

| Command | Description |
|---|---|
| `stats` | Scan `slides_dir` and save `dataset_stats.csv` |
| `process` | Generate thumbnail PNGs and JSON metadata per slide |
| `segment` | Tissue segmentation → patch coordinate `.h5` files |
| `debug-segmentation` | Overlay patch tiles onto WSI thumbnail for visual verification |
| `extract-tile` | Extract and save a single high-res tile for inspection |
| `extract` | GPU feature extraction → `.pt` + `.h5` files per slide |
| `train` | *(coming soon)* MIL model training |
| `evaluate` | *(coming soon)* ROC, AUC, classification metrics |
| `heatmap` | *(coming soon)* Spatial attention heatmap generation |

---

## Implemented Modules

| Module | Status | Description |
|---|---|---|
| WSI Reading & Metadata | ✅ | `core/wsi_reader.py` |
| Dataset Statistics | ✅ | `main.py stats` |
| Thumbnail Generation | ✅ | `core/wsi_reader.py` |
| Tissue Segmentation | ✅ | `core/segmenter.py` (HSV + Otsu) |
| Patch Extraction | ✅ | `core/patcher.py` → HDF5 coords |
| Debugging Utilities | ✅ | `pipelines/debug.py` |
| Feature Extraction | ✅ | `pipelines/extract.py` → `.pt` + `.h5` |
| MIL Training | 🔜 | `pipelines/train.py` |
| Evaluation | 🔜 | `pipelines/evaluate.py` |
| Attention Heatmaps | 🔜 | `pipelines/visualize.py` |

---

## Supported Feature Extractors

| Model Key | Architecture | Embedding Dim |
|---|---|---|
| `rn18` | ResNet-18 (ImageNet) | 512 |
| `rn50` | ResNet-50 (ImageNet) | 2048 |
| `vit_l` | ViT-Large (ImageNet, timm) | 1024 |
| `uni` | UNI (MahmoodLab) | 1024 |
| `provgigapath` | Prov-GigaPath (Microsoft) | 1536 |
| `phikon` | Phikon (Owkin) | 768 |
| `hibou_b` | Hibou-B (HistAI) | 768 |
| `hibou_l` | Hibou-L (HistAI) | 1024 |
| `optimus` | H-Optimus-0 (BioOptimus) | 1536 |
| `virchow` | Virchow (Paige) | 2560 |
| `virchow2cls` | Virchow-2 CLS (Paige) | 1280 |

---

## Supported Transforms

Set via `feature_extraction.transforms` in `config.yaml`:

| Key | Description |
|---|---|
| `auto` | Canonical pipeline for the selected model (recommended) |
| `none` | ToTensor + ImageNet normalization |
| `reinhard` | Reinhard stain normalization + ImageNet normalization |
| `macenko` | Macenko stain normalization |
| `uni_default` | Resize 224 + ImageNet normalization |
| `gigapath_default` | Resize 256 + CenterCrop 224 + ImageNet normalization |
| `hibou_default` | Resize + CenterCrop 224 + Hibou statistics |
| `colourjitter` | Color jitter augmentation |

---

## Feature Extraction Output

Both `.pt` and `.h5` files are saved per slide, matching the reference repo behaviour:

```
results/features/patch512_step512_level0__rn50/
    ├── h5_files/
    │   └── <slide_name>.h5      # HDF5: 'features' (N,D) + 'coords' (N,2)
    └── pt_files/
        └── <slide_name>.pt      # PyTorch tensor (N, D)
```

---

## Configuration

The entire pipeline is driven by a single `config/config.yaml`. Key sections:

```yaml
feature_extraction:
  model: rn50           # Model key (see table above)
  batch_size: 64        # Reduce if GPU OOM
  transforms: auto      # 'auto' | 'none' | 'reinhard' | 'macenko' | ...
  weights_path: null    # Local weights path (required for some models)
```

See `docs/setup_and_usage.md` for full environment setup, parameter reference, and HPC instructions.
