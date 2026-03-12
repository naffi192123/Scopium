# WSI Classification Framework

A clean, modular, and YAML-configurable framework for Whole Slide Image (WSI) analysis tasks, including preprocessing, feature extraction, and Multiple Instance Learning (MIL).

## Available Commands

Run via the centralized CLI `main.py`:

```bash
python main.py --help

python main.py stats --config config/config.yaml
python main.py process --config config/config.yaml
python main.py segment --config config/config.yaml
```

Currently implemented modules:
- Reading WSIs (`core.wsi_reader`)
- Extracting whole-slide metadata and Scanning Datasets (`core.wsi_reader`, `main.py stats`)
- Thumbnail Generation (`core.wsi_reader`)
- Tissue Segmentation via HSV/Otsu (`core.segmenter`)
- Patch Coordinate Extraction & HDF5 saving (`core.patcher`)

## Configuration
The pipeline is entirely driven by `config/config.yaml`.
Key parameter groups include:
- `paths`: Defines `slides_dir` and dynamically structured `results_dir`.
- `tiling`: Controls `patch_size`, `step_size`, `level`, and `mode`.
- `feature_extraction`: Sets the `model` (e.g., `rn50`) and `transforms`.
- `task`: Defines prediction objective (`name` & `type`).
- `mil`: Sets Multiple Instance Learning backend (`model` & `encoding_size`).

See `docs/setup_and_usage.md` for environment setup instructions and full documentation.
