# WSI Classification Framework

A clean, modular, and YAML-configurable framework for Whole Slide Image (WSI) analysis tasks, including preprocessing, feature extraction, and Multiple Instance Learning (MIL).

## Available Commands

Run via the centralized CLI `main.py`:

```bash
python main.py --help

python main.py process --config config.yaml
```

Currently implemented modules:
- Reading WSIs (`core.wsi_reader`)
- Extracting whole-slide metadata (`core.wsi_reader`)
- Thumbnail Generation (`core.wsi_reader`)

See `docs/setup_and_usage.md` for environment setup instructions and full documentation.
