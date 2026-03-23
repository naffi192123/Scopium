# Scopium — WSI Classification Framework

A modular, YAML-configurable framework for Whole Slide Image (WSI) analysis — from raw slide ingestion through tissue segmentation, feature extraction, and Multiple Instance Learning (MIL) classification.

---

## Quick Start

```bash
conda activate dl_py39
cd wsi_framework

# 1. Scan dataset metadata
python main.py stats --config config/config.yaml

# 2. Generate thumbnails & JSON metadata
python main.py process --config config/config.yaml

# 3. Tissue segmentation + patch coordinate extraction
python main.py segment --config config/config.yaml

# 4. (Optional) Verify segmentation visually
python main.py debug-segmentation --config config/config.yaml

# 5. GPU feature extraction -> .pt + .h5 per slide
python main.py extract --config config/config.yaml

# 6. Analyse annotation CSV
python main.py analyse --config config/config.yaml

# 7. Create train/val/test splits
python main.py split --config config/config.yaml

# 8. Train MIL model
python main.py train --config config/config.yaml

# 9. Evaluate on test set
python main.py evaluate --config config/config.yaml

# 10. Generate attention heatmaps
python main.py heatmap --config config/config.yaml

# 11. Run patch-level classifier inference
python main.py classify --config config/config.yaml

# 12. Generate prediction heatmaps
python main.py classify-heatmap --config config/config.yaml
```

---

## CLI Command Reference

| Command | Description |
|---|---|
| `stats` | Scan `slides_dir` → `dataset_stats.csv` |
| `process` | Thumbnail PNGs + JSON metadata per slide |
| `segment` | Tissue segmentation → patch coordinate `.h5` files |
| `debug-segmentation` | Tile overlay on WSI thumbnail |
| `extract-tile` | Single high-res tile extraction for visual QC |
| `extract` | GPU feature extraction → `.pt` + `.h5` per slide |
| `analyse` | Annotation CSV validation + class distribution |
| `split` | Train/test (or train/val/test) split generation |
| `train` | MIL model training with history + checkpointing |
| `evaluate` | Evaluation metrics, ROC curves, confusion matrix |
| `heatmap` | Attention heatmap overlay + top-20 tile extraction |
| `classify` | Patch-level classifier inference → CSV predictions + filtered features |
| `classify-heatmap` | Tile-level prediction heatmaps (categorical or confidence) |

---

## Implemented Modules

| Module | Status | Key File |
|---|---|---|
| WSI Reading & Metadata | ✅ | `core/wsi_reader.py` |
| Tissue Segmentation | ✅ | `core/segmenter.py` |
| Patch Extraction | ✅ | `core/patcher.py` |
| Debugging Utilities | ✅ | `pipelines/debug.py` |
| Feature Extraction | ✅ | `pipelines/extract.py` |
| Annotation Analysis | ✅ | `pipelines/analyse_annotations.py` |
| Dataset Splitting | ✅ | `pipelines/split.py` |
| MIL Training | ✅ | `pipelines/train.py` |
| MIL Evaluation | ✅ | `pipelines/evaluate.py` |
| Attention Heatmaps | ✅ | `pipelines/visualize.py` |
| Patch Classification | ✅ | `pipelines/classify.py` |
| Prediction Heatmaps | ✅ | `pipelines/classify_heatmap.py` |

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

```yaml
task:
  name: metastasis
  type: binary          # binary | multiclass
  num_classes: 2
  class_names: [benign, malignant]

mil:
  model: abmil           # see model table above
  encoding_size: 1536    # must match feature extractor dim
  hidden_dim: 256
  dropout: 0.25

training:
  max_epochs: 100
  learning_rate: 0.0002
  early_stopping: true
  early_stopping_patience: 20

split:
  type: train_test       # train_test | train_val_test
  test_size: 0.2
  stratified: true
  random_seed: 42

# Feature subfolder overrides (both optional, null = auto-derive)
tiling:
  patches_subfolder_override: null   # e.g. "patch256_step256_level0_otsu"
feature_extraction:
  # BASE name — model always auto-appended:
  features_subfolder_override: null  # e.g. "patch512_step512_level0"
  # EXACT name — model NOT appended (highest config priority):
  features_dir_override: null        # e.g. "patch512_step512_level0__uni"
```

> **CLI flags** take priority over config keys:
> - `--feature_dir patch512_step512_level0__uni` — exact dir, model NOT appended
> - `--features patch512_step512_level0` — base dir, model auto-appended
> - `--patches patch256_step256_level0` — selects patch coordinate set

---

## Training Output Structure

```
results/experiments/<task>/<model>_<timestamp>/
    best_model.pt             # weights + optimizer + config + class_map
    final_model.pt            # final epoch checkpoint
    training_history.csv      # loss/acc/auc/f1/lr/time per epoch
    config_snapshot.yaml      # exact config used for this run
    evaluate/
        predictions.csv       # slide_id, true_label, pred_label, probs
        roc_data.csv          # fpr, tpr, thresholds (replot anytime)
        confusion_matrix.csv
        classification_report.txt
        metrics.json
        roc_curve.png
        confusion_matrix.png
    heatmaps/<slide_id>/
        <slide>_heatmap.png
        <slide>_attention_scores.csv
        top20_tiles/
            01_<slide>_x..._y..._a....png
            ...
```
