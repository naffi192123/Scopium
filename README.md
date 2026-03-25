# Scopium — WSI Classification Framework

A modular, YAML-configurable framework for Whole Slide Image (WSI) analysis — from raw slide ingestion through tissue segmentation, feature extraction, Multiple Instance Learning (MIL) classification, hyperparameter optimisation, and cross-validation.

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

# 13. Hyperparameter optimisation
python main.py hpo --config config/config.yaml

# 14. K-fold cross-validation
python main.py crossval --config config/config.yaml
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
| `train` | MIL model training with history, checkpointing, LR scheduling |
| `evaluate` | Evaluation metrics, ROC curves, confusion matrix |
| `heatmap` | Attention heatmap overlay + top-20 tile extraction |
| `classify` | Patch-level classifier inference → CSV predictions + filtered features |
| `classify-heatmap` | Tile-level prediction heatmaps (categorical or confidence) |
| `hpo` | Optuna HPO study — searches model arch, LR, regularisation |
| `crossval` | Stratified K-fold cross-validation |

### Key CLI Flags

| Flag | Applies to | Description |
|---|---|---|
| `--features <dir>` | `train`, `evaluate`, `heatmap`, `hpo`, `crossval` | **Exact** feature directory (model suffix not appended) |
| `--feature_dir <dir>` | same as above | Alias for `--features` |
| `--patches <dir>` | `extract`, `train` | Patch coordinate set override |
| `--experiment <dir>` | `evaluate`, `heatmap` | Use a specific experiment directory |
| `--use_best_config` | `train`, `crossval` | Load best HPO config before running |

---

## Feature Directory Selection

Feature extraction can produce multiple directories under `results/features/`:

```
results/features/
    patch512_step512_level0__optimus__TUM/
        pt_files/   h5_files/
    patch256_step256_level0__uni/
        pt_files/   h5_files/
```

**Resolution priority (highest first):**

1. CLI `--features` / `--feature_dir` — exact dir, no suffix appended
2. `feature_extraction.features_dir_override` in config — exact dir
3. `feature_extraction.features_subfolder_override` + auto-appended model suffix
4. Auto-derived from tiling params + model name

```bash
# Use a specific feature dir
python main.py train  --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py hpo    --config config/config.yaml --features patch512_step512_level0__optimus__TUM
python main.py crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM
```

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
| Hyperparameter Optimisation | ✅ | `pipelines/hpo.py` |
| K-Fold Cross-Validation | ✅ | `pipelines/crossval.py` |

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
  weight_decay: 1e-4
  optimizer: AdamW                # AdamW | Adam
  lr_scheduler: plateau           # plateau | cosine | step
  warmup_epochs: 0
  early_stopping: true
  early_stopping_patience: 20
  label_smoothing: 0.0            # [0.0 – 0.15] recommended
  patch_dropout: 0.0              # drop random fraction of patches per bag
  patch_shuffle: false            # shuffle patch order per bag
  max_patches: null               # cap bag size (null = no cap)

split:
  type: train_val_test    # train_test | train_val_test
  test_size: 0.2
  val_size: 0.1
  stratified: true
  random_seed: 42

# Feature subfolder overrides (both optional, null = auto-derive)
tiling:
  patches_subfolder_override: null
feature_extraction:
  features_subfolder_override: null   # base name — model auto-appended
  features_dir_override: null         # exact name — model NOT appended
```

---

## Hyperparameter Optimisation (HPO)

HPO uses [Optuna](https://optuna.org/) to automatically search over model architecture, optimizer, regularization, and scheduler. Studies are resumable via SQLite.

```bash
pip install optuna tqdm   # tqdm enables live progress bars
```

### HPO Workflow

```bash
# 1. Run HPO study (30 trials × 30 epochs each by default)
python main.py hpo --config config/config.yaml

# With a specific features dir
python main.py hpo --config config/config.yaml --features patch512_step512_level0__optimus__TUM

# 2. Train final model using the best-found hyperparameters
python main.py train --config config/config.yaml --use_best_config

# 3. Cross-validate with best config
python main.py crossval --config config/config.yaml --use_best_config
```

### HPO Search Space (all configurable in `hpo.search_space`)

| Parameter | Type | Default Range |
|---|---|---|
| `model` | categorical | all 7 MIL models |
| `optimizer` | categorical | AdamW, Adam |
| `learning_rate` | log-uniform | [1e-4, 2e-3] |
| `weight_decay` | log-uniform | [1e-5, 5e-4] |
| `dropout` | uniform | [0.1, 0.6] |
| `attn_hidden_dim` | categorical | 32, 64, 128 |
| `lr_scheduler` | categorical | cosine, step, plateau |
| `label_smoothing` | uniform | [0.0, 0.15] |
| `patch_dropout` | uniform | [0.0, 0.3] |
| `max_patches` | categorical | null, 1000, 2000 |
| `warmup_epochs` | categorical | 0, 2, 5 |

### K-Fold CV within HPO Trials

Set `hpo.n_folds > 1` to average the metric across folds per trial:

```yaml
hpo:
  n_folds: 5       # 1 = standard train/val split (default)
  n_trials: 30
  epochs_per_trial: 30
  metric: val_auc
  direction: maximize
```

### HPO Experiment Tracking

Each HPO invocation creates a **separate, isolated experiment directory**. This means re-running with different features, configs, or timestamps never overwrites a previous study:

```
results/hpo/
    mil_hpo__patch512_step512_level0__optimus__TUM__20241015_143022/   ← run 1
        experiment_info.json    ← run metadata (features, timestamp, task, device)
        base_config.yaml        ← exact config used for this run
        trial_0000/
            trial_config.yaml   ← hyperparameters for this trial
            trial_metrics.json  ← per-epoch val AUC/acc/f1
        ...
        best_config.yaml        ← best hyperparameters (merged config)
        best_trial.json         ← best trial summary + feature dir used
        hpo_results.csv         ← all trials sorted by metric
        study.db                ← Optuna SQLite (run is individually resumable)
    mil_hpo__patch256_step256_level0__uni__20241016_090011/            ← run 2
        ...
```

When using `--use_best_config`, the **most recently completed run** is automatically selected. To pin a specific run, add to `config.yaml`:

```yaml
hpo:
  best_run_path: results/hpo/mil_hpo__patch512_step512_level0__optimus__TUM__20241015_143022
```

## K-Fold Cross-Validation

Pools train + val splits and runs stratified K-fold, training a fresh model per fold.

```bash
# Standard cross-validation
python main.py crossval --config config/config.yaml

# Cross-validation using best HPO configuration
python main.py crossval --config config/config.yaml --use_best_config

# Cross-validation with a specific feature directory
python main.py crossval --config config/config.yaml --features patch512_step512_level0__optimus__TUM

# Combine: best HPO config + specific feature directory
python main.py crossval --config config/config.yaml --use_best_config --features patch512_step512_level0__optimus__TUM
```

```yaml
crossval:
  study_name: crossval   # output label
  n_folds: 5
  seed: 42
```

#### CV Outputs

```
results/crossval/<study_name>/
    fold_01/
        best_model.pt          ← best checkpoint per fold
        fold_metrics.json
    ...
    cv_summary.json            ← mean ± std across folds
    cv_summary.csv
```

---

## Overfitting Mitigation

The framework includes several regularization tools controllable from config or HPO:

| Technique | Config Key | HPO-Tunable |
|---|---|---|
| Label smoothing | `training.label_smoothing` | ✅ |
| Patch-level dropout | `training.patch_dropout` | ✅ |
| Patch shuffling | `training.patch_shuffle` | – |
| Bag size cap | `training.max_patches` | ✅ |
| Weight decay | `training.weight_decay` | ✅ |
| LR warmup | `training.warmup_epochs` | ✅ |
| LR scheduling | `training.lr_scheduler` | ✅ |
| Early stopping | `training.early_stopping` | – |

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
        roc_data.csv
        confusion_matrix.csv
        classification_report.txt
        metrics.json
        roc_curve.png
        confusion_matrix.png
    heatmaps/<slide_id>/
        <slide>_heatmap.png
        <slide>_attention_scores.csv
        top20_tiles/
```

---

## Parameter Validation

HPO and cross-validation validate all required config keys **before** starting:

```
HPO: config missing required key 'task.class_names'
HPO aborted due to missing configuration keys.
```

Required: `mil.model`, `task.name`, `task.num_classes`, `task.class_names`, `training.max_epochs`, `paths.results_dir`.

---

## Documentation

| File | Contents |
|---|---|
| `docs/setup_and_usage.md` | Full installation, commands, feature dir resolution, HPO, CV |
| `docs/architecture.md` | Pipeline architecture and module relationships |
| `config/config.yaml` | Fully-annotated configuration file |
