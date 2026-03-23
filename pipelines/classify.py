"""
pipelines/classify.py

Patch-level classifier inference pipeline.

What it does
------------
1. Loads a pretrained classifier from ``patch_classifier.checkpoint_path``.
2. Iterates over all feature files in the active feature directory.
3. For each slide: runs batched softmax inference, writes a predictions CSV,
   and generates filtered .h5 / .pt files for the requested tissue categories.

Outputs
-------
results/patch_predictions/{feature_subfolder}/{slide_id}.csv
  columns: coord_x, coord_y, predicted_label, confidence,
           ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM   (one column per class)

results/features/{feature_subfolder}__{CATEGORY}/
  h5_files/{slide_id}.h5   — 'features' (M, D)  + 'coords' (M, 2)
  pt_files/{slide_id}.pt   — FloatTensor (M, D)

where M ≤ N is the number of patches belonging to that category.

Config section
--------------
patch_classifier:
  checkpoint_path: "outputs/checkpoints/BEST_MODEL.pth"
  batch_size: 512
  input_format: h5        # 'h5' or 'pt'
  filter_categories:      # tissue classes to keep in filtered outputs
    - TUM
    - STR
"""

import os
import logging
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.classifier import load_classifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDF5 write helper (reuses the same pattern as extract.py)
# ---------------------------------------------------------------------------

def _write_h5(path: str, features: np.ndarray, coords: np.ndarray):
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features, chunks=(1,) + features.shape[1:])
        f.create_dataset("coords",   data=coords,   chunks=(1,) + coords.shape[1:])


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def _load_h5(h5_path: str):
    """Returns (features np.float32 (N,D), coords np.int64 (N,2))."""
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]        # (N, D)
        coords   = f["coords"][:]          # (N, 2)
    return features.astype(np.float32), coords.astype(np.int64)


def _load_pt(pt_path: str):
    """Returns (features np.float32 (N,D), coords np.int64 (N,2) as row indices)."""
    t = torch.load(pt_path, map_location="cpu")
    features = t.numpy().astype(np.float32) if isinstance(t, torch.Tensor) else t
    n = len(features)
    # No coordinate information available in .pt-only mode — use patch indices
    coords = np.stack([np.arange(n), np.zeros(n, dtype=np.int64)], axis=1)
    return features, coords


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def command_classify(config: dict, dirs_dict: dict, log=None):
    """
    Run patch-level classifier inference over all feature files.

    Parameters
    ----------
    config    : dict  Full parsed YAML config.
    dirs_dict : dict  Directory paths set up by utils.config.setup_directories.
    log       : Logger (falls back to module logger if None).
    """
    _log = log or logger

    # ── Config ───────────────────────────────────────────────────────────────
    clf_cfg    = config.get("patch_classifier", {})
    ckpt_path  = clf_cfg.get("checkpoint_path")
    batch_size = int(clf_cfg.get("batch_size", 512))
    fmt        = clf_cfg.get("input_format", "h5").lower().strip()
    filter_cats = clf_cfg.get("filter_categories", [])

    if not ckpt_path or not os.path.exists(ckpt_path):
        _log.error(
            f"Classifier checkpoint not found: '{ckpt_path}'. "
            "Set patch_classifier.checkpoint_path in config.yaml.")
        return

    if filter_cats == "all" or filter_cats is None:
        filter_cats = []   # handled later — None means save all categories

    # ── Device & Model ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info(f"Patch Classifier | device: {device}")

    try:
        model, class_names = load_classifier(ckpt_path, device=device)
    except Exception as exc:
        _log.error(f"Failed to load classifier: {exc}")
        return

    _log.info(f"Classes: {class_names}")
    _log.info(f"Filter categories: {filter_cats if filter_cats else 'all'}")

    # ── Directories ───────────────────────────────────────────────────────────
    features_dir = dirs_dict["features"]
    results_root = config["paths"]["results_dir"]
    feat_subfolder = dirs_dict.get("_feature_subfolder",
                                   os.path.basename(features_dir))

    # Predictions output directory
    pred_dir = os.path.join(results_root, "patch_predictions", feat_subfolder)
    os.makedirs(pred_dir, exist_ok=True)

    # Pre-create filtered feature directories for each requested category
    active_cats = filter_cats if filter_cats else class_names
    filtered_dirs = {}
    for cat in active_cats:
        cat_upper = cat.upper()
        cat_feat_dir = os.path.join(results_root, "features",
                                    f"{feat_subfolder}__{cat_upper}")
        filtered_dirs[cat_upper] = {
            "h5": os.path.join(cat_feat_dir, "h5_files"),
            "pt": os.path.join(cat_feat_dir, "pt_files"),
        }
        os.makedirs(filtered_dirs[cat_upper]["h5"], exist_ok=True)
        os.makedirs(filtered_dirs[cat_upper]["pt"], exist_ok=True)

    # ── Find feature files ────────────────────────────────────────────────────
    if fmt == "h5":
        feat_subdir = os.path.join(features_dir, "h5_files")
        file_ext    = ".h5"
        loader_fn   = _load_h5
    else:
        feat_subdir = os.path.join(features_dir, "pt_files")
        file_ext    = ".pt"
        loader_fn   = _load_pt

    if not os.path.isdir(feat_subdir):
        _log.error(
            f"Feature subdirectory not found: {feat_subdir}\n"
            f"  Run `python main.py extract --config ...` first, or "
            f"check input_format (currently '{fmt}').")
        return

    feat_files = sorted(
        f for f in os.listdir(feat_subdir) if f.endswith(file_ext))
    if not feat_files:
        _log.error(f"No *{file_ext} files found in {feat_subdir}.")
        return

    _log.info(f"Found {len(feat_files)} slides to classify.")

    success = failed = 0
    wall_start = time.time()

    # ── Per-slide loop ────────────────────────────────────────────────────────
    for feat_file in feat_files:
        slide_name = os.path.splitext(feat_file)[0]
        feat_path  = os.path.join(feat_subdir, feat_file)
        pred_csv   = os.path.join(pred_dir, slide_name + ".csv")

        if os.path.exists(pred_csv):
            _log.info(f"[{slide_name}] Prediction CSV already exists — skipping.")
            success += 1
            continue

        # Load features & coords
        try:
            features_np, coords_np = loader_fn(feat_path)
        except Exception as exc:
            _log.error(f"[{slide_name}] Load error: {exc}")
            failed += 1
            continue

        if len(features_np) == 0:
            _log.warning(f"[{slide_name}] Empty feature file — skipping.")
            continue

        N, D = features_np.shape
        _log.info(f"[{slide_name}] {N} patches | feat_dim={D}")

        # Batched inference
        feat_tensor = torch.from_numpy(features_np)
        dataset_    = TensorDataset(feat_tensor)
        loader_     = DataLoader(dataset_, batch_size=batch_size,
                                 shuffle=False, num_workers=0)

        all_probs   = []
        slide_start = time.time()

        try:
            with torch.no_grad():
                for (batch,) in loader_:
                    batch = batch.to(device)
                    logits = model(batch)
                    probs  = F.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
        except Exception as exc:
            _log.error(f"[{slide_name}] Inference error: {exc}")
            failed += 1
            continue

        all_probs  = np.concatenate(all_probs, axis=0)   # (N, C)
        pred_idx   = all_probs.argmax(axis=1)              # (N,)
        confidence = all_probs.max(axis=1)                 # (N,)
        pred_label = [class_names[i] for i in pred_idx]

        # ── Save predictions CSV ──────────────────────────────────────────────
        row_dict = {
            "coord_x":        coords_np[:, 0],
            "coord_y":        coords_np[:, 1],
            "predicted_label": pred_label,
            "confidence":     np.round(confidence, 6),
        }
        for j, cls in enumerate(class_names):
            row_dict[cls] = np.round(all_probs[:, j], 6)

        df_pred = pd.DataFrame(row_dict)
        df_pred.to_csv(pred_csv, index=False)

        elapsed = time.time() - slide_start
        _log.info(
            f"[{slide_name}] Done | {N} patches | {elapsed:.1f}s | "
            f"CSV → {pred_csv}")

        # ── Generate filtered feature files ───────────────────────────────────
        for cat in active_cats:
            cat_upper = cat.upper()
            if cat_upper not in class_names:
                _log.warning(
                    f"Category '{cat_upper}' not in class_names {class_names} — skipping.")
                continue

            mask = np.array([lbl == cat_upper for lbl in pred_label])
            if mask.sum() == 0:
                _log.info(f"  [{slide_name}] No patches for category {cat_upper}.")
                continue

            cat_features = features_np[mask]      # (M, D)
            cat_coords   = coords_np[mask]         # (M, 2)
            M = int(mask.sum())

            # .h5
            h5_path = os.path.join(
                filtered_dirs[cat_upper]["h5"], slide_name + ".h5")
            _write_h5(h5_path, cat_features, cat_coords)

            # .pt
            pt_path = os.path.join(
                filtered_dirs[cat_upper]["pt"], slide_name + ".pt")
            torch.save(torch.from_numpy(cat_features), pt_path)

            _log.info(
                f"  [{slide_name}] {cat_upper}: {M}/{N} patches saved.")

        success += 1

    total = time.time() - wall_start
    _log.info(
        f"Classification complete | success={success} | failed={failed} | "
        f"wall_time={total:.1f}s")
    _log.info(f"Predictions saved to: {pred_dir}")
    for cat, paths in filtered_dirs.items():
        _log.info(f"  Filtered [{cat}]:  {paths['h5']}")
