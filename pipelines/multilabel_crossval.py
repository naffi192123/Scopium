"""
pipelines/multilabel_crossval.py

K-Fold Cross-Validation for MIL Multi-Label Classification.

Supports:
  - Iterative stratification (scikit-multilearn) when available (recommended)
  - Falls back to StratifiedKFold on most-common label when skmultilearn absent
  - Integration with --use_best_config for HPO param reuse

Output directory structure (unique per invocation):
    results/multilabel/crossval/
        <task_name>/
            <model>__<feat_dir_label>__<YYYYMMDD_HHMMSS>/
                experiment_info.json   — full provenance metadata
                config_snapshot.yaml   — exact config used
                combined_pool.csv      — merged train+val pool
                fold_01/
                    best_model.pt
                    fold_metrics.json
                ...
                cv_summary.json        — mean ± std across folds
                cv_summary.csv

CLI:
    python main.py multilabel-crossval --config config/config.yaml
    python main.py multilabel-crossval --config config/config.yaml --use_best_config
    python main.py multilabel-crossval --config config/config.yaml --features <dir>
    python main.py multilabel-crossval --config config/config.yaml --use_best_config \\
                                        --features <dir>
"""

from __future__ import annotations

import os
import copy
import json
import logging
import datetime
import yaml
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

# Try iterative stratification (preferred for multi-label)
try:
    from skmultilearn.model_selection import IterativeStratification
    _SKMULTILEARN = True
except ImportError:
    _SKMULTILEARN = False

from datasets.mil_multilabel_dataset import (
    MultiLabelMILDataset, multilabel_collate_fn,
)
from models.mil_multilabel_models import build_multilabel_model
from pipelines.multilabel_train import (
    _run_ml_epoch, _build_multilabel_loss,
    _save_ml_checkpoint, _compute_ml_metrics,
)

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_ml_cv_config(config: dict, log) -> bool:
    required = {
        "mil"      : ["model", "encoding_size"],
        "multilabel": ["label_names"],
        "multilabel_training": ["learning_rate", "weight_decay", "max_epochs"],
        "paths"    : ["results_dir"],
    }
    ok = True
    for section, keys in required.items():
        for key in keys:
            if config.get(section, {}).get(key) is None:
                log.error(f"Config missing required key: {section}.{key}")
                ok = False
    return ok


def _iterative_split(label_matrix: np.ndarray, n_splits: int, seed: int):
    """
    Perform iterative stratification for multi-label data.
    Returns list of (train_indices, val_indices) tuples.
    """
    if _SKMULTILEARN:
        k_fold = IterativeStratification(n_splits=n_splits, order=1,
                                         random_state=seed)
        splits = []
        for tr, va in k_fold.split(
                np.arange(len(label_matrix)).reshape(-1, 1), label_matrix):
            splits.append((tr, va))
        return splits
    else:
        logger.warning(
            "scikit-multilearn not installed — using StratifiedKFold on "
            "most-common label as proxy for multi-label stratification.\n"
            "Install with: pip install scikit-multilearn")
        proxy_labels = label_matrix.argmax(axis=1)
        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(skf.split(np.arange(len(label_matrix)), proxy_labels))
        return splits


# ─── Per-fold training ────────────────────────────────────────────────────────────

def _train_cv_fold(
    config: dict,
    train_subset,
    val_subset,
    label_names: List[str],
    device: torch.device,
    fold_dir: str,
    fold_idx: int,
    log,
) -> Dict[str, float]:
    """Train one CV fold. Returns best-epoch val metrics."""
    train_cfg  = config.get("multilabel_training", {})
    nw         = int(config.get("training", {}).get("num_workers", 0))
    max_ep     = int(train_cfg.get("max_epochs", 100))
    patch_drop = float(train_cfg.get("patch_dropout", 0.0))
    patch_shuf = bool(train_cfg.get("patch_shuffle", False))
    warmup_ep  = int(train_cfg.get("warmup_epochs", 0))
    threshold  = float(config.get("multilabel", {}).get("threshold", 0.5))
    sched_name = train_cfg.get("lr_scheduler", "plateau")
    lr         = float(train_cfg.get("learning_rate", 2e-4))
    wd         = float(train_cfg.get("weight_decay", 1e-4))
    mil_name   = config["mil"]["model"]
    monitor    = train_cfg.get("monitor_metric", "macro_auc")
    n_labels   = len(label_names)

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True,
                              num_workers=nw,
                              collate_fn=multilabel_collate_fn)
    val_loader   = DataLoader(val_subset, batch_size=1, shuffle=False,
                              num_workers=nw,
                              collate_fn=multilabel_collate_fn)

    model, _ = build_multilabel_model(config)
    model.to(device)

    opt_name  = train_cfg.get("optimizer", "AdamW")
    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_ep, eta_min=1e-7)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, max_ep // 3), gamma=0.3)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=10)

    # Get raw dataset for pos_weight
    raw_ds = getattr(train_subset, "dataset", train_subset)
    loss_fn = _build_multilabel_loss(
        config,
        raw_ds if hasattr(raw_ds, "label_pos_weights") else None,
        device)

    do_es  = bool(train_cfg.get("early_stopping", True))
    es_pat = int(train_cfg.get("early_stopping_patience", 20))
    es_min = int(train_cfg.get("early_stopping_min_epochs", 10))

    best_val_metric = -float("inf")
    best_val_met    = {}
    es_counter      = 0
    best_ckpt_path  = os.path.join(fold_dir, "best_model.pt")

    for epoch in range(1, max_ep + 1):
        # LR warmup
        if warmup_ep > 0 and epoch <= warmup_ep:
            scale = epoch / max(1, warmup_ep)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * scale

        tr_loss, tr_met = _run_ml_epoch(
            model, train_loader, optimizer, loss_fn,
            n_labels, label_names, device, is_train=True,
            threshold=threshold,
            patch_dropout=patch_drop, patch_shuffle=patch_shuf)

        vl_loss, vl_met = _run_ml_epoch(
            model, val_loader, optimizer, loss_fn,
            n_labels, label_names, device, is_train=False,
            threshold=threshold)

        if sched_name == "plateau":
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        log.info(
            f"  [Fold {fold_idx}] Ep {epoch:>3}/{max_ep} "
            f"tr_loss={tr_loss:.4f} mauc={tr_met['macro_auc']:.3f} | "
            f"val_loss={vl_loss:.4f} mauc={vl_met['macro_auc']:.3f} "
            f"mf1={vl_met['macro_f1']:.3f}")

        cur_metric = vl_met.get(monitor, vl_met["macro_auc"])
        if cur_metric > best_val_metric:
            best_val_metric = cur_metric
            best_val_met    = {
                "epoch": epoch,
                **{f"val_{k}": v
                   for k, v in vl_met.items()
                   if not isinstance(v, dict)}
            }
            es_counter = 0
            _save_ml_checkpoint(
                best_ckpt_path, model, optimizer, scheduler,
                epoch=epoch, metrics=best_val_met,
                config=config, label_names=label_names,
                mil_name=mil_name, is_best=True)
        else:
            es_counter += 1
            if do_es and epoch >= es_min and es_counter >= es_pat:
                log.info(f"  [Fold {fold_idx}] Early stop at epoch {epoch}")
                break

    return best_val_met


# ─── Main command ─────────────────────────────────────────────────────────────────

def command_multilabel_crossval(config: dict, dirs_dict: dict, log=None):
    """
    Run stratified K-fold cross-validation for MIL multi-label classification.

    Results are saved to a unique, timestamped directory:
        results/multilabel/crossval/<task>/<model>__<feat_label>__<YYYYMMDD_HHMMSS>/
    """
    _log = log or logger

    if not _validate_ml_cv_config(config, _log):
        return

    cv_cfg     = config.get("multilabel_crossval", {})
    n_folds    = int(cv_cfg.get("n_folds", 5))
    seed       = int(cv_cfg.get("seed", 42))
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mil_name   = config["mil"]["model"]
    ml_cfg     = config.get("multilabel", {})
    task_name  = ml_cfg.get("task_name") or config.get("task", {}).get("name", "multilabel")
    label_names = ml_cfg.get("label_names", [])

    # ── Resolve feature dir ────────────────────────────────────────────────────
    feat_dir = dirs_dict.get("features")
    if not feat_dir:
        _log.error("No feature directory resolved. Pass --features or set config.")
        return
    pt_dir = os.path.join(feat_dir, "pt_files")
    if not os.path.isdir(pt_dir):
        _log.error(f"Feature pt_files not found: {pt_dir}")
        return

    # ── Build unique run directory ─────────────────────────────────────────────
    # Encodes: task / model / feature-dir-label / datetime
    feat_label = os.path.basename(feat_dir.rstrip("/\\"))
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"{mil_name}__{feat_label}__{timestamp}"
    cv_root    = os.path.join(
        config["paths"]["results_dir"],
        "multilabel", "crossval", task_name, run_name)
    os.makedirs(cv_root, exist_ok=True)

    _log.info("=" * 60)
    _log.info(f"  ML CROSS-VALIDATION")
    _log.info(f"  Task          : {task_name}")
    _log.info(f"  Model         : {mil_name}")
    _log.info(f"  Features      : {feat_label}")
    _log.info(f"  Labels        : {label_names}")
    _log.info(f"  Folds         : {n_folds}")
    _log.info(f"  Device        : {device}")
    _log.info(f"  Run dir       : {cv_root}")
    if _SKMULTILEARN:
        _log.info("  Stratification: IterativeStratification (scikit-multilearn)")
    else:
        _log.info("  Stratification: StratifiedKFold proxy (install scikit-multilearn for full support)")
    _log.info("=" * 60)

    # ── Save provenance metadata ───────────────────────────────────────────────
    exp_info = {
        "run_name"       : run_name,
        "task_name"      : task_name,
        "model"          : mil_name,
        "n_folds"        : n_folds,
        "label_names"    : label_names,
        "feature_dir"    : feat_dir,
        "feat_label"     : feat_label,
        "timestamp"      : timestamp,
        "device"         : str(device),
        "encoding_size"  : config["mil"].get("encoding_size"),
        "stratification" : "IterativeStratification" if _SKMULTILEARN else "StratifiedKFold(proxy)",
        "cv_root"        : cv_root,
    }
    with open(os.path.join(cv_root, "experiment_info.json"), "w") as f:
        json.dump(exp_info, f, indent=2)
    with open(os.path.join(cv_root, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Load + combine train/val splits ────────────────────────────────────────
    splits_dir  = os.path.join(config["paths"]["results_dir"],
                               "multilabel", "splits", task_name)

    dfs = []
    for split in ("train", "val"):
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        _log.error(
            f"No multilabel train/val CSV found in {splits_dir}.\n"
            "  Run: python main.py multilabel-split --config ... --csv labels.csv")
        return

    combined_df  = pd.concat(dfs, ignore_index=True)
    combined_csv = os.path.join(cv_root, "combined_pool.csv")
    combined_df.to_csv(combined_csv, index=False)

    max_patches = config.get("multilabel_training", {}).get("max_patches")
    full_dataset = MultiLabelMILDataset(
        csv_path       = combined_csv,
        pt_dir         = pt_dir,
        label_names    = label_names,
        binary_columns = ml_cfg.get("binary_columns", True),
        labels_col     = ml_cfg.get("labels_string_col"),
        max_patches    = max_patches,
        threshold      = float(ml_cfg.get("threshold", 0.5)),
    )

    n_valid = len(full_dataset)
    if n_valid == 0:
        _log.error("Full dataset has 0 valid slides. Check feature dir and label CSV.")
        return

    _log.info(f"  Total slides (train+val pool): {n_valid}")

    # ── Build label matrix for stratification ─────────────────────────────────
    label_matrix = np.stack(full_dataset.label_vecs)   # (N, n_labels)
    all_idxs     = np.arange(n_valid)

    # ── Iterative stratified split ────────────────────────────────────────────
    splits = _iterative_split(label_matrix, n_folds, seed)

    fold_results = []
    for fold_idx, (train_idxs, val_idxs) in enumerate(splits, start=1):
        _log.info(f"\n{'─'*50}")
        _log.info(f"  FOLD {fold_idx}/{n_folds} "
                  f"(train={len(train_idxs)}, val={len(val_idxs)})")

        fold_dir = os.path.join(cv_root, f"fold_{fold_idx:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        train_sub   = Subset(full_dataset, train_idxs)
        val_sub     = Subset(full_dataset, val_idxs)
        fold_config = copy.deepcopy(config)
        _seed(seed + fold_idx)

        fold_metrics = _train_cv_fold(
            fold_config, train_sub, val_sub,
            label_names, device, fold_dir, fold_idx, _log)

        fold_metrics["fold"] = fold_idx
        fold_results.append(fold_metrics)

        with open(os.path.join(fold_dir, "fold_metrics.json"), "w") as f:
            json.dump(fold_metrics, f, indent=2)

        _log.info(f"  Fold {fold_idx} best: " +
                  " | ".join(f"{k}={v:.4f}"
                              for k, v in fold_metrics.items()
                              if isinstance(v, float)))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    _log.info(f"\n{'='*60}")
    _log.info("  ML CROSS-VALIDATION RESULTS")

    metric_keys = [k for k in fold_results[0]
                   if isinstance(fold_results[0][k], float)]
    summary = {
        "run_name"      : run_name,
        "task_name"     : task_name,
        "model"         : mil_name,
        "feat_label"    : feat_label,
        "n_folds"       : n_folds,
        "label_names"   : label_names,
        "timestamp"     : timestamp,
        "stratification": "IterativeStratification" if _SKMULTILEARN else "StratifiedKFold(proxy)",
        "cv_root"       : cv_root,
    }
    for mk in metric_keys:
        vals = [r[mk] for r in fold_results if mk in r]
        summary[f"mean_{mk}"] = float(np.mean(vals))
        summary[f"std_{mk}"]  = float(np.std(vals))
        _log.info(f"  {mk:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    _log.info("=" * 60)

    with open(os.path.join(cv_root, "cv_summary.json"), "w") as f:
        json.dump({"summary": summary, "fold_results": fold_results}, f, indent=2)

    pd.DataFrame(fold_results).to_csv(
        os.path.join(cv_root, "cv_summary.csv"), index=False)

    _log.info(f"\n  CV results → {cv_root}")
    return cv_root
