"""
pipelines/crossval.py

Dedicated K-Fold Cross-Validation pipeline for MIL.

Runs stratified k-fold cross-validation either:
  (a) with default config  — to evaluate a chosen model/hyperparams
  (b) with best HPO config — via --use_best_config flag

For each fold:
  - Trains for max_epochs with early stopping
  - Evaluates on the held-out fold
  - Saves per-fold checkpoint + metrics

Aggregates results across folds and saves a summary CSV.

CLI:
    # Standard CV
    python main.py crossval --config config/config.yaml

    # CV with best HPO config
    python main.py crossval --config config/config.yaml --use_best_config

Output structure:
    results/crossval/<study_name>/
        fold_<k>/
            best_model.pt
            fold_metrics.json
        cv_summary.json       — per-fold + mean/std metrics
        cv_summary.csv
"""

import os
import copy
import json
import logging
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from datasets.mil_dataset import MILBagDataset, mil_collate_fn
from models.mil_models import build_mil_model
from pipelines.train import (
    _run_epoch, _build_loss_fn, _compute_metrics, _save_checkpoint
)

logger = logging.getLogger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer_cv(config: dict, model: nn.Module):
    train_cfg  = config.get('training', {})
    opt_name   = train_cfg.get('optimizer', 'AdamW')
    lr  = float(train_cfg.get('learning_rate', 2e-4))
    wd  = float(train_cfg.get('weight_decay', 1e-4))
    b1  = float(train_cfg.get('beta1', 0.9))
    b2  = float(train_cfg.get('beta2', 0.999))
    if opt_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))


def _build_scheduler_cv(config: dict, optimizer, n_epochs: int):
    train_cfg  = config.get('training', {})
    sched_name = train_cfg.get('lr_scheduler', 'plateau')
    if sched_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-7)
    elif sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, n_epochs // 3), gamma=0.3)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        factor   = float(train_cfg.get('lr_scheduler_factor', 0.5)),
        patience = int(train_cfg.get('lr_scheduler_patience', 10)))


def _validate_config(config: dict, log) -> bool:
    """Check all required keys exist before training starts."""
    required = {
        'mil':      ['model', 'encoding_size'],
        'training': ['learning_rate', 'weight_decay', 'max_epochs'],
        'task':     ['name', 'num_classes', 'class_names'],
        'paths':    ['results_dir'],
    }
    ok = True
    for section, keys in required.items():
        for key in keys:
            if config.get(section, {}).get(key) is None:
                log.error(f"Config missing required key: {section}.{key}")
                ok = False
    return ok


# ─── Per-fold training loop ────────────────────────────────────────────────────

def _train_fold(config: dict, train_dataset, val_dataset,
                class_names: list, device: torch.device,
                fold_dir: str, fold_idx: int, log):
    """Train one fold. Returns val metrics dict."""
    train_cfg = config.get('training', {})
    nw        = int(train_cfg.get('num_workers', 0))
    max_ep    = int(train_cfg.get('max_epochs', 100))
    patch_dropout = float(train_cfg.get('patch_dropout', 0.0))
    patch_shuffle = bool(train_cfg.get('patch_shuffle', False))
    warmup_ep     = int(train_cfg.get('warmup_epochs', 0))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=nw, collate_fn=mil_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=nw, collate_fn=mil_collate_fn)

    model, n_params = build_mil_model(config)
    model.to(device)
    use_clam   = config['mil']['model'].startswith('clam')
    n_classes  = config['task'].get('num_classes', 2)
    bag_weight = float(config['mil'].get('bag_weight', 0.7))
    mil_name   = config['mil']['model']
    sched_name = train_cfg.get('lr_scheduler', 'plateau')
    lr         = float(train_cfg.get('learning_rate', 2e-4))

    optimizer  = _build_optimizer_cv(config, model)
    scheduler  = _build_scheduler_cv(config, optimizer, max_ep)
    loss_fn    = _build_loss_fn(config, class_names, device)

    do_es  = bool(train_cfg.get('early_stopping', True))
    es_pat = int(train_cfg.get('early_stopping_patience', 20))
    es_min = int(train_cfg.get('early_stopping_min_epochs', 10))

    best_val_loss  = float('inf')
    best_val_met   = {}
    es_counter     = 0
    best_ckpt_path = os.path.join(fold_dir, 'best_model.pt')

    for epoch in range(1, max_ep + 1):
        # LR warmup
        if warmup_ep > 0 and epoch <= warmup_ep:
            scale = epoch / max(1, warmup_ep)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * scale

        tr_loss, tr_met = _run_epoch(
            model, train_loader, optimizer, loss_fn,
            n_classes, device, is_train=True,
            use_clam=use_clam, bag_weight=bag_weight,
            patch_dropout=patch_dropout, patch_shuffle=patch_shuffle)

        vl_loss, vl_met = _run_epoch(
            model, val_loader, optimizer, loss_fn,
            n_classes, device, is_train=False,
            use_clam=use_clam, bag_weight=bag_weight)

        if sched_name == 'plateau':
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        log.info(
            f"  [Fold {fold_idx}] Epoch {epoch:>3}/{max_ep} "
            f"tr_loss={tr_loss:.4f} auc={tr_met['auc']:.3f} | "
            f"val_loss={vl_loss:.4f} auc={vl_met['auc']:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_met  = {'epoch': epoch, **{f'val_{k}': v for k, v in vl_met.items()}}
            es_counter    = 0
            _save_checkpoint(
                best_ckpt_path, model, optimizer, scheduler,
                epoch=epoch, metrics=best_val_met,
                config=config, class_names=class_names,
                mil_name=mil_name, is_best=True)
        else:
            es_counter += 1
            if do_es and epoch >= es_min and es_counter >= es_pat:
                log.info(f"  [Fold {fold_idx}] Early stop at epoch {epoch}")
                break

    return best_val_met


# ─── Main command ───────────────────────────────────────────────────────────────

def command_crossval(config: dict, dirs_dict: dict, log=None):
    """
    Run stratified k-fold cross-validation for MIL.
    Saves per-fold metrics and aggregate summary.
    """
    _log = log or logger
    _seed(42)

    if not _validate_config(config, _log):
        return

    cv_cfg     = config.get('crossval', {})
    n_folds    = int(cv_cfg.get('n_folds', 5))
    study_name = cv_cfg.get('study_name', 'crossval')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _log.info("=" * 60)
    _log.info(f"  CROSS-VALIDATION: {study_name}")
    _log.info(f"  Folds    : {n_folds}")
    _log.info(f"  Model    : {config['mil']['model']}")
    _log.info(f"  Device   : {device}")
    _log.info("=" * 60)

    # ── Resolve feature dir ────────────────────────────────────────────────────
    feat_dir   = dirs_dict.get('features')
    if feat_dir:
        pt_dir = os.path.join(feat_dir, 'pt_files')
    else:
        _log.error("No feature directory resolved. Pass --features or set config.")
        return

    if not os.path.isdir(pt_dir):
        _log.error(f"Feature pt_files directory not found: {pt_dir}")
        return

    # ── Load full dataset (train + val combined for CV) ─────────────────────
    task_name   = config['task']['name']
    class_names = config['task']['class_names']
    splits_dir  = os.path.join(config['paths']['results_dir'], 'splits', task_name)

    # Combine train + val splits into one pool for stratified CV
    dfs = []
    for split in ('train', 'val'):
        csv_path = os.path.join(splits_dir, f'{split}.csv')
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        _log.error(f"No train/val split CSV found in {splits_dir}")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    # Detect label column
    cols    = [c.lower() for c in combined_df.columns]
    lbl_col = next((c for c in cols if c in ('label', 'subtype', 'diagnosis', 'class')), cols[-1])
    # Map label strings to indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels_all = combined_df.iloc[:, cols.index(lbl_col)].astype(str).map(
        lambda x: class_to_idx.get(x, 0)).values

    # Build full dataset
    # Write combined CSV temporarily so MILBagDataset can read it
    cv_root = os.path.join(config['paths']['results_dir'], 'crossval', study_name)
    os.makedirs(cv_root, exist_ok=True)
    combined_csv = os.path.join(cv_root, 'combined_pool.csv')
    combined_df.to_csv(combined_csv, index=False)

    max_patches = config.get('training', {}).get('max_patches')
    full_dataset = MILBagDataset(
        csv_path    = combined_csv,
        pt_dir      = pt_dir,
        class_names = class_names,
        max_patches = max_patches,
    )

    n_valid = len(full_dataset)
    if n_valid == 0:
        _log.error("Full dataset has 0 valid slides. Check feature directory.")
        return

    _log.info(f"  Total slides (train+val pool): {n_valid}")

    # ── Stratified K-Fold ──────────────────────────────────────────────────────
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_idxs  = np.arange(n_valid)

    # For stratification: get labels from the dataset
    slide_labels = np.array([full_dataset[i][1] for i in range(n_valid)])

    fold_results = []

    for fold_idx, (train_idxs, val_idxs) in enumerate(
            skf.split(all_idxs, slide_labels), start=1):

        _log.info(f"\n{'─'*50}")
        _log.info(f"  FOLD {fold_idx}/{n_folds}  "
                  f"(train={len(train_idxs)}, val={len(val_idxs)})")

        fold_dir = os.path.join(cv_root, f'fold_{fold_idx:02d}')
        os.makedirs(fold_dir, exist_ok=True)

        train_sub = Subset(full_dataset, train_idxs)
        val_sub   = Subset(full_dataset, val_idxs)

        fold_config = copy.deepcopy(config)  # isolate per-fold config
        _seed(42 + fold_idx)

        fold_metrics = _train_fold(
            fold_config, train_sub, val_sub,
            class_names, device, fold_dir, fold_idx, _log)

        fold_metrics['fold'] = fold_idx
        fold_results.append(fold_metrics)

        # Save fold metrics
        with open(os.path.join(fold_dir, 'fold_metrics.json'), 'w') as f:
            json.dump(fold_metrics, f, indent=2)

        _log.info(f"  Fold {fold_idx} best: "
                  + " | ".join(f"{k}={v:.4f}" for k, v in fold_metrics.items()
                                if isinstance(v, float)))

    # ── Aggregate ──────────────────────────────────────────────────────────────
    _log.info(f"\n{'='*60}")
    _log.info("  CROSS-VALIDATION RESULTS")

    metric_keys = [k for k in fold_results[0] if isinstance(fold_results[0][k], float)]
    summary = {'n_folds': n_folds, 'model': config['mil']['model'],
               'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
    for mk in metric_keys:
        vals = [r[mk] for r in fold_results if mk in r]
        summary[f'mean_{mk}'] = float(np.mean(vals))
        summary[f'std_{mk}']  = float(np.std(vals))
        _log.info(f"  {mk:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    _log.info("=" * 60)

    # Save summary
    with open(os.path.join(cv_root, 'cv_summary.json'), 'w') as f:
        json.dump({'summary': summary, 'fold_results': fold_results}, f, indent=2)

    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(cv_root, 'cv_summary.csv'), index=False)

    _log.info(f"\n  CV results → {cv_root}")
    return cv_root
