"""
pipelines/train.py

MIL Training Pipeline.

Usage via CLI:
    python main.py train --config config/config.yaml

Outputs (in results/experiments/<task>/<model>_<timestamp>/):
    best_model.pt          - best checkpoint (weights + meta)
    final_model.pt         - final epoch checkpoint
    train_history.csv      - two rows per epoch: phase=train, phase=val
                             columns: epoch, phase, loss, accuracy, auc,
                                      precision, recall, f1, lr, epoch_time_s
    config_snapshot.yaml   - copy of the YAML config used
    plots/
        train_loss_curve.png
        val_auc_curve.png
        val_acc_curve.png
        learning_rate.png
"""

import os
import time
import logging
import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score,
                             balanced_accuracy_score)
import pandas as pd

from datasets.mil_dataset import build_mil_datasets, mil_collate_fn
from models.mil_models import build_mil_model, has_attention

logger = logging.getLogger(__name__)

# ─── History columns (matches reference schema) ───────────────────────────────
HISTORY_COLS = [
    'epoch', 'phase',
    'loss', 'accuracy', 'auc', 'precision', 'recall', 'f1',
    'lr', 'epoch_time_s',
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_metrics(all_probs, all_preds, all_labels, n_classes):
    """Return dict of scalar metrics."""
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds,
                    average='binary' if n_classes == 2 else 'macro',
                    zero_division=0)
    prec = precision_score(all_labels, all_preds,
                           average='binary' if n_classes == 2 else 'macro',
                           zero_division=0)
    rec  = recall_score(all_labels, all_preds,
                        average='binary' if n_classes == 2 else 'macro',
                        zero_division=0)
    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        auc = 0.0
    return dict(acc=acc, f1=f1, precision=prec, recall=rec, auc=auc)


def _build_loss_fn(config, class_names, device):
    """Build CrossEntropyLoss, optionally class-weighted and with label smoothing."""
    train_cfg       = config.get('training', {})
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    if train_cfg.get('weighted_loss', False):
        task_name = config['task']['name']
        results   = config['paths']['results_dir']
        train_csv = os.path.join(results, 'splits', task_name, 'train.csv')
        df  = pd.read_csv(train_csv)
        lbl = df.iloc[:, -1].astype(str)
        tot = len(lbl)
        w   = [tot / (len(class_names) * (lbl == c).sum()) for c in class_names]
        weight = torch.tensor(w, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _run_epoch(model, loader, optimizer, loss_fn,
               n_classes, device, is_train, use_clam, bag_weight,
               patch_dropout=0.0, patch_shuffle=False):
    """Run one epoch. Returns (avg_loss, metrics_dict)."""
    model.train(is_train)
    total_loss   = 0.
    all_probs    = []
    all_preds    = []
    all_labels   = []

    with torch.set_grad_enabled(is_train):
        for feats_list, labels, _ in loader:
            for feats, label in zip(feats_list, labels):
                feats = feats.to(device)
                label = label.unsqueeze(0).to(device)

                # ── Bag-level regularisation (training only) ──────────────────
                if is_train:
                    if patch_shuffle:
                        idx   = torch.randperm(feats.size(0), device=device)
                        feats = feats[idx]
                    if patch_dropout > 0 and feats.size(0) > 4:
                        keep  = max(4, int(feats.size(0) * (1 - patch_dropout)))
                        idx   = torch.randperm(feats.size(0), device=device)[:keep]
                        feats = feats[idx]

                if use_clam:
                    logits, Y_prob, Y_hat, _, extras = model(
                        feats, label=label, instance_eval=True)
                    bag_loss  = loss_fn(logits, label)
                    inst_loss = extras.get('instance_loss', 0.)
                    if torch.is_tensor(inst_loss):
                        loss = bag_weight * bag_loss + (1 - bag_weight) * inst_loss
                    else:
                        loss = bag_loss
                else:
                    logits, Y_prob, Y_hat, _, _ = model(feats)
                    loss = loss_fn(logits, label)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                all_probs.append(Y_prob.detach().cpu().numpy()[0])
                all_preds.append(Y_hat.detach().cpu().item())
                all_labels.append(label.cpu().item())

    avg_loss   = total_loss / max(len(all_labels), 1)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics    = _compute_metrics(all_probs, all_preds, all_labels, n_classes)
    return avg_loss, metrics


def _append_history_row(exp_dir, row):
    """Append one row to train_history.csv (create with header if new)."""
    record  = {c: row.get(c, float('nan')) for c in HISTORY_COLS}
    df_new  = pd.DataFrame([record])
    csv_path = os.path.join(exp_dir, 'train_history.csv')
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode='w', header=True, index=False)


def _save_checkpoint(path, model, optimizer, scheduler,
                     epoch, metrics, config, class_names, mil_name, is_best):
    """Save a rich checkpoint dict."""
    ckpt = {
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch':           epoch,
        'metrics':         metrics,
        'config':          config,
        'class_names':     class_names,
        'class_to_idx':    {c: i for i, c in enumerate(class_names)},
        'model_type':      mil_name,
        'timestamp':       datetime.datetime.now().isoformat(timespec='seconds'),
        'is_best':         is_best,
    }
    torch.save(ckpt, path)
    tag = 'BEST' if is_best else 'final'
    logger.info(f"  Checkpoint [{tag}] saved → {path}")


def _plot_training_curves(exp_dir):
    """Save training curve PNGs to exp_dir/plots/."""
    history_csv = os.path.join(exp_dir, 'train_history.csv')
    if not os.path.exists(history_csv):
        return []
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("  matplotlib not found — skipping training curve plots")
        return []

    df = pd.read_csv(history_csv)
    if df.empty:
        return []

    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_df = df[df['phase'] == 'train']
    val_df   = df[df['phase'] == 'val']
    saved    = []

    # Loss curve
    if 'loss' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        if not train_df.empty:
            ax.plot(train_df['epoch'], train_df['loss'],
                    label='Train loss', lw=1.5, color='steelblue')
        if not val_df.empty:
            ax.plot(val_df['epoch'], val_df['loss'],
                    label='Val loss', lw=1.5, color='tomato', linestyle='--')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(plots_dir, 'train_loss_curve.png')
        fig.savefig(out, dpi=120); plt.close(fig)
        saved.append(out)

    # Val AUC
    if 'auc' in val_df.columns and not val_df['auc'].isna().all():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(val_df['epoch'], val_df['auc'],
                label='Val AUC', lw=1.5, color='mediumseagreen')
        ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
        ax.set_title('Validation AUC'); ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(plots_dir, 'val_auc_curve.png')
        fig.savefig(out, dpi=120); plt.close(fig)
        saved.append(out)

    # Val Accuracy
    if 'accuracy' in val_df.columns and not val_df['accuracy'].isna().all():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(val_df['epoch'], val_df['accuracy'],
                label='Val Accuracy', lw=1.5, color='darkorchid')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy'); ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(plots_dir, 'val_acc_curve.png')
        fig.savefig(out, dpi=120); plt.close(fig)
        saved.append(out)

    # Learning rate
    if 'lr' in train_df.columns and not train_df['lr'].isna().all():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(train_df['epoch'], train_df['lr'],
                    lw=1.5, color='goldenrod')
        ax.set_xlabel('Epoch'); ax.set_ylabel('LR (log scale)')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        out = os.path.join(plots_dir, 'learning_rate.png')
        fig.savefig(out, dpi=120); plt.close(fig)
        saved.append(out)

    if saved:
        logger.info(f"  Training curves → {plots_dir}/ ({len(saved)} plots)")
    return saved


# ─── Main command ──────────────────────────────────────────────────────────────

def command_train(config: dict, dirs_dict: dict, log=None):
    _log = log or logger
    _seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.info(f"Training device: {device}")

    # ── Datasets ───────────────────────────────────────────────────────────────
    datasets, class_names = build_mil_datasets(config, dirs_dict)
    if 'train' not in datasets:
        _log.error("No train.csv split found. Run: python main.py split --config ...")
        return

    train_cfg = config.get('training', {})
    nw        = int(train_cfg.get('num_workers', 0))

    # Guard: raise a clear error if the training dataset is empty (0 valid bags).
    # This almost always means features haven't been extracted yet, or the wrong
    # feature subfolder is selected.
    if len(datasets['train']) == 0:
        feat_dir = dirs_dict.get('features', '<unknown>')
        _log.error(
            "Training dataset has 0 valid bags. No .pt feature files were found."
            f"\n  Expected features in : {feat_dir}/pt_files/"
            f"\n  To fix, either:"
            f"\n    1. Run feature extraction first:  python main.py extract --config ..."
            f"\n    2. Point to an existing feature set (model name auto-appended):"
            f"\n       python main.py train --features patch512_step512_level0"
            f"\n       or set feature_extraction.features_subfolder_override in config.yaml")
        return

    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=True,
                              num_workers=nw, collate_fn=mil_collate_fn)
    val_loader   = None
    if 'val' in datasets:
        if len(datasets['val']) == 0:
            _log.warning("Validation dataset has 0 valid bags — skipping validation.")
        else:
            val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False,
                                    num_workers=nw, collate_fn=mil_collate_fn)

    # ── Model ──────────────────────────────────────────────────────────────────
    model, n_params = build_mil_model(config)
    model.to(device)
    use_clam   = config['mil']['model'].startswith('clam')
    n_classes  = config['task'].get('num_classes', 2)
    bag_weight = float(config['mil'].get('bag_weight', 0.7))
    mil_name   = config['mil']['model']

    _log.info(f"Model: {mil_name} | Params: {n_params:,} | Classes: {n_classes}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    lr           = float(train_cfg.get('learning_rate',   2e-4))
    wd           = float(train_cfg.get('weight_decay',    1e-4))
    b1           = float(train_cfg.get('beta1',           0.9))
    b2           = float(train_cfg.get('beta2',           0.999))
    opt_name     = train_cfg.get('optimizer', 'AdamW')
    warmup_ep    = int(train_cfg.get('warmup_epochs', 0))
    sched_name   = train_cfg.get('lr_scheduler', 'plateau')
    patch_dropout = float(train_cfg.get('patch_dropout', 0.0))
    patch_shuffle = bool(train_cfg.get('patch_shuffle', False))

    if opt_name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))
    else:  # AdamW (default)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))

    if sched_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_ep, eta_min=1e-7)
    elif sched_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, max_ep // 3), gamma=0.3)
    else:  # plateau (default)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            factor   = float(train_cfg.get('lr_scheduler_factor', 0.5)),
            patience = int(train_cfg.get('lr_scheduler_patience', 10)))

    loss_fn = _build_loss_fn(config, class_names, device)

    if warmup_ep > 0:
        _log.info(f"  LR warmup: {warmup_ep} epochs")
    if patch_dropout > 0:
        _log.info(f"  Patch dropout: {patch_dropout:.2f}")
    if patch_shuffle:
        _log.info("  Patch shuffle: enabled")

    # ── Early stopping ─────────────────────────────────────────────────────────
    do_es  = bool(train_cfg.get('early_stopping', True))
    es_pat = int(train_cfg.get('early_stopping_patience', 20))
    es_min = int(train_cfg.get('early_stopping_min_epochs', 10))
    max_ep = int(train_cfg.get('max_epochs', 100))

    best_val_loss = float('inf')
    es_counter    = 0

    # ── Experiment directory ───────────────────────────────────────────────────
    stamp     = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_name = config['task']['name']
    exp_dir   = os.path.join(config['paths']['results_dir'],
                             'experiments', task_name,
                             f"{mil_name}_{stamp}")
    os.makedirs(exp_dir, exist_ok=True)
    _log.info(f"Experiment directory: {exp_dir}")

    # Save config snapshot
    with open(os.path.join(exp_dir, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    best_model_path  = os.path.join(exp_dir, 'best_model.pt')
    best_epoch_metrics = {}

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, max_ep + 1):
        t0 = time.time()

        # LR warmup
        if warmup_ep > 0 and epoch <= warmup_ep:
            scale = epoch / max(1, warmup_ep)
            for pg in optimizer.param_groups:
                pg['lr'] = lr * scale

        tr_loss, tr_met = _run_epoch(model, train_loader, optimizer, loss_fn,
                                     n_classes, device, is_train=True,
                                     use_clam=use_clam, bag_weight=bag_weight,
                                     patch_dropout=patch_dropout,
                                     patch_shuffle=patch_shuffle)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        # ── Write train row ────────────────────────────────────────────────────
        _append_history_row(exp_dir, {
            'epoch':        epoch,
            'phase':        'train',
            'loss':         tr_loss,
            'accuracy':     tr_met['acc'],
            'auc':          tr_met['auc'],
            'precision':    tr_met['precision'],
            'recall':       tr_met['recall'],
            'f1':           tr_met['f1'],
            'lr':           current_lr,
            'epoch_time_s': round(elapsed, 2),
        })

        # ── Validation ─────────────────────────────────────────────────────────
        if val_loader is not None:
            vl_loss, vl_met = _run_epoch(model, val_loader, optimizer, loss_fn,
                                         n_classes, device, is_train=False,
                                         use_clam=use_clam, bag_weight=bag_weight)
            # Step scheduler
            if sched_name == 'plateau':
                scheduler.step(vl_loss)
            else:
                scheduler.step()
            monitor_loss = vl_loss

            _append_history_row(exp_dir, {
                'epoch':     epoch,
                'phase':     'val',
                'loss':      vl_loss,
                'accuracy':  vl_met['acc'],
                'auc':       vl_met['auc'],
                'precision': vl_met['precision'],
                'recall':    vl_met['recall'],
                'f1':        vl_met['f1'],
                'lr':        current_lr,
            })

            _log.info(
                f"Epoch {epoch:>3}/{max_ep} | "
                f"tr_loss={tr_loss:.4f} acc={tr_met['acc']:.3f} auc={tr_met['auc']:.3f} | "
                f"val_loss={vl_loss:.4f} acc={vl_met['acc']:.3f} auc={vl_met['auc']:.3f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s")
        else:
            if sched_name == 'plateau':
                scheduler.step(tr_loss)
            else:
                scheduler.step()
            monitor_loss = tr_loss
            vl_met = tr_met
            vl_loss = tr_loss
            _log.info(
                f"Epoch {epoch:>3}/{max_ep} | "
                f"tr_loss={tr_loss:.4f} acc={tr_met['acc']:.3f} auc={tr_met['auc']:.3f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s")

        # ── Best checkpoint ────────────────────────────────────────────────────
        if monitor_loss < best_val_loss:
            best_val_loss      = monitor_loss
            best_epoch_metrics = {'epoch': epoch, **{f'val_{k}': v
                                  for k, v in vl_met.items()}}
            es_counter = 0
            _save_checkpoint(best_model_path, model, optimizer, scheduler,
                             epoch=epoch,
                             metrics=best_epoch_metrics,
                             config=config,
                             class_names=class_names,
                             mil_name=mil_name,
                             is_best=True)
        else:
            es_counter += 1
            if do_es and epoch >= es_min and es_counter >= es_pat:
                _log.info(f"Early stopping at epoch {epoch} (patience={es_pat})")
                break

    # ── Final checkpoint ───────────────────────────────────────────────────────
    final_path = os.path.join(exp_dir, 'final_model.pt')
    _save_checkpoint(final_path, model, optimizer, scheduler,
                     epoch=epoch,
                     metrics={'epoch': epoch, **{f'val_{k}': v
                              for k, v in vl_met.items()}},
                     config=config,
                     class_names=class_names,
                     mil_name=mil_name,
                     is_best=False)

    # ── Training curve plots ───────────────────────────────────────────────────
    _plot_training_curves(exp_dir)

    _log.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    _log.info(f"Best model : {best_model_path}")
    _log.info(f"Final model: {final_path}")
    _log.info(f"History    : {os.path.join(exp_dir, 'train_history.csv')}")
    _log.info(f"Experiment : {exp_dir}")

    return exp_dir
