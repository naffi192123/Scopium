"""
pipelines/train.py

MIL Training Pipeline.

Usage via CLI:
    python main.py train --config config/config.yaml

Outputs (in results/experiments/<task>/<model>_<timestamp>/):
    best_model.pt          - best checkpoint (dict with weights + meta)
    final_model.pt         - final epoch checkpoint
    training_history.csv   - per-epoch: loss/acc/auc/f1/lr/time for train+val
    config_snapshot.yaml   - copy of the YAML config used
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


# ─── Helpers ────────────────────────────────────────────────────────────────

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
    bacc = balanced_accuracy_score(all_labels, all_preds)
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
    return dict(acc=acc, bal_acc=bacc, f1=f1, precision=prec,
                recall=rec, auc=auc)


def _build_loss_fn(config, class_names, device):
    """Build CrossEntropyLoss, optionally class-weighted."""
    if config.get('training', {}).get('weighted_loss', False):
        # inverse frequency weights (requires train CSV)
        task_name = config['task']['name']
        results   = config['paths']['results_dir']
        train_csv = os.path.join(results, 'splits', task_name, 'train.csv')
        df  = pd.read_csv(train_csv)
        lbl = df.iloc[:, -1].astype(str)
        tot = len(lbl)
        w   = [tot / (len(class_names) * (lbl == c).sum()) for c in class_names]
        weight = torch.tensor(w, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=weight)
    return nn.CrossEntropyLoss()


def _run_epoch(model, loader, optimizer, loss_fn,
               n_classes, device, is_train, use_clam, bag_weight):
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

    avg_loss = total_loss / max(len(all_labels), 1)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics    = _compute_metrics(all_probs, all_preds, all_labels, n_classes)
    return avg_loss, metrics


# ─── Main command ────────────────────────────────────────────────────────────

def command_train(config: dict, dirs_dict: dict, log=None):
    _log = log or logger
    _seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.info(f"Training device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    datasets, class_names = build_mil_datasets(config, dirs_dict)
    if 'train' not in datasets:
        _log.error("No train.csv split found. Run: python main.py split --config ...")
        return

    train_cfg = config.get('training', {})
    nw        = int(train_cfg.get('num_workers', 0))

    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=True,
                              num_workers=nw, collate_fn=mil_collate_fn)
    val_loader   = None
    if 'val' in datasets:
        val_loader = DataLoader(datasets['val'], batch_size=1, shuffle=False,
                                num_workers=nw, collate_fn=mil_collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, n_params = build_mil_model(config)
    model.to(device)
    use_clam   = config['mil']['model'].startswith('clam')
    n_classes  = config['task'].get('num_classes', 2)
    bag_weight = float(config['mil'].get('bag_weight', 0.7))

    _log.info(f"Model: {config['mil']['model']} | Params: {n_params:,} | Classes: {n_classes}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    lr      = float(train_cfg.get('learning_rate',   2e-4))
    wd      = float(train_cfg.get('weight_decay',    1e-4))
    b1      = float(train_cfg.get('beta1',           0.9))
    b2      = float(train_cfg.get('beta2',           0.999))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=wd, betas=(b1, b2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min',
        factor   = float(train_cfg.get('lr_scheduler_factor', 0.5)),
        patience = int(train_cfg.get('lr_scheduler_patience', 10)),
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn   = _build_loss_fn(config, class_names, device)

    # ── Early stopping ────────────────────────────────────────────────────────
    do_es     = bool(train_cfg.get('early_stopping', True))
    es_pat    = int(train_cfg.get('early_stopping_patience', 20))
    es_min    = int(train_cfg.get('early_stopping_min_epochs', 10))
    max_ep    = int(train_cfg.get('max_epochs', 100))

    best_val_loss = float('inf')
    es_counter    = 0

    # ── Experiment directory ──────────────────────────────────────────────────
    stamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_name = config['task']['name']
    mil_name  = config['mil']['model']
    exp_dir   = os.path.join(config['paths']['results_dir'],
                             'experiments', task_name,
                             f"{mil_name}_{stamp}")
    os.makedirs(exp_dir, exist_ok=True)
    _log.info(f"Experiment directory: {exp_dir}")

    # Save config snapshot
    with open(os.path.join(exp_dir, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    history_rows = []
    best_model_path = os.path.join(exp_dir, 'best_model.pt')

    for epoch in range(1, max_ep + 1):
        t0 = time.time()

        tr_loss, tr_met = _run_epoch(model, train_loader, optimizer, loss_fn,
                                     n_classes, device, is_train=True,
                                     use_clam=use_clam, bag_weight=bag_weight)
        elapsed = time.time() - t0

        row = {
            'epoch': epoch, 'time_s': round(elapsed, 1),
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': tr_loss,
            'train_acc': tr_met['acc'], 'train_auc': tr_met['auc'],
            'train_f1': tr_met['f1'], 'train_bacc': tr_met['bal_acc'],
        }

        if val_loader is not None:
            vl_loss, vl_met = _run_epoch(model, val_loader, optimizer, loss_fn,
                                         n_classes, device, is_train=False,
                                         use_clam=use_clam, bag_weight=bag_weight)
            scheduler.step(vl_loss)
            row.update({
                'val_loss':  vl_loss,
                'val_acc':   vl_met['acc'], 'val_auc':  vl_met['auc'],
                'val_f1':    vl_met['f1'],  'val_bacc': vl_met['bal_acc'],
            })
            monitor_loss = vl_loss
            _log.info(
                f"Epoch {epoch:>3}/{max_ep} | "
                f"tr_loss={tr_loss:.4f} acc={tr_met['acc']:.3f} auc={tr_met['auc']:.3f} | "
                f"val_loss={vl_loss:.4f} acc={vl_met['acc']:.3f} auc={vl_met['auc']:.3f} | "
                f"lr={row['lr']:.2e} | {elapsed:.1f}s")
        else:
            scheduler.step(tr_loss)
            monitor_loss = tr_loss
            _log.info(
                f"Epoch {epoch:>3}/{max_ep} | "
                f"tr_loss={tr_loss:.4f} acc={tr_met['acc']:.3f} auc={tr_met['auc']:.3f} | "
                f"lr={row['lr']:.2e} | {elapsed:.1f}s")

        history_rows.append(row)

        # Save best model
        if monitor_loss < best_val_loss:
            best_val_loss = monitor_loss
            es_counter    = 0
            ckpt = {
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'optim_state':   optimizer.state_dict(),
                'sched_state':   scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics':       row,
                'config':        config,
                'class_names':   class_names,
                'class_to_idx':  {c: i for i, c in enumerate(class_names)},
                'model_type':    mil_name,
                'timestamp':     stamp,
            }
            torch.save(ckpt, best_model_path)
            _log.info(f"  -> best model saved (val_loss={best_val_loss:.4f})")
        else:
            es_counter += 1
            if do_es and epoch >= es_min and es_counter >= es_pat:
                _log.info(f"Early stopping at epoch {epoch} (patience={es_pat})")
                break

        # Save history incrementally
        pd.DataFrame(history_rows).to_csv(
            os.path.join(exp_dir, 'training_history.csv'), index=False)

    # Save final model
    final_path = os.path.join(exp_dir, 'final_model.pt')
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'config':      config,
        'class_names': class_names,
        'model_type':  mil_name,
        'timestamp':   stamp,
    }, final_path)

    _log.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    _log.info(f"Best model : {best_model_path}")
    _log.info(f"Final model: {final_path}")
    _log.info(f"History    : {os.path.join(exp_dir, 'training_history.csv')}")
    _log.info(f"Experiment : {exp_dir}")

    return exp_dir
