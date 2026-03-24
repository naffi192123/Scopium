"""
pipelines/hpo.py

Hyperparameter Optimisation (HPO) Pipeline for MIL.

Uses Optuna to search over the MIL model architecture and training
hyperparameters. Each trial performs a fast train + validate loop and
reports validation AUC (or another configured metric). The best
configuration is saved as best_config.yaml for use by command_train.

CLI:
    python main.py hpo --config config/config.yaml

After HPO completes, train the final model with the best config:
    python main.py train --config config/config.yaml --use_best_config

Output directory structure:
    results/hpo/<study_name>/
        trial_<n>/
            trial_config.yaml      — merged config for this trial
            trial_metrics.json     — val AUC / acc / f1 for each epoch
        best_config.yaml           — config for the best trial
        best_trial.json            — summary of the best trial
        hpo_results.csv            — all trial results sorted by metric
        study.db                   — Optuna SQLite storage (for resuming)
"""

import os
import copy
import json
import math
import time
import logging
import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from datasets.mil_dataset import build_mil_datasets, mil_collate_fn
from models.mil_models import build_mil_model, _MODEL_REGISTRY

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(trial_cfg: dict, model: nn.Module):
    """Build optimizer from trial config."""
    train_cfg  = trial_cfg.get('training', {})
    optimizer_name = train_cfg.get('optimizer', 'AdamW')
    lr  = float(train_cfg.get('learning_rate', 2e-4))
    wd  = float(train_cfg.get('weight_decay', 1e-4))
    b1  = float(train_cfg.get('beta1', 0.9))
    b2  = float(train_cfg.get('beta2', 0.999))

    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=wd, betas=(b1, b2))
    else:
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=wd, betas=(b1, b2))


def _build_scheduler(trial_cfg: dict, optimizer, n_epochs: int):
    """Build LR scheduler from trial config."""
    train_cfg      = trial_cfg.get('training', {})
    scheduler_name = train_cfg.get('lr_scheduler', 'plateau')

    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-7)
    elif scheduler_name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, n_epochs // 3), gamma=0.3)
    else:  # plateau (default)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            factor   = float(train_cfg.get('lr_scheduler_factor', 0.5)),
            patience = int(train_cfg.get('lr_scheduler_patience', 5)))


def _warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    """Linear LR warmup for first `warmup_epochs` epochs."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        scale = epoch / max(1, warmup_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * scale


def _build_loss_fn(trial_cfg: dict, class_names: list, device):
    train_cfg      = trial_cfg.get('training', {})
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    if train_cfg.get('weighted_loss', False):
        task_name = trial_cfg['task']['name']
        results   = trial_cfg['paths']['results_dir']
        import pandas as pd
        train_csv = os.path.join(results, 'splits', task_name, 'train.csv')
        df  = pd.read_csv(train_csv)
        lbl = df.iloc[:, -1].astype(str)
        tot = len(lbl)
        w   = [tot / (len(class_names) * (lbl == c).sum()) for c in class_names]
        weight = torch.tensor(w, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _run_epoch_hpo(model, loader, optimizer, loss_fn,
                   n_classes, device, is_train,
                   use_clam, bag_weight,
                   patch_dropout=0.0, patch_shuffle=False):
    """One epoch of training or validation within an HPO trial."""
    model.train(is_train)
    total_loss  = 0.
    all_probs   = []
    all_preds   = []
    all_labels  = []

    with torch.set_grad_enabled(is_train):
        for feats_list, labels, _ in loader:
            for feats, label in zip(feats_list, labels):
                feats = feats.to(device)
                label = label.unsqueeze(0).to(device)

                # ── Bag-level regularisation ──────────────────────────────────
                if is_train:
                    if patch_shuffle:
                        idx   = torch.randperm(feats.size(0), device=device)
                        feats = feats[idx]
                    if patch_dropout > 0 and feats.size(0) > 4:
                        keep = max(4, int(feats.size(0) * (1 - patch_dropout)))
                        idx  = torch.randperm(feats.size(0), device=device)[:keep]
                        feats = feats[idx]

                # ── Forward pass ──────────────────────────────────────────────
                if use_clam:
                    logits, Y_prob, Y_hat, _, extras = model(
                        feats, label=label, instance_eval=True)
                    bag_loss  = loss_fn(logits, label)
                    inst_loss = extras.get('instance_loss', 0.)
                    loss = (bag_weight * bag_loss + (1 - bag_weight) * inst_loss
                            if torch.is_tensor(inst_loss) else bag_loss)
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

    if not all_labels:
        return 0., {'auc': 0., 'acc': 0., 'f1': 0.}

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / len(all_labels)

    acc = float(accuracy_score(all_labels, all_preds))
    f1  = float(f1_score(all_labels, all_preds,
                         average='binary' if n_classes == 2 else 'macro',
                         zero_division=0))
    try:
        auc = float(roc_auc_score(all_labels, all_probs[:, 1])
                    if n_classes == 2
                    else roc_auc_score(all_labels, all_probs, multi_class='ovr'))
    except Exception:
        auc = 0.0

    return avg_loss, {'auc': auc, 'acc': acc, 'f1': f1}


# ─── Search-space suggestion ───────────────────────────────────────────────────

def _suggest_hyperparams(trial, hpo_cfg: dict, base_config: dict) -> dict:
    """Use Optuna trial to suggest all hyperparameters. Returns config overrides."""
    ss = hpo_cfg.get('search_space', {})

    def _cat(key, default):
        choices = ss.get(key)
        return trial.suggest_categorical(key, choices) if choices else default

    def _loguni(key, default_lo, default_hi):
        rng = ss.get(key, [default_lo, default_hi])
        return trial.suggest_float(key, rng[0], rng[1], log=True)

    def _uni(key, default_lo, default_hi):
        rng = ss.get(key, [default_lo, default_hi])
        return trial.suggest_float(key, rng[0], rng[1])

    model_key      = _cat('model', 'abmil')
    optimizer_name = _cat('optimizer', 'AdamW')
    lr             = _loguni('learning_rate', 1e-4, 2e-3)
    wd             = _loguni('weight_decay', 1e-5, 5e-4)
    dropout        = _uni('dropout', 0.1, 0.6)
    dropout_attn   = _uni('dropout_attn', 0.2, 0.4)
    dropout_clf    = _uni('dropout_classifier', 0.1, 0.3)
    hidden_dim     = _cat('attn_hidden_dim', 256)
    proj_dim       = _cat('feature_proj_dim', 512)
    scheduler_name = _cat('lr_scheduler', 'plateau')
    label_smoothing = _uni('label_smoothing', 0.0, 0.15)
    es_patience    = _cat('early_stop_patience', 15)
    warmup_epochs  = _cat('warmup_epochs', 0)
    patch_dropout  = _uni('patch_dropout', 0.0, 0.3)
    patch_shuffle  = True   # always on (no meaningful trade-off to tune)
    max_patches    = _cat('max_patches', None)

    return {
        'mil': {
            'model':            model_key,
            'hidden_dim':       hidden_dim,
            'dropout':          dropout,
            'dropout_attn':     dropout_attn,
            'dropout_classifier': dropout_clf,
            'feature_proj_dim': proj_dim,
        },
        'training': {
            'optimizer':              optimizer_name,
            'learning_rate':          lr,
            'weight_decay':           wd,
            'lr_scheduler':           scheduler_name,
            'label_smoothing':        label_smoothing,
            'early_stopping_patience': es_patience,
            'warmup_epochs':          warmup_epochs,
            'patch_dropout':          patch_dropout,
            'patch_shuffle':          patch_shuffle,
            'max_patches':            max_patches,
        },
    }


# ─── Objective ────────────────────────────────────────────────────────────────

def _objective(trial, base_config: dict, dirs_dict: dict,
               hpo_cfg: dict, trial_root: str,
               device: torch.device) -> float:
    """Optuna objective: one full trial → returns optimisation metric."""
    _seed(42 + trial.number)

    # Merge trial suggestions into config
    overrides   = _suggest_hyperparams(trial, hpo_cfg, base_config)
    trial_cfg   = _deep_merge(base_config, overrides)
    max_patches = trial_cfg['training'].get('max_patches')

    # ── Datasets ──────────────────────────────────────────────────────────────
    try:
        datasets, class_names = build_mil_datasets(
            trial_cfg, dirs_dict,
            max_patches=max_patches)
    except Exception as e:
        logger.warning(f"Trial {trial.number}: dataset build failed — {e}")
        raise optuna.TrialPruned()

    if 'train' not in datasets or len(datasets['train']) == 0:
        raise optuna.TrialPruned()

    n_classes  = trial_cfg['task'].get('num_classes', 2)
    train_cfg  = trial_cfg.get('training', {})
    nw         = int(train_cfg.get('num_workers', 0))

    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=True,
                              num_workers=nw, collate_fn=mil_collate_fn)
    val_loader   = (DataLoader(datasets['val'], batch_size=1, shuffle=False,
                               num_workers=nw, collate_fn=mil_collate_fn)
                    if 'val' in datasets and len(datasets['val']) > 0 else None)

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        model, _ = build_mil_model(trial_cfg)
    except Exception as e:
        logger.warning(f"Trial {trial.number}: model build failed — {e}")
        raise optuna.TrialPruned()
    model.to(device)

    use_clam   = trial_cfg['mil']['model'].startswith('clam')
    bag_weight = float(trial_cfg['mil'].get('bag_weight', 0.7))

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    n_epochs      = int(hpo_cfg.get('epochs_per_trial', 30))
    warmup_epochs = int(train_cfg.get('warmup_epochs', 0))
    base_lr       = float(train_cfg.get('learning_rate', 2e-4))
    optimizer  = _build_optimizer(trial_cfg, model)
    scheduler  = _build_scheduler(trial_cfg, optimizer, n_epochs)
    loss_fn    = _build_loss_fn(trial_cfg, class_names, device)

    patch_dropout = float(train_cfg.get('patch_dropout', 0.0))
    patch_shuffle = bool(train_cfg.get('patch_shuffle', False))

    # ── Early stopping ────────────────────────────────────────────────────────
    es_patience = int(train_cfg.get('early_stopping_patience', 15))
    es_min      = int(train_cfg.get('early_stopping_min_epochs', 5))
    best_metric = 0. if hpo_cfg.get('direction', 'maximize') == 'maximize' else float('inf')
    es_counter  = 0

    epoch_logs  = []
    metric_key  = hpo_cfg.get('metric', 'val_auc')

    # ── Per-trial directory ───────────────────────────────────────────────────
    trial_dir = os.path.join(trial_root, f'trial_{trial.number:04d}')
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, 'trial_config.yaml'), 'w') as f:
        yaml.dump(trial_cfg, f, default_flow_style=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        _warmup_lr(optimizer, epoch, warmup_epochs, base_lr)

        tr_loss, tr_met = _run_epoch_hpo(
            model, train_loader, optimizer, loss_fn,
            n_classes, device, is_train=True,
            use_clam=use_clam, bag_weight=bag_weight,
            patch_dropout=patch_dropout, patch_shuffle=patch_shuffle)

        if val_loader is not None:
            vl_loss, vl_met = _run_epoch_hpo(
                model, val_loader, optimizer, loss_fn,
                n_classes, device, is_train=False,
                use_clam=use_clam, bag_weight=bag_weight)
        else:
            vl_loss, vl_met = tr_loss, tr_met

        # Step scheduler
        sched_name = train_cfg.get('lr_scheduler', 'plateau')
        if sched_name == 'plateau':
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        current_metric = vl_met.get(metric_key.replace('val_', ''), 0.0)
        epoch_logs.append({'epoch': epoch, **{f'val_{k}': v for k, v in vl_met.items()},
                           **{f'tr_{k}': v for k, v in tr_met.items()},
                           'tr_loss': tr_loss, 'val_loss': vl_loss})

        # Optuna intermediate reporting + pruning
        trial.report(current_metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        direction = hpo_cfg.get('direction', 'maximize')
        improved  = (current_metric > best_metric if direction == 'maximize'
                     else current_metric < best_metric)
        if improved:
            best_metric = current_metric
            es_counter  = 0
        else:
            es_counter += 1
            if epoch >= es_min and es_counter >= es_patience:
                break

    # Save per-trial metrics
    with open(os.path.join(trial_dir, 'trial_metrics.json'), 'w') as f:
        json.dump({'best_metric': best_metric,
                   'metric_name': metric_key,
                   'n_epochs_run': epoch,
                   'epoch_logs': epoch_logs}, f, indent=2)

    return best_metric


# ─── Main command ──────────────────────────────────────────────────────────────

def command_hpo(config: dict, dirs_dict: dict, log=None):
    """
    Run an Optuna HPO study for MIL hyperparameter optimisation.
    Saves best_config.yaml to results/hpo/<study_name>/ upon completion.
    """
    _log = log or logger

    if not _OPTUNA_AVAILABLE:
        _log.error(
            "Optuna is not installed. Install it with:\n"
            "  pip install optuna\n"
            "Then re-run: python main.py hpo --config config/config.yaml")
        return

    hpo_cfg    = config.get('hpo', {})
    study_name = hpo_cfg.get('study_name', 'mil_hpo')
    n_trials   = int(hpo_cfg.get('n_trials', 30))
    direction  = hpo_cfg.get('direction', 'maximize')
    timeout    = hpo_cfg.get('timeout_hours')
    timeout_s  = float(timeout) * 3600 if timeout else None
    metric     = hpo_cfg.get('metric', 'val_auc')
    use_pruner = bool(hpo_cfg.get('pruning', True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Output directories ────────────────────────────────────────────────────
    hpo_root   = os.path.join(config['paths']['results_dir'], 'hpo', study_name)
    trial_root = hpo_root
    os.makedirs(hpo_root, exist_ok=True)

    # Optuna SQLite storage for resumable studies
    storage_path = os.path.join(hpo_root, 'study.db')
    storage      = f"sqlite:///{storage_path}"

    pruner  = MedianPruner(n_startup_trials=3, n_warmup_steps=5) if use_pruner else None
    sampler = TPESampler(seed=42)

    _log.info("=" * 60)
    _log.info(f"  HPO STUDY: {study_name}")
    _log.info(f"  Trials   : {n_trials}  |  Direction: {direction}")
    _log.info(f"  Metric   : {metric}")
    _log.info(f"  Device   : {device}")
    _log.info(f"  Output   : {hpo_root}")
    _log.info("=" * 60)

    study = optuna.create_study(
        study_name = study_name,
        storage    = storage,
        direction  = direction,
        sampler    = sampler,
        pruner     = pruner,
        load_if_exists = True,   # allow resuming
    )

    study.optimize(
        lambda trial: _objective(trial, config, dirs_dict,
                                 hpo_cfg, trial_root, device),
        n_trials  = n_trials,
        timeout   = timeout_s,
        gc_after_trial = True,
    )

    # ── Best trial summary ────────────────────────────────────────────────────
    best = study.best_trial
    _log.info("\n" + "=" * 60)
    _log.info(f"  HPO COMPLETE")
    _log.info(f"  Best trial    : #{best.number}")
    _log.info(f"  Best {metric} : {best.value:.4f}")
    _log.info(f"  Best params   :")
    for k, v in best.params.items():
        _log.info(f"      {k}: {v}")
    _log.info("=" * 60)

    # ── Build best config ─────────────────────────────────────────────────────
    overrides    = _suggest_from_params(best.params, config)
    best_config  = _deep_merge(config, overrides)

    best_config_path = os.path.join(hpo_root, 'best_config.yaml')
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    # ── best_trial.json ───────────────────────────────────────────────────────
    best_trial_info = {
        'trial_number'  : best.number,
        'metric_name'   : metric,
        'best_value'    : best.value,
        'params'        : best.params,
        'best_config_path': best_config_path,
        'timestamp'     : datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(hpo_root, 'best_trial.json'), 'w') as f:
        json.dump(best_trial_info, f, indent=2)

    # ── hpo_results.csv ───────────────────────────────────────────────────────
    import pandas as pd
    rows = []
    for t in study.trials:
        if t.state.name == 'COMPLETE':
            row = {'trial': t.number, metric: t.value}
            row.update(t.params)
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows).sort_values(metric, ascending=(direction == 'minimize'))
        df.to_csv(os.path.join(hpo_root, 'hpo_results.csv'), index=False)

    _log.info(f"\n  Best config saved → {best_config_path}")
    _log.info(f"  All results CSV  → {os.path.join(hpo_root, 'hpo_results.csv')}")
    _log.info(
        f"\n  Train with best config:\n"
        f"    python main.py train --config config/config.yaml --use_best_config")

    return best_config_path


def _suggest_from_params(params: dict, base_config: dict) -> dict:
    """Reconstruct config overrides dict from Optuna best trial params."""
    return {
        'mil': {
            'model':               params.get('model', base_config.get('mil', {}).get('model', 'abmil')),
            'hidden_dim':          params.get('attn_hidden_dim', 256),
            'dropout':             params.get('dropout', 0.25),
            'dropout_attn':        params.get('dropout_attn', 0.25),
            'dropout_classifier':  params.get('dropout_classifier', 0.1),
            'feature_proj_dim':    params.get('feature_proj_dim', 512),
        },
        'training': {
            'optimizer':               params.get('optimizer', 'AdamW'),
            'learning_rate':           params.get('learning_rate', 2e-4),
            'weight_decay':            params.get('weight_decay', 1e-4),
            'lr_scheduler':            params.get('lr_scheduler', 'plateau'),
            'label_smoothing':         params.get('label_smoothing', 0.0),
            'early_stopping_patience': params.get('early_stop_patience', 15),
            'warmup_epochs':           params.get('warmup_epochs', 0),
            'patch_dropout':           params.get('patch_dropout', 0.0),
            'patch_shuffle':           params.get('patch_shuffle', True),
            'max_patches':             params.get('max_patches'),
        },
    }


def load_best_config(config: dict, log=None) -> dict:
    """
    Load best_config.yaml from the HPO results directory and merge it into config.
    Called by command_train when --use_best_config is passed.
    Returns the merged config, or the original config if no HPO results found.
    """
    _log = log or logger
    hpo_cfg    = config.get('hpo', {})
    study_name = hpo_cfg.get('study_name', 'mil_hpo')
    best_path  = os.path.join(config['paths']['results_dir'],
                               'hpo', study_name, 'best_config.yaml')
    if not os.path.exists(best_path):
        _log.warning(
            f"No best_config.yaml found at {best_path}. "
            "Run `python main.py hpo --config ...` first.")
        return config

    with open(best_path) as f:
        best_cfg = yaml.safe_load(f)

    merged = _deep_merge(config, best_cfg)
    _log.info(f"Loaded best HPO config from: {best_path}")
    _log.info(f"  Model     : {merged.get('mil', {}).get('model')}")
    _log.info(f"  LR        : {merged.get('training', {}).get('learning_rate')}")
    _log.info(f"  Dropout   : {merged.get('mil', {}).get('dropout')}")
    _log.info(f"  Scheduler : {merged.get('training', {}).get('lr_scheduler')}")
    return merged
