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
import logging
import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    _tqdm = None

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

from datasets.mil_dataset import build_mil_datasets, mil_collate_fn, MILBagDataset
from models.mil_models import build_mil_model, _MODEL_REGISTRY

logger = logging.getLogger(__name__)


def _validate_hpo_config(config: dict, log) -> bool:
    """Validate all required config keys are present before HPO starts."""
    required = {
        'mil':      ['model'],
        'task':     ['name', 'num_classes', 'class_names'],
        'training': ['max_epochs'],
        'paths':    ['results_dir'],
    }
    ok = True
    for section, keys in required.items():
        for key in keys:
            val = config.get(section, {}).get(key)
            if val is None:
                log.error(f"HPO: config missing required key '{section}.{key}'")
                ok = False
    return ok


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
    train_cfg      = trial_cfg.get('training', {})
    optimizer_name = train_cfg.get('optimizer', 'AdamW')
    lr  = float(train_cfg.get('learning_rate', 2e-3))
    wd  = float(train_cfg.get('weight_decay', 1e-3))
    b1  = float(train_cfg.get('beta1', 0.75))
    b2  = float(train_cfg.get('beta2', 0.95))
    eps = float(train_cfg.get('eps',   1e-2))

    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=wd, betas=(b1, b2), eps=eps)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=wd, betas=(b1, b2), eps=eps)


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
            factor   = float(train_cfg.get('lr_scheduler_factor',  0.75)),
            patience = int(train_cfg.get('lr_scheduler_patience', 20)))


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
    wd             = _loguni('weight_decay', 1e-5, 5e-3)
    b1             = _uni('beta1', 0.5, 0.99)
    b2             = _uni('beta2', 0.9, 0.999)
    eps            = _loguni('eps', 1e-8, 1e-2)
    dropout        = _uni('dropout', 0.1, 0.6)
    dropout_attn   = _uni('dropout_attn', 0.2, 0.5)
    dropout_clf    = _uni('dropout_classifier', 0.1, 0.4)
    hidden_dim     = _cat('attn_hidden_dim', 256)
    proj_dim       = _cat('proj_dim', 512)
    scheduler_name = _cat('lr_scheduler', 'plateau')
    lr_factor      = _uni('lr_factor', 0.1, 0.9)
    lr_patience    = _cat('lr_patience', 20)
    label_smoothing = _uni('label_smoothing', 0.0, 0.15)
    es_patience    = _cat('early_stop_patience', 20)
    warmup_epochs  = _cat('warmup_epochs', 0)
    patch_dropout  = _uni('patch_dropout', 0.0, 0.3)
    patch_shuffle  = True   # always on (no meaningful trade-off to tune)
    max_patches    = _cat('max_patches', 800)   # A_patches: [600,700,800,900,1000,1100,1200]

    return {
        'mil': {
            'model':     model_key,
            'hidden_dim': hidden_dim,
            'dropout':   dropout,
            'dropout_attn':       dropout_attn,
            'dropout_classifier': dropout_clf,
            'proj_dim':  proj_dim,         # canonical name — same as Optuna param and config key
        },
        'training': {
            'optimizer':               optimizer_name,
            'learning_rate':           lr,
            'weight_decay':            wd,
            'beta1':                   b1,
            'beta2':                   b2,
            'eps':                     eps,
            'lr_scheduler':            scheduler_name,
            'lr_scheduler_factor':     lr_factor,
            'lr_scheduler_patience':   lr_patience,
            'label_smoothing':         label_smoothing,
            'early_stopping_patience': es_patience,
            'warmup_epochs':           warmup_epochs,
            'patch_dropout':           patch_dropout,
            'patch_shuffle':           patch_shuffle,
            'max_patches':             max_patches,
        },
    }


# ─── Objective ────────────────────────────────────────────────────────────────

def _objective(trial, base_config: dict, dirs_dict: dict,
               hpo_cfg: dict, trial_root: str,
               device: torch.device) -> float:
    """
    Optuna objective: one full trial -> returns optimisation metric.
    Routes to k-fold CV variant when hpo.n_folds > 1.
    """
    _seed(42 + trial.number)

    overrides   = _suggest_hyperparams(trial, hpo_cfg, base_config)
    trial_cfg   = _deep_merge(base_config, overrides)
    max_patches = trial_cfg['training'].get('max_patches')
    n_folds_cv  = int(hpo_cfg.get('n_folds', 1))

    logger.info(
        f"[HPO] Trial {trial.number:>3} | "
        f"model={trial_cfg['mil']['model']:10s} | "
        f"lr={float(trial_cfg['training'].get('learning_rate', 2e-4)):.2e} | "
        f"sched={trial_cfg['training'].get('lr_scheduler', 'plateau')}")

    trial_dir = os.path.join(trial_root, f'trial_{trial.number:04d}')
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, 'trial_config.yaml'), 'w') as f:
        yaml.dump(trial_cfg, f, default_flow_style=False)

    if n_folds_cv > 1:
        return _objective_kfold(trial, trial_cfg, dirs_dict, hpo_cfg,
                                trial_dir, device, max_patches, n_folds_cv)

    # ── Single train/val split ─────────────────────────────────────────────────
    try:
        datasets, class_names = build_mil_datasets(
            trial_cfg, dirs_dict, max_patches=max_patches)
    except Exception as e:
        logger.warning(f"Trial {trial.number}: dataset build failed — {e}")
        raise optuna.TrialPruned()

    if 'train' not in datasets or len(datasets['train']) == 0:
        raise optuna.TrialPruned()

    n_classes = trial_cfg['task'].get('num_classes', 2)
    nw        = int(trial_cfg.get('training', {}).get('num_workers', 0))

    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=True,
                              num_workers=nw, collate_fn=mil_collate_fn)
    val_loader   = (DataLoader(datasets['val'], batch_size=1, shuffle=False,
                               num_workers=nw, collate_fn=mil_collate_fn)
                    if 'val' in datasets and len(datasets['val']) > 0 else None)

    try:
        model, _ = build_mil_model(trial_cfg)
    except Exception as e:
        logger.warning(f"Trial {trial.number}: model build failed — {e}")
        raise optuna.TrialPruned()
    model.to(device)

    use_clam   = trial_cfg['mil']['model'].startswith('clam')
    bag_weight = float(trial_cfg['mil'].get('bag_weight', 0.7))

    return _run_trial_loop(trial, trial_cfg, hpo_cfg, train_loader, val_loader,
                           model, use_clam, bag_weight, n_classes, device,
                           trial_dir, class_names)


def _objective_kfold(trial, trial_cfg: dict, dirs_dict: dict,
                     hpo_cfg: dict, trial_dir: str, device: torch.device,
                     max_patches, n_folds: int) -> float:
    """K-fold CV variant: averages the metric across folds."""
    import pandas as pd
    feat_dir = dirs_dict.get('features')
    if not feat_dir:
        logger.warning("Trial kfold: no features dir in dirs_dict.")
        raise optuna.TrialPruned()
    pt_dir = os.path.join(feat_dir, 'pt_files')
    if not os.path.isdir(pt_dir):
        logger.warning(f"Trial kfold: pt_files not found: {pt_dir}")
        raise optuna.TrialPruned()

    task_name   = trial_cfg['task']['name']
    class_names = trial_cfg['task']['class_names']
    splits_dir  = os.path.join(trial_cfg['paths']['results_dir'], 'splits', task_name)

    dfs = [pd.read_csv(os.path.join(splits_dir, f'{s}.csv'))
           for s in ('train', 'val')
           if os.path.exists(os.path.join(splits_dir, f'{s}.csv'))]
    if not dfs:
        raise optuna.TrialPruned()

    pool_csv = os.path.join(trial_dir, 'pool.csv')
    pd.concat(dfs, ignore_index=True).to_csv(pool_csv, index=False)

    full_ds = MILBagDataset(pool_csv, pt_dir, class_names, max_patches=max_patches)
    if len(full_ds) == 0:
        raise optuna.TrialPruned()

    labels  = np.array([full_ds[i][1] for i in range(len(full_ds))])
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_classes = trial_cfg['task'].get('num_classes', 2)
    nw        = int(trial_cfg.get('training', {}).get('num_workers', 0))
    scores    = []

    for fold_i, (tr_idx, va_idx) in enumerate(
            skf.split(np.arange(len(full_ds)), labels), start=1):
        tr_loader = DataLoader(Subset(full_ds, tr_idx), batch_size=1,
                               shuffle=True, num_workers=nw, collate_fn=mil_collate_fn)
        va_loader = DataLoader(Subset(full_ds, va_idx), batch_size=1,
                               shuffle=False, num_workers=nw, collate_fn=mil_collate_fn)
        try:
            model, _ = build_mil_model(trial_cfg)
        except Exception:
            continue
        model.to(device)
        use_clam   = trial_cfg['mil']['model'].startswith('clam')
        bag_weight = float(trial_cfg['mil'].get('bag_weight', 0.7))
        score = _run_trial_loop(
            trial, trial_cfg, hpo_cfg, tr_loader, va_loader,
            model, use_clam, bag_weight, n_classes, device,
            trial_dir, class_names, fold_suffix=f'_fold{fold_i}')
        scores.append(score)
        logger.info(f"  [HPO] Trial {trial.number} fold {fold_i}/{n_folds} "
                    f"score={score:.4f}")

    return float(np.mean(scores)) if scores else 0.0


def _run_trial_loop(trial, trial_cfg: dict, hpo_cfg: dict,
                    train_loader, val_loader,
                    model, use_clam: bool, bag_weight: float,
                    n_classes: int, device: torch.device,
                    trial_dir: str, class_names: list,
                    fold_suffix: str = '') -> float:
    """Inner train+validate loop shared by single-split and k-fold objectives."""
    train_cfg     = trial_cfg.get('training', {})
    n_epochs      = int(hpo_cfg.get('epochs_per_trial', 30))
    warmup_ep     = int(train_cfg.get('warmup_epochs', 0))
    base_lr       = float(train_cfg.get('learning_rate', 2e-4))
    patch_dropout = float(train_cfg.get('patch_dropout', 0.0))
    patch_shuffle = bool(train_cfg.get('patch_shuffle', False))
    sched_name    = train_cfg.get('lr_scheduler', 'plateau')
    metric_key    = hpo_cfg.get('metric', 'val_auc')
    direction     = hpo_cfg.get('direction', 'maximize')
    es_patience   = int(train_cfg.get('early_stopping_patience', 15))
    es_min        = int(train_cfg.get('early_stopping_min_epochs', 5))

    optimizer   = _build_optimizer(trial_cfg, model)
    scheduler   = _build_scheduler(trial_cfg, optimizer, n_epochs)
    loss_fn     = _build_loss_fn(trial_cfg, class_names, device)

    best_metric = 0. if direction == 'maximize' else float('inf')
    es_counter  = 0
    epoch_logs  = []
    last_epoch  = 1

    # Live epoch progress via tqdm (if installed, else logs each epochs via logger)
    epoch_range = range(1, n_epochs + 1)
    if _TQDM_AVAILABLE:
        desc = f"  Trial {trial.number}{fold_suffix}"
        epoch_range = _tqdm(epoch_range, desc=desc, unit='ep',
                            leave=False, ncols=95,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')

    for epoch in epoch_range:
        last_epoch = epoch
        _warmup_lr(optimizer, epoch, warmup_ep, base_lr)

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

        if sched_name == 'plateau':
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        current_metric = vl_met.get(metric_key.replace('val_', ''), 0.0)
        epoch_logs.append({
            'epoch': epoch,
            **{f'val_{k}': v for k, v in vl_met.items()},
            **{f'tr_{k}': v for k, v in tr_met.items()},
            'tr_loss': tr_loss, 'val_loss': vl_loss
        })

        # Update tqdm postfix for live progress
        if _TQDM_AVAILABLE and hasattr(epoch_range, 'set_postfix'):
            epoch_range.set_postfix(**{
                metric_key: f"{current_metric:.4f}",
                'loss': f"{vl_loss:.4f}"
            })
        else:
            # Plain-text fallback every 5 epochs
            if epoch % 5 == 0:
                logger.info(
                    f"  [HPO] Trial {trial.number}{fold_suffix} "
                    f"ep {epoch}/{n_epochs} | {metric_key}={current_metric:.4f} "
                    f"val_loss={vl_loss:.4f}")

        # Optuna pruning (single-split only; k-fold doesn't need per-epoch pruning)
        if not fold_suffix:
            trial.report(current_metric, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        improved = (current_metric > best_metric if direction == 'maximize'
                    else current_metric < best_metric)
        if improved:
            best_metric = current_metric
            es_counter  = 0
        else:
            es_counter += 1
            if epoch >= es_min and es_counter >= es_patience:
                break

    with open(os.path.join(trial_dir, f'trial_metrics{fold_suffix}.json'), 'w') as f:
        json.dump({'best_metric': best_metric, 'metric_name': metric_key,
                   'n_epochs_run': last_epoch, 'epoch_logs': epoch_logs}, f, indent=2)

    return best_metric


# ─── Main command ──────────────────────────────────────────────────────────────

def command_hpo(config: dict, dirs_dict: dict, log=None):
    """
    Run an Optuna HPO study for MIL hyperparameter optimisation.

    Each invocation creates a **separate, timestamped experiment directory** so
    that runs with different feature dirs, configs, or timestamps are all
    tracked independently:

        results/hpo/<study_name>__<feat_dir>__<YYYYMMDD_HHMMSS>/

    Saves best_config.yaml there upon completion.
    """
    _log = log or logger

    if not _OPTUNA_AVAILABLE:
        _log.error(
            "Optuna is not installed. Install it with:\n"
            "  pip install optuna\n"
            "Then re-run: python main.py hpo --config config/config.yaml")
        return

    # Validate required config keys before starting
    if not _validate_hpo_config(config, _log):
        _log.error("HPO aborted due to missing configuration keys.")
        return

    hpo_cfg   = config.get('hpo', {})
    base_name = hpo_cfg.get('study_name', 'mil_hpo')
    n_trials  = int(hpo_cfg.get('n_trials', 30))
    direction  = hpo_cfg.get('direction', 'maximize')
    timeout    = hpo_cfg.get('timeout_hours')
    timeout_s  = float(timeout) * 3600 if timeout else None
    metric     = hpo_cfg.get('metric', 'val_auc')
    use_pruner = bool(hpo_cfg.get('pruning', True))
    n_folds    = int(hpo_cfg.get('n_folds', 1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build a unique run identifier ─────────────────────────────────────────
    # Format: <study_name>__<feat_dir_basename>__<YYYYMMDD_HHMMSS>
    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    feat_dir   = dirs_dict.get('features', '')
    feat_label = os.path.basename(feat_dir.rstrip('/\\')) if feat_dir else 'default_features'
    run_id     = f"{base_name}__{feat_label}__{timestamp}"

    # ── Output directories ────────────────────────────────────────────────────
    hpo_root   = os.path.join(config['paths']['results_dir'], 'hpo', run_id)
    trial_root = hpo_root
    os.makedirs(hpo_root, exist_ok=True)

    # Optuna SQLite storage — unique per run (not shared across runs)
    storage_path = os.path.join(hpo_root, 'study.db')
    storage      = f"sqlite:///{storage_path}"

    pruner  = MedianPruner(n_startup_trials=3, n_warmup_steps=5) if use_pruner else None
    sampler = TPESampler(seed=42)

    _log.info("=" * 70)
    _log.info(f"  HPO STUDY     : {base_name}")
    _log.info(f"  Run ID        : {run_id}")
    _log.info(f"  Features dir  : {feat_dir or '(default)'}")
    _log.info(f"  Trials        : {n_trials}  |  Folds/trial: {n_folds}")
    _log.info(f"  Metric        : {metric}  |  Direction: {direction}")
    _log.info(f"  Device        : {device}")
    _log.info(f"  Output        : {hpo_root}")
    _log.info("=" * 70)

    # ── Persist experiment metadata ───────────────────────────────────────────
    exp_info = {
        'run_id'        : run_id,
        'study_name'    : base_name,
        'features_dir'  : feat_dir,
        'feat_label'    : feat_label,
        'timestamp'     : timestamp,
        'n_trials'      : n_trials,
        'n_folds'       : n_folds,
        'metric'        : metric,
        'direction'     : direction,
        'model'         : config.get('mil', {}).get('model'),
        'task'          : config.get('task', {}).get('name'),
        'device'        : str(device),
        'hpo_root'      : hpo_root,
    }
    with open(os.path.join(hpo_root, 'experiment_info.json'), 'w') as f:
        json.dump(exp_info, f, indent=2)

    # Also save the base config used for this run
    with open(os.path.join(hpo_root, 'base_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    study = optuna.create_study(
        study_name     = run_id,        # unique per run
        storage        = storage,
        direction      = direction,
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True,          # allow resuming *this specific run*
    )

    study.optimize(
        lambda trial: _objective(trial, config, dirs_dict,
                                 hpo_cfg, trial_root, device),
        n_trials       = n_trials,
        timeout        = timeout_s,
        gc_after_trial = True,
    )

    # ── Best trial summary ────────────────────────────────────────────────────
    best = study.best_trial
    _log.info("\n" + "=" * 70)
    _log.info(f"  HPO COMPLETE  — Run: {run_id}")
    _log.info(f"  Best trial    : #{best.number}")
    _log.info(f"  Best {metric} : {best.value:.4f}")
    _log.info(f"  Best params   :")
    for k, v in best.params.items():
        _log.info(f"      {k}: {v}")
    _log.info("=" * 70)

    # ── Build & save best config ──────────────────────────────────────────────
    overrides        = _suggest_from_params(best.params, config)
    best_config      = _deep_merge(config, overrides)
    best_config_path = os.path.join(hpo_root, 'best_config.yaml')
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    # ── best_trial.json ───────────────────────────────────────────────────────
    best_trial_info = {
        'run_id'          : run_id,
        'trial_number'    : best.number,
        'metric_name'     : metric,
        'best_value'      : best.value,
        'params'          : best.params,
        'features_dir'    : feat_dir,
        'best_config_path': best_config_path,
        'timestamp'       : datetime.datetime.now().isoformat(timespec='seconds'),
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
        f"    python main.py train --config config/config.yaml --use_best_config\n"
        f"\n  Cross-validate with best config:\n"
        f"    python main.py crossval --config config/config.yaml --use_best_config")

    return best_config_path


def _suggest_from_params(params: dict, base_config: dict) -> dict:
    """Reconstruct config overrides dict from Optuna best trial params.

    Maps Optuna parameter names → config keys.
    Note: Optuna stores the suggested value under the name passed to
    trial.suggest_*() — e.g. 'feature_proj_dim' for the proj_dim search param.
    We normalise everything here so downstream code always reads 'proj_dim'.
    """
    mil_base = base_config.get('mil', {})
    return {
        'mil': {
            'model':              params.get('model',          mil_base.get('model', 'abmil')),
            'hidden_dim':         params.get('attn_hidden_dim', 256),
            'dropout':            params.get('dropout',         0.4),
            'dropout_attn':       params.get('dropout_attn',    0.25),
            'dropout_classifier': params.get('dropout_classifier', 0.1),
            # Canonical key: proj_dim — Optuna param is also 'proj_dim' now.
            'proj_dim':           params.get('proj_dim', 512),
        },
        'training': {
            'optimizer':               params.get('optimizer',        'AdamW'),
            'learning_rate':           params.get('learning_rate',    2e-3),
            'weight_decay':            params.get('weight_decay',     1e-3),
            'beta1':                   params.get('beta1',            0.75),
            'beta2':                   params.get('beta2',            0.95),
            'eps':                     params.get('eps',              1e-2),
            'lr_scheduler':            params.get('lr_scheduler',     'plateau'),
            'lr_scheduler_factor':     params.get('lr_factor',        0.75),
            'lr_scheduler_patience':   params.get('lr_patience',      20),
            'label_smoothing':         params.get('label_smoothing',  0.0),
            'early_stopping_patience': params.get('early_stop_patience', 20),
            'warmup_epochs':           params.get('warmup_epochs',    0),
            'patch_dropout':           params.get('patch_dropout',    0.0),
            'patch_shuffle':           params.get('patch_shuffle',    False),
            'max_patches':             params.get('max_patches',      800),
        },
    }


def load_best_config(config: dict, log=None) -> dict:
    """
    Load best_config.yaml from the HPO results directory and merge into config.

    Resolution order (first found wins):
      1. config['hpo']['best_run_path'] — explicit path to a specific run dir
      2. Most recently modified run dir under results/hpo/ matching study_name prefix
      3. Legacy: results/hpo/<study_name>/best_config.yaml (single-run layout)

    Called by command_train / command_crossval when --use_best_config is passed.
    """
    _log        = log or logger
    hpo_cfg     = config.get('hpo', {})
    base_name   = hpo_cfg.get('study_name', 'mil_hpo')
    results_dir = config['paths']['results_dir']
    hpo_base    = os.path.join(results_dir, 'hpo')

    # 1. Explicit override in config
    explicit = hpo_cfg.get('best_run_path')
    if explicit:
        candidate = os.path.join(explicit, 'best_config.yaml') \
            if not explicit.endswith('.yaml') else explicit
        if os.path.exists(candidate):
            return _load_and_merge(config, candidate, _log)
        _log.warning(f"Explicit best_run_path not found: {candidate}")

    # 2. Auto-find: walk hpo_base and find most recently modified run dir
    #    whose name starts with the study_name prefix
    best_path = None
    if os.path.isdir(hpo_base):
        candidates = []
        for entry in os.scandir(hpo_base):
            if entry.is_dir() and entry.name.startswith(base_name):
                bc = os.path.join(entry.path, 'best_config.yaml')
                if os.path.isfile(bc):
                    candidates.append((entry.stat().st_mtime, bc, entry.name))
        if candidates:
            candidates.sort(reverse=True)   # most recent first
            _, best_path, run_id = candidates[0]
            _log.info(f"Auto-selected most recent HPO run: {run_id}")
            if len(candidates) > 1:
                _log.info(f"  ({len(candidates)} matching runs found; "
                          "set hpo.best_run_path to pin a specific one)")

    # 3. Legacy single-run path
    if best_path is None:
        legacy = os.path.join(hpo_base, base_name, 'best_config.yaml')
        if os.path.isfile(legacy):
            best_path = legacy
            _log.info("Using legacy single-run HPO path.")

    if best_path is None:
        _log.warning(
            f"No HPO best_config.yaml found under {hpo_base}/{base_name}*.\n"
            "Run `python main.py hpo --config ...` first.")
        return config

    return _load_and_merge(config, best_path, _log)


def _load_and_merge(config: dict, best_path: str, _log) -> dict:
    """Load a best_config.yaml and deep-merge into config."""
    with open(best_path) as f:
        best_cfg = yaml.safe_load(f)
    merged = _deep_merge(config, best_cfg)
    _log.info(f"Loaded best HPO config from: {best_path}")
    _log.info(f"  Model     : {merged.get('mil', {}).get('model')}")
    _log.info(f"  LR        : {merged.get('training', {}).get('learning_rate')}")
    _log.info(f"  Dropout   : {merged.get('mil', {}).get('dropout')}")
    _log.info(f"  Scheduler : {merged.get('training', {}).get('lr_scheduler')}")
    return merged
