"""
pipelines/multilabel_hpo.py

Hyperparameter Optimisation (HPO) for MIL Multi-Label Classification.

Uses Optuna to search over:
  model, optimizer, learning_rate, weight_decay, dropout, dropout_attn,
  dropout_classifier, attn_hidden_dim, feature_proj_dim, lr_scheduler,
  label_smoothing, early_stop_patience, warmup_epochs, patch_dropout,
  patch_shuffle, max_patches, loss_fn, focal_gamma, threshold

Each invocation creates a unique timestamped and feature-labelled experiment:
    results/multilabel/hpo/<study_name>__<feat_dir>__<YYYYMMDD_HHMMSS>/
        experiment_info.json
        base_config.yaml
        trial_XXXX/
            trial_config.yaml
            trial_metrics.json
        best_config.yaml
        best_trial.json
        hpo_results.csv
        study.db

CLI:
    python main.py multilabel-hpo --config config/config.yaml
    python main.py multilabel-hpo --config config/config.yaml --features <dir>
"""

from __future__ import annotations

import os
import copy
import json
import logging
import datetime
from typing import Dict, List, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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

from datasets.mil_multilabel_dataset import (
    build_multilabel_datasets, multilabel_collate_fn,
    MultiLabelMILDataset,
)
from models.mil_multilabel_models import build_multilabel_model, _ML_MODEL_REGISTRY
from pipelines.multilabel_train import (
    _run_ml_epoch, FocalLoss, _build_multilabel_loss,
)

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

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


def _validate_ml_hpo_config(config: dict, log) -> bool:
    """Validate all required config keys exist before HPO starts."""
    required = {
        "mil"      : ["model", "encoding_size"],
        "multilabel": ["label_names"],
        "multilabel_training": ["max_epochs"],
        "paths"    : ["results_dir"],
    }
    ok = True
    for section, keys in required.items():
        for key in keys:
            val = config.get(section, {}).get(key)
            if val is None:
                log.error(f"MultiLabel HPO: config missing required key '{section}.{key}'")
                ok = False
    return ok


# ─── Trial objective ──────────────────────────────────────────────────────────────

def _ml_objective(
    trial: "optuna.Trial",
    base_config: dict,
    dirs_dict: dict,
    hpo_cfg: dict,
    trial_root: str,
    device: torch.device,
) -> float:
    """
    Optuna objective for one HPO trial.
    Returns the val metric (higher = better if direction=maximize).
    """
    ss = hpo_cfg.get("search_space", {})

    # ── Suggest hyperparameters ───────────────────────────────────────────────
    model_choices = ss.get("model", list(_ML_MODEL_REGISTRY.keys()))
    if not isinstance(model_choices, list):
        model_choices = list(_ML_MODEL_REGISTRY.keys())
    model_key = trial.suggest_categorical("model", model_choices)

    optimizer_name = trial.suggest_categorical(
        "optimizer", ss.get("optimizer", ["AdamW", "Adam"]))

    lr_range  = ss.get("learning_rate", [1e-4, 2e-3])
    lr        = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)

    wd_range  = ss.get("weight_decay", [1e-5, 5e-4])
    wd        = trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)

    do_range  = ss.get("dropout", [0.1, 0.6])
    dropout   = trial.suggest_float("dropout", do_range[0], do_range[1])

    doa_range = ss.get("dropout_attn", [0.2, 0.4])
    dropout_attn = trial.suggest_float("dropout_attn", doa_range[0], doa_range[1])

    doc_range = ss.get("dropout_classifier", [0.1, 0.3])
    dropout_clf = trial.suggest_float("dropout_classifier", doc_range[0], doc_range[1])

    ahd_choices = ss.get("attn_hidden_dim", [32, 64, 128])
    attn_hidden = trial.suggest_categorical("attn_hidden_dim", ahd_choices)

    fpd_choices = ss.get("feature_proj_dim", [256, 512])
    feat_proj   = trial.suggest_categorical("feature_proj_dim", fpd_choices)

    sched_choices = ss.get("lr_scheduler", ["cosine", "step", "plateau"])
    sched_name    = trial.suggest_categorical("lr_scheduler", sched_choices)

    ls_range   = ss.get("label_smoothing", [0.0, 0.15])
    label_sm   = trial.suggest_float("label_smoothing", ls_range[0], ls_range[1])

    es_choices = ss.get("early_stop_patience", [10, 15])
    es_pat     = trial.suggest_categorical("early_stop_patience", es_choices)

    wu_choices = ss.get("warmup_epochs", [0, 2, 5])
    wu_ep      = trial.suggest_categorical("warmup_epochs", wu_choices)

    pd_range   = ss.get("patch_dropout", [0.0, 0.3])
    patch_drop = trial.suggest_float("patch_dropout", pd_range[0], pd_range[1])

    mp_choices = ss.get("max_patches", [None, 1000, 2000])
    max_p      = trial.suggest_categorical("max_patches", mp_choices)

    # Multi-label specific
    loss_choices = ss.get("loss", ["bce", "focal"])
    loss_fn_name = trial.suggest_categorical("loss", loss_choices)

    fg_range   = ss.get("focal_gamma", [1.0, 3.0])
    focal_g    = trial.suggest_float("focal_gamma", fg_range[0], fg_range[1])

    thr_range  = ss.get("threshold", [0.3, 0.7])
    threshold  = trial.suggest_float("threshold", thr_range[0], thr_range[1])

    # ── Build trial config ────────────────────────────────────────────────────
    trial_config = _deep_merge(base_config, {
        "mil": {
            "model"          : model_key,
            "hidden_dim"     : attn_hidden,
            "dropout"        : dropout,
            "feature_proj_dim": feat_proj,
        },
        "multilabel": {
            "threshold": threshold,
        },
        "multilabel_training": {
            "optimizer"              : optimizer_name,
            "learning_rate"          : lr,
            "weight_decay"           : wd,
            "lr_scheduler"           : sched_name,
            "label_smoothing"        : label_sm,
            "early_stopping_patience": es_pat,
            "warmup_epochs"          : wu_ep,
            "patch_dropout"          : patch_drop,
            "max_patches"            : max_p,
            "loss"                   : loss_fn_name,
            "focal_gamma"            : focal_g,
            "max_epochs"             : int(hpo_cfg.get("epochs_per_trial", 30)),
        },
    })

    # ── Save trial config ─────────────────────────────────────────────────────
    trial_dir = os.path.join(trial_root, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, "trial_config.yaml"), "w") as f:
        yaml.dump(trial_config, f, default_flow_style=False)

    # ── Build datasets ────────────────────────────────────────────────────────
    n_folds = int(hpo_cfg.get("n_folds", 1))
    metric  = hpo_cfg.get("metric", "val_macro_auc")
    metric_key = metric.replace("val_", "")   # e.g. "macro_auc"

    max_patches_val = max_p if max_p else None
    datasets, label_names = build_multilabel_datasets(
        trial_config, dirs_dict, max_patches=max_patches_val)
    n_labels = len(label_names)

    if "train" not in datasets or len(datasets["train"]) == 0:
        raise optuna.exceptions.TrialPruned()

    # ── Single split or K-fold ────────────────────────────────────────────────
    if n_folds <= 1:
        # Standard train/val
        if "val" not in datasets or len(datasets["val"]) == 0:
            raise optuna.exceptions.TrialPruned()
        score = _train_trial_fold(
            trial_config, datasets["train"], datasets["val"],
            label_names, device, trial_dir, trial.number, hpo_cfg)
    else:
        # K-fold: pool train + val
        all_slides = list(range(len(datasets["train"])))
        val_slides = list(range(len(datasets.get("val", []))))
        combined   = datasets["train"]

        if "val" in datasets:
            from torch.utils.data import ConcatDataset
            combined = ConcatDataset([datasets["train"], datasets.get("val")])
            all_slides = list(range(len(combined)))

        # Stratification: use first active label as proxy
        labels_proxy = []
        for i in all_slides:
            try:
                _, lv, _ = combined[i]
                labels_proxy.append(int(lv.numpy().argmax()))
            except Exception:
                labels_proxy.append(0)
        labels_proxy = np.array(labels_proxy)

        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(
                skf.split(all_slides, labels_proxy)):
            tr_sub = Subset(combined, tr_idx)
            va_sub = Subset(combined, va_idx)
            s = _train_trial_fold(
                trial_config, tr_sub, va_sub,
                label_names, device,
                os.path.join(trial_dir, f"fold_{fold_i}"),
                trial.number, hpo_cfg, metric_key)
            scores.append(s)
        score = float(np.mean(scores))

    # ── Save trial metrics ────────────────────────────────────────────────────
    with open(os.path.join(trial_dir, "trial_metrics.json"), "w") as f:
        json.dump({"trial": trial.number, metric: score,
                   "params": trial.params}, f, indent=2)

    return score


def _train_trial_fold(config, train_ds, val_ds,
                       label_names, device, trial_dir, trial_num, hpo_cfg,
                       metric_key="macro_auc"):
    """Train one trial (or fold). Return best metric value."""
    nw        = int(config.get("training", {}).get("num_workers", 0))
    train_cfg = config.get("multilabel_training", {})
    max_ep    = int(train_cfg.get("max_epochs", 30))
    threshold = float(config.get("multilabel", {}).get("threshold", 0.5))
    sched_name= train_cfg.get("lr_scheduler", "plateau")
    lr        = float(train_cfg.get("learning_rate", 2e-4))
    wd        = float(train_cfg.get("weight_decay", 1e-4))
    do_es     = True
    es_pat    = int(train_cfg.get("early_stopping_patience", 15))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=nw, collate_fn=multilabel_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=nw, collate_fn=multilabel_collate_fn)

    model, _ = build_multilabel_model(config)
    model.to(device)

    opt_name = train_cfg.get("optimizer", "AdamW")
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
            optimizer, "min", factor=0.5, patience=5)

    # Get pos_weight from raw training dataset
    raw_ds = getattr(train_ds, "dataset", train_ds)
    loss_fn = _build_multilabel_loss(config, raw_ds if hasattr(raw_ds, "label_pos_weights") else None, device)

    n_labels     = len(label_names)
    best_val_metric = -float("inf")
    es_counter   = 0

    # Progress bar
    epoch_iter = (range(1, max_ep + 1) if not _TQDM_AVAILABLE
                  else _tqdm(range(1, max_ep + 1),
                             desc=f"Trial", unit="ep", leave=False))

    epoch_metrics_history = []

    for epoch in epoch_iter:
        _, _ = _run_ml_epoch(
            model, train_loader, optimizer, loss_fn,
            n_labels, label_names, device, is_train=True,
            threshold=threshold,
            patch_dropout=float(train_cfg.get("patch_dropout", 0.0)),
            patch_shuffle=bool(train_cfg.get("patch_shuffle", False)))

        vl_loss, vl_met = _run_ml_epoch(
            model, val_loader, optimizer, loss_fn,
            n_labels, label_names, device, is_train=False,
            threshold=threshold)

        if sched_name == "plateau":
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        cur_val = vl_met.get(metric_key, vl_met.get("macro_auc", 0.0))
        epoch_metrics_history.append({"epoch": epoch, "val_loss": vl_loss,
                                       metric_key: cur_val})

        if _TQDM_AVAILABLE and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix({metric_key: f"{cur_val:.3f}",
                                    "loss": f"{vl_loss:.4f}"})

        if cur_val > best_val_metric:
            best_val_metric = cur_val
            es_counter = 0
        else:
            es_counter += 1
            if do_es and es_counter >= es_pat:
                break

    # Save epoch metrics to trial dir
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, "epoch_metrics.json"), "w") as f:
        json.dump(epoch_metrics_history, f, indent=2)

    return best_val_metric


# ─── Build best config from params ────────────────────────────────────────────────

def _ml_suggest_from_params(params: dict, base_config: dict) -> dict:
    """Reconstruct ML config overrides dict from Optuna best trial params."""
    return {
        "mil": {
            "model"          : params.get("model", base_config.get("mil", {}).get("model", "abmil")),
            "hidden_dim"     : params.get("attn_hidden_dim", 256),
            "dropout"        : params.get("dropout", 0.25),
            "feature_proj_dim": params.get("feature_proj_dim", 512),
        },
        "multilabel": {
            "threshold": params.get("threshold", 0.5),
        },
        "multilabel_training": {
            "optimizer"              : params.get("optimizer", "AdamW"),
            "learning_rate"          : params.get("learning_rate", 2e-4),
            "weight_decay"           : params.get("weight_decay", 1e-4),
            "lr_scheduler"           : params.get("lr_scheduler", "plateau"),
            "label_smoothing"        : params.get("label_smoothing", 0.0),
            "early_stopping_patience": params.get("early_stop_patience", 15),
            "warmup_epochs"          : params.get("warmup_epochs", 0),
            "patch_dropout"          : params.get("patch_dropout", 0.0),
            "max_patches"            : params.get("max_patches"),
            "loss"                   : params.get("loss", "bce"),
            "focal_gamma"            : params.get("focal_gamma", 2.0),
        },
    }


# ─── Main command ─────────────────────────────────────────────────────────────────

def command_multilabel_hpo(config: dict, dirs_dict: dict, log=None):
    """
    Run an Optuna HPO study for MIL multi-label classification.
    Each invocation creates a unique, isolated experiment directory.
    """
    _log = log or logger

    if not _OPTUNA_AVAILABLE:
        _log.error(
            "Optuna not installed. Install with:\n  pip install optuna tqdm\n"
            "Then re-run: python main.py multilabel-hpo --config ...")
        return

    if not _validate_ml_hpo_config(config, _log):
        _log.error("MultiLabel HPO aborted due to missing configuration keys.")
        return

    hpo_cfg   = config.get("multilabel_hpo", {})
    base_name = hpo_cfg.get("study_name", "ml_hpo")
    n_trials  = int(hpo_cfg.get("n_trials", 30))
    direction = hpo_cfg.get("direction", "maximize")
    timeout   = hpo_cfg.get("timeout_hours")
    timeout_s = float(timeout) * 3600 if timeout else None
    metric    = hpo_cfg.get("metric", "val_macro_auc")
    use_pruner= bool(hpo_cfg.get("pruning", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Unique run ID ─────────────────────────────────────────────────────────
    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feat_dir   = dirs_dict.get("features", "")
    feat_label = os.path.basename(feat_dir.rstrip("/\\")) if feat_dir else "default_features"
    run_id     = f"{base_name}__{feat_label}__{timestamp}"

    # ── Output directory ──────────────────────────────────────────────────────
    hpo_root   = os.path.join(config["paths"]["results_dir"],
                               "multilabel", "hpo", run_id)
    os.makedirs(hpo_root, exist_ok=True)

    storage_path = os.path.join(hpo_root, "study.db")
    storage      = f"sqlite:///{storage_path}"

    pruner  = MedianPruner(n_startup_trials=3, n_warmup_steps=5) if use_pruner else None
    sampler = TPESampler(seed=42)

    _log.info("=" * 70)
    _log.info(f"  ML HPO STUDY  : {base_name}")
    _log.info(f"  Run ID        : {run_id}")
    _log.info(f"  Features dir  : {feat_dir or '(default)'}")
    _log.info(f"  Trials        : {n_trials}  |  Metric: {metric}")
    _log.info(f"  Labels        : {config.get('multilabel', {}).get('label_names', [])}")
    _log.info(f"  Device        : {device}")
    _log.info(f"  Output        : {hpo_root}")
    _log.info("=" * 70)

    # ── Save metadata ─────────────────────────────────────────────────────────
    exp_info = {
        "run_id": run_id, "study_name": base_name,
        "features_dir": feat_dir, "feat_label": feat_label,
        "timestamp": timestamp, "n_trials": n_trials, "metric": metric,
        "direction": direction,
        "model": config.get("mil", {}).get("model"),
        "label_names": config.get("multilabel", {}).get("label_names", []),
        "task": config.get("task", {}).get("name"),
        "device": str(device), "hpo_root": hpo_root,
    }
    with open(os.path.join(hpo_root, "experiment_info.json"), "w") as f:
        json.dump(exp_info, f, indent=2)
    with open(os.path.join(hpo_root, "base_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Run study ─────────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name     = run_id,
        storage        = storage,
        direction      = direction,
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True,
    )

    study.optimize(
        lambda trial: _ml_objective(trial, config, dirs_dict,
                                    hpo_cfg, hpo_root, device),
        n_trials       = n_trials,
        timeout        = timeout_s,
        gc_after_trial = True,
    )

    # ── Best trial ────────────────────────────────────────────────────────────
    best = study.best_trial
    _log.info("\n" + "=" * 70)
    _log.info(f"  ML HPO COMPLETE  — Run: {run_id}")
    _log.info(f"  Best trial    : #{best.number}")
    _log.info(f"  Best {metric} : {best.value:.4f}")
    for k, v in best.params.items():
        _log.info(f"      {k}: {v}")
    _log.info("=" * 70)

    overrides        = _ml_suggest_from_params(best.params, config)
    best_config      = _deep_merge(config, overrides)
    best_config_path = os.path.join(hpo_root, "best_config.yaml")
    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    best_trial_info = {
        "run_id"          : run_id,
        "trial_number"    : best.number,
        "metric_name"     : metric,
        "best_value"      : best.value,
        "params"          : best.params,
        "features_dir"    : feat_dir,
        "best_config_path": best_config_path,
        "timestamp"       : datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(hpo_root, "best_trial.json"), "w") as f:
        json.dump(best_trial_info, f, indent=2)

    import pandas as pd
    rows = []
    for t in study.trials:
        if t.state.name == "COMPLETE":
            row = {"trial": t.number, metric: t.value}
            row.update(t.params)
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows).sort_values(
            metric, ascending=(direction == "minimize"))
        df.to_csv(os.path.join(hpo_root, "hpo_results.csv"), index=False)

    _log.info(f"\n  Best config → {best_config_path}")
    _log.info(
        "\n  Next steps:\n"
        "    python main.py multilabel-train --config config/config.yaml --use_best_config\n"
        "    python main.py multilabel-crossval --config config/config.yaml --use_best_config")
    return best_config_path


# ─── Load best multilabel config ──────────────────────────────────────────────────

def load_best_multilabel_config(config: dict, log=None) -> dict:
    """
    Auto-find and load the best_config.yaml from the most recent multilabel HPO run.
    Falls back to explicit hpo.best_run_path if set.
    """
    _log      = log or logger
    hpo_cfg   = config.get("multilabel_hpo", {})
    base_name = hpo_cfg.get("study_name", "ml_hpo")
    hpo_base  = os.path.join(config["paths"]["results_dir"], "multilabel", "hpo")

    # Explicit override
    explicit = hpo_cfg.get("best_run_path")
    if explicit:
        candidate = (os.path.join(explicit, "best_config.yaml")
                     if not explicit.endswith(".yaml") else explicit)
        if os.path.exists(candidate):
            return _load_and_merge(config, candidate, _log)
        _log.warning(f"Explicit best_run_path not found: {candidate}")

    # Auto-find most recent matching run
    best_path = None
    if os.path.isdir(hpo_base):
        candidates = []
        for entry in os.scandir(hpo_base):
            if entry.is_dir() and entry.name.startswith(base_name):
                bc = os.path.join(entry.path, "best_config.yaml")
                if os.path.isfile(bc):
                    candidates.append((entry.stat().st_mtime, bc, entry.name))
        if candidates:
            candidates.sort(reverse=True)
            _, best_path, run_id = candidates[0]
            _log.info(f"Auto-selected most recent ML HPO run: {run_id}")
            if len(candidates) > 1:
                _log.info(f"  ({len(candidates)} matching runs found; "
                          "set multilabel_hpo.best_run_path to pin one)")

    if best_path is None:
        _log.warning(
            f"No multilabel HPO best_config.yaml found under {hpo_base}/{base_name}*.\n"
            "Run `python main.py multilabel-hpo --config ...` first.")
        return config

    return _load_and_merge(config, best_path, _log)


def _load_and_merge(config: dict, best_path: str, _log) -> dict:
    """Load a best_config.yaml and deep-merge into config."""
    with open(best_path) as f:
        best_cfg = yaml.safe_load(f)
    merged = _deep_merge(config, best_cfg)
    _log.info(f"Loaded best ML HPO config from: {best_path}")
    _log.info(f"  Model     : {merged.get('mil', {}).get('model')}")
    _log.info(f"  LR        : {merged.get('multilabel_training', {}).get('learning_rate')}")
    _log.info(f"  Loss      : {merged.get('multilabel_training', {}).get('loss')}")
    _log.info(f"  Threshold : {merged.get('multilabel', {}).get('threshold')}")
    return merged
