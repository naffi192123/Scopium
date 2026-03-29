"""
pipelines/multilabel_train.py

MIL Multi-Label Training Pipeline.

Trains any MIL model for multi-label classification (sigmoid + BCE/Focal loss).
Results are saved to results/multilabel/experiments/<task>/<model>_<timestamp>/
completely separate from the single-label pipeline.

CLI:
    python main.py multilabel-train --config config/config.yaml
    python main.py multilabel-train --config config/config.yaml --features <dir>
    python main.py multilabel-train --config config/config.yaml --use_best_config

Loss functions
--------------
    bce   : BCEWithLogitsLoss (default), optionally with per-label pos_weight
    focal : Focal loss — (1 - p)^gamma * BCE, configurable alpha + gamma

Outputs
-------
    results/multilabel/experiments/<task>/<model>_<timestamp>/
        best_model.pt
        final_model.pt
        train_history.csv
        config_snapshot.yaml
        plots/
"""

from __future__ import annotations

import os
import time
import logging
import datetime
import json
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    hamming_loss, coverage_error,
    precision_score, recall_score,
)

from datasets.mil_multilabel_dataset import (
    build_multilabel_datasets,
    multilabel_collate_fn,
    MultiLabelMILDataset,
)
from models.mil_multilabel_models import build_multilabel_model

logger = logging.getLogger(__name__)

# ─── History columns ────────────────────────────────────────────────────────────
ML_HISTORY_COLS = [
    "epoch", "phase",
    "loss", "subset_acc", "macro_auc", "micro_auc",
    "macro_f1", "micro_f1", "hamming_loss",
    "lr", "epoch_time_s",
]


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for multi-label classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Applied element-wise on sigmoid outputs.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.reduction  = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        pt  = torch.exp(-bce)
        fl  = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl


# ─── Loss builder ────────────────────────────────────────────────────────────────

def _build_multilabel_loss(
    config: dict,
    train_dataset: Optional[MultiLabelMILDataset],
    device: torch.device,
) -> nn.Module:
    """Build BCEWithLogitsLoss or FocalLoss with optional per-label pos_weight."""
    train_cfg = config.get("multilabel_training", {})
    loss_name = train_cfg.get("loss", "bce").lower()
    ls        = float(train_cfg.get("label_smoothing", 0.0))

    pos_weight = None
    if train_cfg.get("weighted_loss", True) and train_dataset is not None:
        pos_weight = train_dataset.label_pos_weights().to(device)
        logger.info(f"  Per-label pos_weights: {pos_weight.cpu().tolist()}")

    if loss_name == "focal":
        alpha = float(train_cfg.get("focal_alpha", 0.25))
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        logger.info(f"  Loss: FocalLoss (alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma,
                         pos_weight=pos_weight).to(device)

    # Default: BCEWithLogitsLoss
    logger.info(f"  Loss: BCEWithLogitsLoss (label_smoothing={ls})")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)


# ─── Metrics ─────────────────────────────────────────────────────────────────────

def _compute_ml_metrics(
    all_probs: np.ndarray,          # (N, n_labels)
    all_preds: np.ndarray,          # (N, n_labels) binary
    all_labels: np.ndarray,         # (N, n_labels) binary
    label_names: List[str],
) -> Dict[str, float]:
    """Compute multi-label metrics. Returns dict of scalar floats."""
    metrics = {}

    # Subset accuracy (exact match ratio)
    metrics["subset_acc"] = float(accuracy_score(all_labels, all_preds))

    # Hamming loss
    metrics["hamming_loss"] = float(hamming_loss(all_labels, all_preds))

    # Macro / micro F1
    metrics["macro_f1"] = float(
        f1_score(all_labels, all_preds, average="macro", zero_division=0))
    metrics["micro_f1"] = float(
        f1_score(all_labels, all_preds, average="micro", zero_division=0))

    # Macro / micro AUC (robust to missing classes)
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(all_labels, all_probs, average="macro"))
    except Exception:
        metrics["macro_auc"] = 0.0
    try:
        metrics["micro_auc"] = float(
            roc_auc_score(all_labels, all_probs, average="micro"))
    except Exception:
        metrics["micro_auc"] = 0.0

    # Per-label AUC
    per_label = {}
    for i, lbl in enumerate(label_names):
        try:
            per_label[lbl] = float(
                roc_auc_score(all_labels[:, i], all_probs[:, i]))
        except Exception:
            per_label[lbl] = float("nan")
    metrics["per_label_auc"] = per_label

    return metrics


# ─── Single epoch ────────────────────────────────────────────────────────────────

def _run_ml_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn: nn.Module,
    n_labels: int,
    label_names: List[str],
    device: torch.device,
    is_train: bool,
    threshold: float = 0.5,
    patch_dropout: float = 0.0,
    patch_shuffle: bool = False,
) -> Tuple[float, Dict]:
    """Run one epoch. Returns (avg_loss, metrics_dict)."""
    model.train(is_train)
    total_loss  = 0.0
    all_probs   = []
    all_preds   = []
    all_labels  = []

    with torch.set_grad_enabled(is_train):
        for feats_list, label_vecs, _ in loader:
            for feats, label_vec in zip(feats_list, label_vecs):
                feats     = feats.to(device)              # (N, D)
                label_vec = label_vec.unsqueeze(0).to(device)  # (1, n_labels)

                # Bag-level regularisation (training only)
                if is_train:
                    if patch_shuffle:
                        idx   = torch.randperm(feats.size(0), device=device)
                        feats = feats[idx]
                    if patch_dropout > 0 and feats.size(0) > 4:
                        keep  = max(4, int(feats.size(0) * (1 - patch_dropout)))
                        idx   = torch.randperm(feats.size(0), device=device)[:keep]
                        feats = feats[idx]

                logits, probs, preds, _, _ = model(feats)  # (1,n_labels) each

                # Label smoothing on targets
                loss = loss_fn(logits, label_vec)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                all_probs.append(probs.detach().cpu().numpy()[0])    # (n_labels,)
                all_preds.append(preds.detach().cpu().numpy()[0])
                all_labels.append(label_vec.cpu().numpy()[0])

    avg_loss   = total_loss / max(len(all_labels), 1)
    all_probs  = np.array(all_probs)    # (N, n_labels)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = _compute_ml_metrics(all_probs, all_preds, all_labels, label_names)
    return avg_loss, metrics


# ─── History / checkpoints ───────────────────────────────────────────────────────

def _append_ml_history_row(exp_dir: str, row: dict):
    record   = {c: row.get(c, float("nan")) for c in ML_HISTORY_COLS}
    df_new   = pd.DataFrame([record])
    csv_path = os.path.join(exp_dir, "train_history.csv")
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode="w", header=True, index=False)


def _save_ml_checkpoint(path, model, optimizer, scheduler,
                         epoch, metrics, config, label_names,
                         mil_name, is_best):
    ckpt = {
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch"          : epoch,
        "metrics"        : metrics,
        "config"         : config,
        "label_names"    : label_names,
        "model_type"     : mil_name,
        "task_type"      : "multilabel",
        "threshold"      : config.get("multilabel", {}).get("threshold", 0.5),
        "timestamp"      : datetime.datetime.now().isoformat(timespec="seconds"),
        "is_best"        : is_best,
    }
    torch.save(ckpt, path)
    tag = "BEST" if is_best else "final"
    logger.info(f"  Checkpoint [{tag}] saved → {path}")


# ─── Training plots ───────────────────────────────────────────────────────────────

def _plot_ml_curves(exp_dir: str):
    csv_path = os.path.join(exp_dir, "train_history.csv")
    if not os.path.exists(csv_path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df       = pd.read_csv(csv_path)
    if df.empty:
        return

    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    train_df = df[df["phase"] == "train"]
    val_df   = df[df["phase"] == "val"]

    def _save(x_tr, y_tr, x_v, y_v, ylabel, title, fname, color_tr, color_v):
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(x_tr):
            ax.plot(x_tr, y_tr, label=f"Train {ylabel}", lw=1.5, color=color_tr)
        if len(x_v):
            ax.plot(x_v, y_v, label=f"Val {ylabel}",   lw=1.5,
                    color=color_v, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=120)
        plt.close(fig)

    _save(train_df["epoch"], train_df["loss"],
          val_df["epoch"],   val_df["loss"],
          "Loss", "Training & Validation Loss", "loss_curve.png",
          "steelblue", "tomato")

    if "macro_auc" in val_df.columns:
        _save(train_df["epoch"], train_df.get("macro_auc", []),
              val_df["epoch"],   val_df["macro_auc"],
              "Macro AUC", "Validation Macro AUC", "macro_auc_curve.png",
              "mediumseagreen", "mediumseagreen")

    if "macro_f1" in val_df.columns:
        _save(train_df["epoch"], train_df.get("macro_f1", []),
              val_df["epoch"],   val_df["macro_f1"],
              "Macro F1", "Validation Macro F1", "macro_f1_curve.png",
              "darkorchid", "darkorchid")

    if "hamming_loss" in val_df.columns:
        _save(train_df["epoch"], train_df.get("hamming_loss", []),
              val_df["epoch"],   val_df["hamming_loss"],
              "Hamming Loss", "Hamming Loss", "hamming_loss_curve.png",
              "coral", "coral")

    logger.info(f"  Training curves → {plots_dir}/")


# ─── Main command ─────────────────────────────────────────────────────────────────

def command_multilabel_train(config: dict, dirs_dict: dict, log=None):
    """
    Train a MIL model for multi-label classification.

    Results are saved to results/multilabel/experiments/<task>/<model>_<ts>/
    """
    _log = log or logger
    _seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info(f"[MultiLabel Train] Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    datasets, label_names = build_multilabel_datasets(config, dirs_dict)
    if "train" not in datasets:
        _log.error(
            "No train split found for multi-label task.\n"
            "  Run: python main.py multilabel-split --config ... --csv labels.csv")
        return

    n_labels  = len(label_names)
    train_cfg = config.get("multilabel_training", {})
    ml_cfg    = config.get("multilabel", {})
    nw        = int(config.get("training", {}).get("num_workers", 0))
    threshold = float(ml_cfg.get("threshold", 0.5))

    if len(datasets["train"]) == 0:
        _log.error("Training dataset has 0 valid bags. Check feature directory.")
        return

    train_loader = DataLoader(datasets["train"], batch_size=1, shuffle=True,
                              num_workers=nw,
                              collate_fn=multilabel_collate_fn)
    val_loader = None
    if "val" in datasets and len(datasets["val"]) > 0:
        val_loader = DataLoader(datasets["val"], batch_size=1, shuffle=False,
                                num_workers=nw,
                                collate_fn=multilabel_collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, n_params = build_multilabel_model(config)
    model.to(device)
    mil_name = config["mil"]["model"]

    _log.info(f"  Model   : {mil_name} | Params: {n_params:,}")
    _log.info(f"  Labels  : {n_labels} — {label_names}")

    # ── Optimiser & Scheduler ──────────────────────────────────────────────────
    lr         = float(train_cfg.get("learning_rate", 2e-4))
    wd         = float(train_cfg.get("weight_decay", 1e-4))
    b1         = float(config.get("training", {}).get("beta1", 0.9))
    b2         = float(config.get("training", {}).get("beta2", 0.999))
    opt_name   = train_cfg.get("optimizer", "AdamW")
    sched_name = train_cfg.get("lr_scheduler", "plateau")
    warmup_ep  = int(train_cfg.get("warmup_epochs", 0))
    max_ep     = int(train_cfg.get("max_epochs", 100))
    patch_dropout = float(train_cfg.get("patch_dropout", 0.0))
    patch_shuffle = bool(train_cfg.get("patch_shuffle", False))
    monitor    = train_cfg.get("monitor_metric", "macro_auc")

    if opt_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_ep, eta_min=1e-7)
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, max_ep // 3), gamma=0.3)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=10)

    loss_fn = _build_multilabel_loss(config, datasets.get("train"), device)

    # ── Early stopping ─────────────────────────────────────────────────────────
    do_es  = bool(train_cfg.get("early_stopping", True))
    es_pat = int(train_cfg.get("early_stopping_patience", 20))
    es_min = int(train_cfg.get("early_stopping_min_epochs", 10))

    _log.info(f"  Optimizer : {opt_name} | LR={lr:.2e} | WD={wd:.2e}")
    _log.info(f"  Scheduler : {sched_name} | max_epochs={max_ep}")
    _log.info(f"  Monitor   : {monitor} | ES patience={es_pat}")

    # ── Experiment directory ────────────────────────────────────────────────────
    stamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = ml_cfg.get("task_name") or config.get("task", {}).get("name", "multilabel")
    exp_dir   = os.path.join(
        config["paths"]["results_dir"],
        "multilabel", "experiments", task_name,
        f"{mil_name}_{stamp}")
    os.makedirs(exp_dir, exist_ok=True)
    _log.info(f"  Experiment dir: {exp_dir}")

    with open(os.path.join(exp_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    best_model_path = os.path.join(exp_dir, "best_model.pt")
    best_metric_val = -float("inf")    # we maximise monitor_metric
    best_epoch_metrics: dict = {}
    es_counter = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, max_ep + 1):
        t0 = time.time()

        # LR warmup
        if warmup_ep > 0 and epoch <= warmup_ep:
            scale = epoch / max(1, warmup_ep)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * scale

        tr_loss, tr_met = _run_ml_epoch(
            model, train_loader, optimizer, loss_fn,
            n_labels, label_names, device, is_train=True,
            threshold=threshold,
            patch_dropout=patch_dropout, patch_shuffle=patch_shuffle)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        _append_ml_history_row(exp_dir, {
            "epoch": epoch, "phase": "train",
            "loss": tr_loss, "subset_acc": tr_met["subset_acc"],
            "macro_auc": tr_met["macro_auc"], "micro_auc": tr_met["micro_auc"],
            "macro_f1": tr_met["macro_f1"], "micro_f1": tr_met["micro_f1"],
            "hamming_loss": tr_met["hamming_loss"],
            "lr": current_lr, "epoch_time_s": round(elapsed, 2),
        })

        # ── Validation ──────────────────────────────────────────────────────────
        if val_loader is not None:
            vl_loss, vl_met = _run_ml_epoch(
                model, val_loader, optimizer, loss_fn,
                n_labels, label_names, device, is_train=False,
                threshold=threshold)

            if sched_name == "plateau":
                scheduler.step(vl_loss)
            else:
                scheduler.step()

            _append_ml_history_row(exp_dir, {
                "epoch": epoch, "phase": "val",
                "loss": vl_loss, "subset_acc": vl_met["subset_acc"],
                "macro_auc": vl_met["macro_auc"], "micro_auc": vl_met["micro_auc"],
                "macro_f1": vl_met["macro_f1"],   "micro_f1": vl_met["micro_f1"],
                "hamming_loss": vl_met["hamming_loss"],
                "lr": current_lr,
            })

            _log.info(
                f"Epoch {epoch:>3}/{max_ep} | "
                f"tr_loss={tr_loss:.4f} mauc={tr_met['macro_auc']:.3f} | "
                f"val_loss={vl_loss:.4f} mauc={vl_met['macro_auc']:.3f} "
                f"mf1={vl_met['macro_f1']:.3f} hl={vl_met['hamming_loss']:.3f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s")

            current_metric = vl_met.get(monitor, vl_met["macro_auc"])
        else:
            if sched_name == "plateau":
                scheduler.step(tr_loss)
            else:
                scheduler.step()
            vl_met = tr_met
            current_metric = tr_met.get(monitor, tr_met["macro_auc"])
            _log.info(
                f"Epoch {epoch:>3}/{max_ep} | "
                f"tr_loss={tr_loss:.4f} mauc={tr_met['macro_auc']:.3f} "
                f"mf1={tr_met['macro_f1']:.3f} | lr={current_lr:.2e} | {elapsed:.1f}s")

        # ── Best checkpoint ─────────────────────────────────────────────────────
        if current_metric > best_metric_val:
            best_metric_val    = current_metric
            best_epoch_metrics = {"epoch": epoch,
                                  **{f"val_{k}": v
                                     for k, v in vl_met.items()
                                     if not isinstance(v, dict)}}
            es_counter = 0
            _save_ml_checkpoint(
                best_model_path, model, optimizer, scheduler,
                epoch=epoch, metrics=best_epoch_metrics,
                config=config, label_names=label_names,
                mil_name=mil_name, is_best=True)
        else:
            es_counter += 1
            if do_es and epoch >= es_min and es_counter >= es_pat:
                _log.info(f"Early stopping at epoch {epoch} (patience={es_pat})")
                break

    # ── Final checkpoint + plots ───────────────────────────────────────────────
    final_path = os.path.join(exp_dir, "final_model.pt")
    _save_ml_checkpoint(
        final_path, model, optimizer, scheduler,
        epoch=epoch,
        metrics={"epoch": epoch,
                 **{f"val_{k}": v for k, v in vl_met.items()
                    if not isinstance(v, dict)}},
        config=config, label_names=label_names,
        mil_name=mil_name, is_best=False)

    _plot_ml_curves(exp_dir)

    _log.info(f"MultiLabel Training complete.")
    _log.info(f"  Best {monitor}: {best_metric_val:.4f}")
    _log.info(f"  Best model : {best_model_path}")
    _log.info(f"  Experiment : {exp_dir}")

    return exp_dir
