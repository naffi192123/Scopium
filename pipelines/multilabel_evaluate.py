"""
pipelines/multilabel_evaluate.py

MIL Multi-Label Evaluation Pipeline.

CLI:
    python main.py multilabel-evaluate --config config/config.yaml
    python main.py multilabel-evaluate --config config/config.yaml --experiment <dir>

Outputs (inside experiment_dir/evaluation/test/):
    predictions.csv         — slide_id, true_<label>, ..., pred_<label>, ..., prob_<label>
    per_label_metrics.csv   — per-label AUC, precision, recall, F1, support
    metrics.json            — aggregate metrics (macro/micro AUC, F1, hamming, subset_acc)
    roc_curves/             — per-label ROC PNG
    confusion/              — per-label confusion matrix PNG (TP/FP/TN/FN)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score, roc_curve,
)
import pandas as pd

from datasets.mil_multilabel_dataset import (
    build_multilabel_datasets, multilabel_collate_fn,
)
from models.mil_multilabel_models import build_multilabel_model

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _find_latest_ml_experiment(config: dict) -> Optional[str]:
    """Return path to the most recently created multilabel experiment dir."""
    ml_cfg    = config.get("multilabel", {})
    task_name = ml_cfg.get("task_name") or config.get("task", {}).get("name", "multilabel")
    exp_root  = os.path.join(
        config["paths"]["results_dir"], "multilabel", "experiments", task_name)
    if not os.path.isdir(exp_root):
        return None
    dirs = sorted(
        [d for d in os.listdir(exp_root)
         if os.path.isdir(os.path.join(exp_root, d))],
        reverse=True)
    return os.path.join(exp_root, dirs[0]) if dirs else None


# ─── Inference loop ───────────────────────────────────────────────────────────────

def _run_inference(model, loader, device, threshold):
    """Run inference. Returns (slide_ids, all_probs, all_preds, all_labels)."""
    model.eval()
    all_probs  = []
    all_preds  = []
    all_labels = []
    slide_ids  = []

    with torch.no_grad():
        for feats_list, label_vecs, sids in loader:
            for feats, lv, sid in zip(feats_list, label_vecs, sids):
                feats = feats.to(device)
                logits, probs, preds, _, _ = model(feats)
                all_probs.append(probs.cpu().numpy()[0])
                all_preds.append(preds.cpu().numpy()[0])
                all_labels.append(lv.numpy())
                slide_ids.append(sid)

    return (slide_ids,
            np.array(all_probs),
            np.array(all_preds),
            np.array(all_labels))


# ─── Evaluation artifacts ─────────────────────────────────────────────────────────

def _save_ml_evaluation(
    exp_dir: str,
    split_name: str,
    slide_ids: List[str],
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    label_names: List[str],
):
    """Save all evaluation artefacts for a given split."""
    eval_dir = os.path.join(exp_dir, "evaluation", split_name)
    os.makedirs(eval_dir, exist_ok=True)
    n_labels = len(label_names)

    # ── predictions.csv ────────────────────────────────────────────────────────
    pred_data: Dict = {"slide_id": slide_ids}
    for i, lbl in enumerate(label_names):
        pred_data[f"true_{lbl}"]  = all_labels[:, i].astype(int)
    for i, lbl in enumerate(label_names):
        pred_data[f"pred_{lbl}"]  = all_preds[:, i].astype(int)
    for i, lbl in enumerate(label_names):
        pred_data[f"prob_{lbl}"]  = np.round(all_probs[:, i], 4)

    pd.DataFrame(pred_data).to_csv(
        os.path.join(eval_dir, "predictions.csv"), index=False)

    # ── Aggregate metrics ───────────────────────────────────────────────────────
    metrics: Dict = {}
    metrics["n_slides"]    = int(len(slide_ids))
    metrics["n_labels"]    = int(n_labels)
    metrics["subset_acc"]  = float(accuracy_score(all_labels, all_preds))
    metrics["hamming_loss"]= float(hamming_loss(all_labels, all_preds))
    metrics["macro_f1"]    = float(f1_score(all_labels, all_preds,
                                             average="macro", zero_division=0))
    metrics["micro_f1"]    = float(f1_score(all_labels, all_preds,
                                             average="micro", zero_division=0))
    metrics["macro_precision"] = float(precision_score(
        all_labels, all_preds, average="macro", zero_division=0))
    metrics["micro_precision"] = float(precision_score(
        all_labels, all_preds, average="micro", zero_division=0))
    metrics["macro_recall"]    = float(recall_score(
        all_labels, all_preds, average="macro", zero_division=0))
    metrics["micro_recall"]    = float(recall_score(
        all_labels, all_preds, average="micro", zero_division=0))
    try:
        metrics["macro_auc"] = float(
            roc_auc_score(all_labels, all_probs, average="macro"))
    except Exception:
        metrics["macro_auc"] = None
    try:
        metrics["micro_auc"] = float(
            roc_auc_score(all_labels, all_probs, average="micro"))
    except Exception:
        metrics["micro_auc"] = None

    # ── Per-label metrics ───────────────────────────────────────────────────────
    per_label_rows = []
    for i, lbl in enumerate(label_names):
        row = {"label": lbl}
        row["support"] = int(all_labels[:, i].sum())
        row["precision"] = float(precision_score(
            all_labels[:, i], all_preds[:, i], zero_division=0))
        row["recall"]    = float(recall_score(
            all_labels[:, i], all_preds[:, i], zero_division=0))
        row["f1"]        = float(f1_score(
            all_labels[:, i], all_preds[:, i], zero_division=0))
        try:
            row["auc"] = float(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        except Exception:
            row["auc"] = float("nan")
        per_label_rows.append(row)

    pl_df = pd.DataFrame(per_label_rows)
    pl_df.to_csv(os.path.join(eval_dir, "per_label_metrics.csv"), index=False)
    metrics["per_label"] = {r["label"]: {k: v for k, v in r.items()
                                          if k != "label"}
                             for r in per_label_rows}

    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Per-label ROC curves ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        roc_dir = os.path.join(eval_dir, "roc_curves")
        os.makedirs(roc_dir, exist_ok=True)

        for i, lbl in enumerate(label_names):
            if all_labels[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
            auc_val      = metrics["per_label"][lbl].get("auc", float("nan"))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, lw=2,
                    label=f"AUC = {auc_val:.3f}" if not np.isnan(auc_val) else "AUC = N/A",
                    color="steelblue")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC — {lbl}")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(roc_dir, f"roc_{lbl}.png"), dpi=100)
            plt.close(fig)

    except Exception as e:
        logger.warning(f"ROC curve plotting failed: {e}")

    return metrics


# ─── Main command ─────────────────────────────────────────────────────────────────

def command_multilabel_evaluate(
    config: dict,
    dirs_dict: dict,
    log=None,
    experiment_dir: Optional[str] = None,
    split: str = "test",
):
    """
    Evaluate a multi-label MIL model on the test (or val) split.
    """
    _log = log or logger

    # ── Find experiment ──────────────────────────────────────────────────────
    if experiment_dir is None:
        experiment_dir = _find_latest_ml_experiment(config)
    if experiment_dir is None or not os.path.isdir(experiment_dir):
        _log.error("No experiment directory found. Train a model first.")
        return

    best_ckpt = os.path.join(experiment_dir, "best_model.pt")
    if not os.path.exists(best_ckpt):
        _log.error(f"No best_model.pt found in {experiment_dir}")
        return

    _log.info(f"[MultiLabel Evaluate] Experiment: {experiment_dir}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    label_names = ckpt.get("label_names",
                           config.get("multilabel", {}).get("label_names", []))
    threshold   = ckpt.get("threshold",
                           float(config.get("multilabel", {}).get("threshold", 0.5)))

    # Patch config with checkpoint label info
    config_ckpt = ckpt.get("config", config)
    if label_names:
        config_ckpt.setdefault("multilabel", {})["label_names"] = label_names
    if threshold:
        config_ckpt.setdefault("multilabel", {})["threshold"] = threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_multilabel_model(config_ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    _log.info(f"  Model     : {ckpt.get('model_type', '?')}")
    _log.info(f"  Labels    : {label_names}")
    _log.info(f"  Threshold : {threshold}")
    _log.info(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    datasets, _ = build_multilabel_datasets(config_ckpt, dirs_dict)
    if split not in datasets:
        _log.error(f"No '{split}' split CSV found.")
        return

    nw     = int(config.get("training", {}).get("num_workers", 0))
    loader = DataLoader(datasets[split], batch_size=1, shuffle=False,
                        num_workers=nw, collate_fn=multilabel_collate_fn)

    # ── Inference ────────────────────────────────────────────────────────────
    _log.info(f"  Running inference on '{split}' ({len(datasets[split])} slides)...")
    slide_ids, all_probs, all_preds, all_labels = _run_inference(
        model, loader, device, threshold)

    # ── Save artefacts ────────────────────────────────────────────────────────
    metrics = _save_ml_evaluation(
        experiment_dir, split,
        slide_ids, all_labels, all_preds, all_probs, label_names)

    _log.info(f"\n  ── {split.upper()} RESULTS ──────────────────────────────")
    _log.info(f"  Subset Accuracy : {metrics.get('subset_acc', 0):.4f}")
    _log.info(f"  Hamming Loss    : {metrics.get('hamming_loss', 0):.4f}")
    _log.info(f"  Macro AUC       : {metrics.get('macro_auc', 0) or 0:.4f}")
    _log.info(f"  Macro F1        : {metrics.get('macro_f1', 0):.4f}")
    _log.info(f"  Micro F1        : {metrics.get('micro_f1', 0):.4f}")
    _log.info(f"\n  ── Per-Label AUC ───────────────────────────────────────")
    for lbl, lbl_met in metrics.get("per_label", {}).items():
        auc = lbl_met.get("auc", float("nan"))
        f1  = lbl_met.get("f1", 0.0)
        sup = lbl_met.get("support", 0)
        _log.info(f"    {lbl:15s} AUC={auc:.3f}  F1={f1:.3f}  support={sup}")

    out_dir = os.path.join(experiment_dir, "evaluation", split)
    _log.info(f"\n  Results → {out_dir}")
    return metrics
