"""
pipelines/evaluate.py

MIL Evaluation Pipeline.

Usage via CLI:
    python main.py evaluate --config config/config.yaml
    python main.py evaluate --config config/config.yaml --experiment results/experiments/metastasis/abmil_20260313_010000

Outputs (inside experiment_dir/evaluate/):
    predictions.csv           - slide_id, true_label, pred_label, prob_class0, ...
    roc_data.csv              - fpr, tpr, thresholds (for binary tasks)
    confusion_matrix.csv
    classification_report.txt
    metrics.json
    roc_curve.png
    confusion_matrix.png
"""

import os
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, balanced_accuracy_score,
    roc_curve, confusion_matrix, classification_report,
)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.mil_dataset import build_mil_datasets, mil_collate_fn
from models.mil_models import build_mil_model

logger = logging.getLogger(__name__)


def _find_latest_experiment(config):
    """Return path to the most recently created experiment dir."""
    task_name   = config['task']['name']
    results_dir = config['paths']['results_dir']
    exp_root    = os.path.join(results_dir, 'experiments', task_name)
    if not os.path.isdir(exp_root):
        return None
    dirs = sorted([d for d in os.listdir(exp_root)
                   if os.path.isdir(os.path.join(exp_root, d))],
                  reverse=True)
    return os.path.join(exp_root, dirs[0]) if dirs else None


def command_evaluate(config: dict, dirs_dict: dict, log=None,
                     experiment_dir: str = None):
    _log = log or logger

    # ── Find experiment ───────────────────────────────────────────────────────
    if experiment_dir is None:
        experiment_dir = _find_latest_experiment(config)
    if experiment_dir is None:
        _log.error("No experiment directory found. Train first: python main.py train")
        return

    best_ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    if not os.path.exists(best_ckpt_path):
        _log.error(f"best_model.pt not found in {experiment_dir}")
        return

    _log.info(f"Evaluating checkpoint: {best_ckpt_path}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    # Use config from checkpoint if available
    ckpt_config  = ckpt.get('config', config)
    class_names  = ckpt.get('class_names', config['task'].get('class_names', []))
    n_classes    = len(class_names)

    model, _     = build_mil_model(ckpt_config)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    datasets, _ = build_mil_datasets(ckpt_config, dirs_dict)
    if 'test' not in datasets:
        _log.error("No test.csv found. Run: python main.py split")
        return

    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=mil_collate_fn)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_probs   = []
    all_preds   = []
    all_labels  = []
    all_slides  = []

    with torch.no_grad():
        for feats_list, labels, slide_ids in test_loader:
            for feats, label, sid in zip(feats_list, labels, slide_ids):
                feats = feats.to(device)
                logits, Y_prob, Y_hat, _, _ = model(feats)
                all_probs.append(Y_prob.cpu().numpy()[0])
                all_preds.append(Y_hat.cpu().item())
                all_labels.append(label.item())
                all_slides.append(sid)

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
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
        auc = (roc_auc_score(all_labels, all_probs[:, 1]) if n_classes == 2
               else roc_auc_score(all_labels, all_probs, multi_class='ovr'))
    except Exception:
        auc = 0.0

    metrics = dict(accuracy=acc, balanced_accuracy=bacc, f1=f1,
                   precision=prec, recall=rec, roc_auc=auc,
                   n_samples=int(len(all_labels)))

    _log.info(f"  acc={acc:.4f}  bal_acc={bacc:.4f}  f1={f1:.4f}  "
              f"prec={prec:.4f}  rec={rec:.4f}  auc={auc:.4f}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(experiment_dir, 'evaluate')
    os.makedirs(out_dir, exist_ok=True)

    # predictions.csv
    pred_rows = {'slide_id': all_slides,
                 'true_label': [class_names[int(l)] for l in all_labels],
                 'pred_label': [class_names[int(p)] for p in all_preds]}
    for c in range(n_classes):
        pred_rows[f'prob_{class_names[c]}'] = all_probs[:, c]
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(out_dir, 'predictions.csv'), index=False)

    # confusion_matrix.csv
    cm = confusion_matrix(all_labels, all_preds)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(out_dir, 'confusion_matrix.csv'))

    # classification_report.txt
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, zero_division=0)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # metrics.json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── ROC curve (binary) ────────────────────────────────────────────────────
    if n_classes == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}).to_csv(
            os.path.join(out_dir, 'roc_data.csv'), index=False)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=150)
        plt.close(fig)

    # confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(max(4, n_classes * 1.5 + 1),
                                    max(4, n_classes * 1.5)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)

    _log.info(f"Evaluation outputs saved to: {out_dir}")
    return out_dir
