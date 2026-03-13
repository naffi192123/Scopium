"""
pipelines/evaluate.py

MIL Evaluation Pipeline.

Usage via CLI:
    python main.py evaluate --config config/config.yaml
    python main.py evaluate --config config/config.yaml --experiment results/experiments/metastasis/abmil_20260313_010000

Outputs (inside experiment_dir/evaluation/test/):
    predictions.csv           - slide_id, true_label, pred_label, prob_<class>, ...
    metrics.json              - accuracy, AUC, per_class_auc, precision, recall, F1
    roc_data.npz              - fpr, tpr, thresholds per class (replot offline)
    confusion_matrix.csv
    classification_report.txt
    roc_curve.png

Also saves experiment_dir/test_results.csv (flat one-line summary).
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


def _save_evaluation_artifacts(exp_dir, split_name, slide_ids,
                                all_labels, all_preds, all_probs,
                                class_names, extra_metrics=None):
    """
    Save all evaluation artifacts to <exp_dir>/evaluation/<split_name>/.

    Artifacts
    ---------
    predictions.csv           — per-slide true/pred/prob columns
    metrics.json              — accuracy, auc, per_class_auc, precision, recall, f1
    roc_data.npz              — fpr, tpr, thresholds per class (binary-compressed)
    confusion_matrix.csv      — N×N with class-name axes
    classification_report.txt
    roc_curve.png
    """
    from datetime import datetime
    eval_dir  = os.path.join(exp_dir, 'evaluation', split_name)
    os.makedirs(eval_dir, exist_ok=True)
    n_classes = len(class_names)

    # ── 1. Predictions CSV ────────────────────────────────────────────────────
    pred_data = {
        'slide_id':   slide_ids,
        'true_label': [class_names[l] if l < n_classes else str(l)
                       for l in all_labels],
        'pred_label': [class_names[p] if p < n_classes else str(p)
                       for p in all_preds],
    }
    for c, cname in enumerate(class_names):
        safe = cname.replace(' ', '_').replace('/', '_')
        pred_data[f'prob_{safe}'] = all_probs[:, c].tolist()
    pd.DataFrame(pred_data).to_csv(
        os.path.join(eval_dir, 'predictions.csv'), index=False)

    # ── 2. ROC data + per-class AUC ───────────────────────────────────────────
    roc_arrays    = {}
    per_class_auc = []
    try:
        if n_classes == 2:
            fpr, tpr, thresh = roc_curve(all_labels, all_probs[:, 1])
            roc_arrays.update(fpr_0=fpr, tpr_0=tpr, thresh_0=thresh)
            auc_val = roc_auc_score(all_labels, all_probs[:, 1])
            per_class_auc = [auc_val]
        else:
            from sklearn.preprocessing import label_binarize
            lb = label_binarize(all_labels, classes=list(range(n_classes)))
            for c in range(n_classes):
                fpr, tpr, thresh = roc_curve(lb[:, c], all_probs[:, c])
                roc_arrays[f'fpr_{c}']    = fpr
                roc_arrays[f'tpr_{c}']    = tpr
                roc_arrays[f'thresh_{c}'] = thresh
                per_class_auc.append(
                    roc_auc_score(lb[:, c], all_probs[:, c]))
            auc_val = float(np.mean(per_class_auc))
    except Exception as e:
        logger.warning(f"  ROC computation failed: {e}")
        auc_val       = 0.0
        per_class_auc = [0.0] * n_classes

    # Save ROC arrays as binary-compressed npz (for offline replot)
    np.savez_compressed(os.path.join(eval_dir, 'roc_data.npz'), **roc_arrays)

    # ── 3. Metrics JSON ───────────────────────────────────────────────────────
    avg  = 'binary' if n_classes == 2 else 'macro'
    try:
        acc  = float(accuracy_score(all_labels, all_preds))
        bacc = float(balanced_accuracy_score(all_labels, all_preds))
        prec = float(precision_score(all_labels, all_preds,
                                     average=avg, zero_division=0))
        rec  = float(recall_score(all_labels, all_preds,
                                  average=avg, zero_division=0))
        f1   = float(f1_score(all_labels, all_preds,
                              average=avg, zero_division=0))
    except Exception:
        acc = bacc = prec = rec = f1 = 0.0

    metrics_dict = {
        'split':           split_name,
        'n_samples':       int(len(all_labels)),
        'accuracy':        round(acc,      4),
        'balanced_acc':    round(bacc,     4),
        'auc':             round(auc_val,  4),
        'precision':       round(prec,     4),
        'recall':          round(rec,      4),
        'f1':              round(f1,       4),
        'per_class_auc':   {class_names[c]: round(float(v), 4)
                            for c, v in enumerate(per_class_auc)},
        'class_names':     class_names,
        'timestamp':       datetime.now().isoformat(timespec='seconds'),
    }
    if extra_metrics:
        metrics_dict.update(extra_metrics)
    with open(os.path.join(eval_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    # ── 4. Confusion matrix CSV ───────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = 'true \\ pred'
    cm_df.to_csv(os.path.join(eval_dir, 'confusion_matrix.csv'))

    # ── 5. Classification report ──────────────────────────────────────────────
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, zero_division=0)
    with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Split: {split_name}\n")
        f.write(f"Timestamp: {metrics_dict['timestamp']}\n\n")
        f.write(report)
        f.write(f"\nAUC: {auc_val:.4f}\n")
        if n_classes > 2:
            for cname, cauc in metrics_dict['per_class_auc'].items():
                f.write(f"  AUC[{cname}]: {cauc:.4f}\n")

    # ── 6. ROC curve PNG ──────────────────────────────────────────────────────
    if roc_arrays:
        try:
            colours = plt.cm.tab10.colors
            fig, ax = plt.subplots(figsize=(7, 6))
            if n_classes == 2:
                fpr_ = roc_arrays.get('fpr_0', np.array([]))
                tpr_ = roc_arrays.get('tpr_0', np.array([]))
                if len(fpr_):
                    ax.plot(fpr_, tpr_, lw=2, color=colours[0],
                            label=f'{class_names[1 if n_classes > 1 else 0]} '
                                  f'(AUC={round(auc_val, 3)})')
            else:
                for c in range(n_classes):
                    fpr_ = roc_arrays.get(f'fpr_{c}', np.array([]))
                    tpr_ = roc_arrays.get(f'tpr_{c}', np.array([]))
                    if len(fpr_):
                        cauc = per_class_auc[c] if c < len(per_class_auc) else 0.0
                        ax.plot(fpr_, tpr_, lw=2, color=colours[c % 10],
                                label=f'{class_names[c]} (AUC={cauc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(eval_dir, 'roc_curve.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"  ROC plot failed: {e}")

    # Confusion matrix heatmap PNG
    try:
        fig, ax = plt.subplots(figsize=(max(4, n_classes * 1.5 + 1),
                                        max(4, n_classes * 1.5)))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(eval_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"  Confusion matrix plot failed: {e}")

    logger.info(f"  Evaluation artifacts → {eval_dir}/")
    logger.info(f"    predictions.csv | metrics.json | roc_data.npz | "
                f"confusion_matrix.csv | classification_report.txt")
    return eval_dir


def command_evaluate(config: dict, dirs_dict: dict, log=None,
                     experiment_dir: str = None):
    _log = log or logger

    # ── Find experiment ────────────────────────────────────────────────────────
    if experiment_dir is None:
        experiment_dir = _find_latest_experiment(config)
    if experiment_dir is None:
        _log.error("No experiment directory found. Train first: python main.py train")
        return

    best_ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    if not os.path.exists(best_ckpt_path):
        _log.error(f"best_model.pt not found in {experiment_dir}")
        return

    _log.info(f"{'─'*60}")
    _log.info(f"  EVALUATION")
    _log.info(f"{'─'*60}")
    _log.info(f"  Checkpoint: {best_ckpt_path}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(best_ckpt_path, map_location=device, weights_only=False)

    # Use config from checkpoint if available (guarantees consistency)
    ckpt_config = ckpt.get('config', config)
    class_names = ckpt.get('class_names', config['task'].get('class_names', []))
    n_classes   = len(class_names)
    model_type  = ckpt.get('model_type', ckpt_config['mil']['model'])

    model, _    = build_mil_model(ckpt_config)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    _log.info(f"  Model: {model_type} | Classes: {class_names} | Device: {device}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    datasets, _ = build_mil_datasets(ckpt_config, dirs_dict)
    if 'test' not in datasets:
        _log.error("No test.csv found. Run: python main.py split")
        return

    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=mil_collate_fn)

    # ── Inference ──────────────────────────────────────────────────────────────
    all_probs  = []
    all_preds  = []
    all_labels = []
    all_slides = []

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

    acc = float(accuracy_score(all_labels, all_preds))
    _log.info(f"  Test slides: {len(all_labels)} | Accuracy: {acc:.4f}")

    # ── Save full artifacts to evaluation/test/ ───────────────────────────────
    eval_dir = _save_evaluation_artifacts(
        exp_dir      = experiment_dir,
        split_name   = 'test',
        slide_ids    = all_slides,
        all_labels   = all_labels,
        all_preds    = all_preds,
        all_probs    = all_probs,
        class_names  = class_names,
        extra_metrics = {
            'checkpoint': best_ckpt_path,
            'model_type': model_type,
        },
    )

    # ── Flat one-line summary at experiment root ───────────────────────────────
    metrics_path = os.path.join(eval_dir, 'metrics.json')
    with open(metrics_path) as f:
        m = json.load(f)

    flat = {k: m[k] for k in ('accuracy', 'balanced_acc', 'auc',
                               'precision', 'recall', 'f1', 'n_samples')
            if k in m}
    flat['model_type'] = model_type
    flat['checkpoint'] = best_ckpt_path
    out_csv = os.path.join(experiment_dir, 'test_results.csv')
    pd.DataFrame([flat]).to_csv(out_csv, index=False)

    _log.info(f"  Summary CSV: {out_csv}")
    _log.info(f"  Full output: {eval_dir}/")
    _log.info(f"\n  Replot ROC offline:")
    _log.info(f"    import numpy as np")
    _log.info(f"    d = np.load('{eval_dir}/roc_data.npz')")
    _log.info(f"    import matplotlib.pyplot as plt")
    _log.info(f"    plt.plot(d['fpr_0'], d['tpr_0'])")

    return eval_dir
