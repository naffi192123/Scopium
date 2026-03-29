"""
datasets/mil_multilabel_dataset.py

Multi-Label MIL Bag Dataset — loads pre-extracted feature tensors and
multi-label binary vectors for each slide.

Supported CSV formats
---------------------
1. Binary columns (one column per label, 0/1 values):
       slide_id, ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM
       TCGA-XX, 1, 0, 1, 0, 0, 0, 1, 0

2. String column (one column with comma-separated active labels):
       slide_id, labels
       TCGA-XX, "ADI,LYM,STR"

The format is auto-detected from the CSV header.
Both formats output: feats (N, D), label_vec (n_labels,) float32, slide_id str.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Tuple, Dict

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultiLabelMILDataset(Dataset):
    """
    Multiple Instance Learning dataset for multi-label classification.

    Parameters
    ----------
    csv_path      : Path to split CSV.
    pt_dir        : Directory containing <slide_id>.pt feature files.
    label_names   : Ordered list of label names (defines index in multi-hot vector).
    slide_col     : Column name for slide IDs (auto-detect if None).
    labels_col    : Column name for the string labels column.
                    If None AND binary_columns=True, each label_name is expected
                    as a separate column in the CSV.
    binary_columns: If True, expects one 0/1 column per label name.
                    If False, expects a single string column (comma-separated).
    max_patches   : Maximum number of patches to load per slide (None = all).
    threshold     : Decision threshold for binarising sigmoid outputs (stored
                    in the dataset for downstream use, not used during loading).
    """

    def __init__(
        self,
        csv_path: str,
        pt_dir: str,
        label_names: List[str],
        slide_col: Optional[str] = None,
        labels_col: Optional[str] = None,
        binary_columns: bool = True,
        max_patches: Optional[int] = None,
        threshold: float = 0.5,
    ):
        self.pt_dir       = pt_dir
        self.label_names  = label_names
        self.n_labels     = len(label_names)
        self.max_patches  = max_patches
        self.threshold    = threshold
        self.binary_columns = binary_columns

        df = pd.read_csv(csv_path)
        cols_lower = [c.lower() for c in df.columns]

        # ── Auto-detect slide_id column ──────────────────────────────────────
        if slide_col is None:
            for cand in ("slide_id", "slide", "case_id", "filename", "id"):
                if cand in cols_lower:
                    slide_col = df.columns[cols_lower.index(cand)]
                    break
            if slide_col is None:
                slide_col = df.columns[0]

        # ── Detect format ────────────────────────────────────────────────────
        # If all label_names are column headers → binary_columns mode
        has_binary_cols = all(lbl in df.columns for lbl in label_names)
        if not has_binary_cols:
            binary_columns = False  # fall back to string mode

        self.binary_columns = binary_columns

        if binary_columns:
            # Validate that all label columns exist
            missing_cols = [l for l in label_names if l not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"MultiLabelMILDataset: label columns missing from CSV: {missing_cols}\n"
                    f"  CSV columns: {list(df.columns)}\n"
                    f"  Expected label columns: {label_names}")
        else:
            # String column — auto-detect if not given
            if labels_col is None:
                for cand in ("labels", "label", "classes", "categories"):
                    if cand in cols_lower:
                        labels_col = df.columns[cols_lower.index(cand)]
                        break
                if labels_col is None:
                    labels_col = df.columns[-1]
            self._labels_col = labels_col

        # ── Build slide list ─────────────────────────────────────────────────
        self.slide_ids  : List[str]             = []
        self.label_vecs : List[np.ndarray]      = []    # (n_labels,) float32 each
        self.pt_paths   : List[str]             = []

        missing_pt = 0
        invalid_label = 0

        for _, row in df.iterrows():
            sid = str(row[slide_col]).strip()
            pt  = os.path.join(pt_dir, sid + ".pt")
            if not os.path.exists(pt):
                missing_pt += 1
                continue

            # Parse label vector
            if binary_columns:
                vec = np.array([float(row[lbl]) for lbl in label_names],
                               dtype=np.float32)
            else:
                raw = str(row[self._labels_col]).strip()
                active = {s.strip() for s in raw.split(",") if s.strip()}
                unknown = active - set(label_names)
                if unknown:
                    logger.warning(
                        f"Slide {sid}: unknown labels {unknown} — treating as 0.")
                vec = np.array(
                    [1.0 if lbl in active else 0.0 for lbl in label_names],
                    dtype=np.float32)

            # Skip slides with no positive labels
            if vec.sum() == 0:
                logger.warning(
                    f"Slide {sid}: no positive labels — skipping.")
                invalid_label += 1
                continue

            self.slide_ids.append(sid)
            self.label_vecs.append(vec)
            self.pt_paths.append(pt)

        if missing_pt:
            logger.warning(
                f"MultiLabelMILDataset: {missing_pt} slides have no .pt file in {pt_dir}.")
        if invalid_label:
            logger.warning(
                f"MultiLabelMILDataset: {invalid_label} slides skipped (no positive labels).")

        logger.info(
            f"MultiLabelMILDataset loaded: {len(self)} slides | "
            f"labels={label_names} | "
            f"label_dist={self._label_dist()}")

    # ── Statistics ───────────────────────────────────────────────────────────

    def _label_dist(self) -> Dict[str, int]:
        if not self.label_vecs:
            return {}
        mat = np.stack(self.label_vecs)           # (N, n_labels)
        return {lbl: int(mat[:, i].sum())
                for i, lbl in enumerate(self.label_names)}

    def label_pos_weights(self) -> torch.Tensor:
        """
        Per-label positive weights for BCEWithLogitsLoss.
        pos_weight[i] = (# negative) / (# positive), clamped to [0.1, 20].
        """
        if not self.label_vecs:
            return torch.ones(self.n_labels)
        mat  = np.stack(self.label_vecs)               # (N, n_labels)
        n    = len(mat)
        pos  = mat.sum(axis=0).clip(min=1)
        neg  = (n - pos).clip(min=0)
        w    = neg / pos
        w    = np.clip(w, 0.1, 20.0)
        return torch.tensor(w, dtype=torch.float32)

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        feats = torch.load(self.pt_paths[idx], map_location="cpu",
                           weights_only=False)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)                 # (1, D) edge-case

        if self.max_patches and len(feats) > self.max_patches:
            perm  = torch.randperm(len(feats))[:self.max_patches]
            feats = feats[perm]

        label_vec = torch.tensor(self.label_vecs[idx], dtype=torch.float32)
        return feats, label_vec, self.slide_ids[idx]


# ─── Collate ──────────────────────────────────────────────────────────────────

def multilabel_collate_fn(batch):
    """
    Collate for multi-label MIL.
    Returns:
        features  : list of (N_i, D) tensors  (variable bag sizes)
        labels    : (B, n_labels) float32 tensor
        slide_ids : list of str
    """
    features  = [item[0] for item in batch]
    labels    = torch.stack([item[1] for item in batch])   # (B, n_labels)
    slide_ids = [item[2] for item in batch]
    return features, labels, slide_ids


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_multilabel_datasets(
    config: dict,
    dirs_dict: dict,
    pt_dir: Optional[str] = None,
    max_patches: Optional[int] = None,
) -> Tuple[Dict[str, MultiLabelMILDataset], List[str]]:
    """
    Build train / val (optional) / test MultiLabelMILDataset instances.

    Parameters
    ----------
    config    : Full parsed YAML config.
    dirs_dict : Directory dict from utils.config.setup_directories.
    pt_dir    : Explicit path to the pt_files directory.
                If None, resolved from dirs_dict['features']/pt_files.
    max_patches : Cap bag size. If None, uses multilabel_training.max_patches.

    Returns
    -------
    (dict of {split_name: MultiLabelMILDataset}, list of label_names)
    """
    ml_cfg    = config.get("multilabel", {})
    train_cfg = config.get("multilabel_training", {})
    paths_cfg = config.get("paths", {})
    task_cfg  = config.get("task", {})

    label_names = ml_cfg.get("label_names")
    if not label_names:
        raise ValueError(
            "multilabel.label_names must be set in config.yaml.\n"
            "  Example: label_names: [ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM]")

    task_name   = ml_cfg.get("task_name") or task_cfg.get("name", "multilabel_task")
    binary_cols = ml_cfg.get("binary_columns", True)
    labels_col  = ml_cfg.get("labels_string_col", None)
    threshold   = float(ml_cfg.get("threshold", 0.5))

    # ── Resolve pt_dir ────────────────────────────────────────────────────────
    if pt_dir is None:
        feat_dir = dirs_dict.get("features")
        if feat_dir:
            pt_dir = os.path.join(feat_dir, "pt_files")
        else:
            feat_cfg  = config.get("feature_extraction", {})
            model_key = feat_cfg.get("model", "rn50")
            p_size    = config["tiling"].get("patch_size", 512)
            s_size    = config["tiling"].get("step_size", 512)
            lvl       = config["tiling"].get("patch_level", 0)
            base_ovr  = feat_cfg.get("features_subfolder_override")
            feat_base = base_ovr or f"patch{p_size}_step{s_size}_level{lvl}"
            pt_dir    = os.path.join(
                paths_cfg["results_dir"], "features",
                f"{feat_base}__{model_key}", "pt_files")
            logger.warning(
                "build_multilabel_datasets: dirs_dict has no 'features' key — "
                f"falling back to auto-derived pt_dir: {pt_dir}")

    if not os.path.isdir(pt_dir):
        logger.error(
            f"Feature pt_files directory not found: {pt_dir}\n"
            "  Run `python main.py extract --config ...` first, or use "
            "--features to point to an existing feature set.")

    logger.info(f"MultiLabel feature source (pt_files): {pt_dir}")

    # ── Resolve max_patches ───────────────────────────────────────────────────
    if max_patches is None:
        max_patches = train_cfg.get("max_patches")

    # ── Build per-split datasets ──────────────────────────────────────────────
    splits_dir = os.path.join(paths_cfg["results_dir"],
                              "multilabel", "splits", task_name)

    datasets: Dict[str, MultiLabelMILDataset] = {}
    for split_name in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            continue
        datasets[split_name] = MultiLabelMILDataset(
            csv_path       = csv_path,
            pt_dir         = pt_dir,
            label_names    = label_names,
            binary_columns = binary_cols,
            labels_col     = labels_col,
            max_patches    = max_patches,
            threshold      = threshold,
        )

    return datasets, label_names
