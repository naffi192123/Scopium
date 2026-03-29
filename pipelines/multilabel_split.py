"""
pipelines/multilabel_split.py

Dataset splitting for Multi-Label Classification.

Supports:
  - train / test split
  - train / val / test split
  - Iterative stratification (scikit-multilearn) for multi-label CSVs
  - Fallback to random split if scikit-multilearn is not installed

CSV format auto-detected:
  1. Binary columns:   slide_id, ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM
  2. String column:    slide_id, labels  (where labels="ADI,LYM")

Output:
    results/multilabel/splits/<task_name>/
        train.csv
        val.csv   (train_val_test only)
        test.csv
        split_summary.txt

CLI:
    python main.py multilabel-split --config config/config.yaml --csv path/to/labels.csv
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from skmultilearn.model_selection import IterativeStratification
    _SKMULTILEARN = True
except ImportError:
    _SKMULTILEARN = False

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _auto_detect_slide_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for cand in ("slide_id", "slide", "case_id", "filename", "id"):
        if cand in cols:
            return df.columns[cols.index(cand)]
    return df.columns[0]


def _normalise_to_binary_matrix(
    df: pd.DataFrame,
    label_names: List[str],
    slide_col: str,
    labels_col: Optional[str],
    binary_columns: bool,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns (df_clean, label_matrix) where label_matrix is (N, n_labels) int.
    df_clean has slide_col plus one binary column per label.
    """
    if binary_columns:
        missing = [l for l in label_names if l not in df.columns]
        if missing:
            raise ValueError(
                f"Binary column mode: labels {missing} not found in CSV columns.\n"
                f"CSV columns: {list(df.columns)}")
        label_matrix = df[label_names].values.astype(int)
        df_clean     = df[[slide_col] + label_names].copy()
    else:
        # String column mode
        if labels_col is None:
            cols = [c.lower() for c in df.columns]
            for cand in ("labels", "label", "classes", "categories"):
                if cand in cols:
                    labels_col = df.columns[cols.index(cand)]
                    break
            if labels_col is None:
                labels_col = df.columns[-1]

        label_name_set = set(label_names)
        rows = []
        for _, row in df.iterrows():
            active = {s.strip() for s in str(row[labels_col]).split(",") if s.strip()}
            rows.append({lbl: (1 if lbl in active else 0) for lbl in label_names})

        label_df     = pd.DataFrame(rows)
        label_matrix = label_df.values.astype(int)
        df_clean     = pd.concat([df[[slide_col]].reset_index(drop=True),
                                   label_df.reset_index(drop=True)], axis=1)

    return df_clean, label_matrix


def _iterative_train_test_split(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split using iterative stratification (skmultilearn).

    IterativeStratification does not accept random_state directly.
    We pre-shuffle with numpy to achieve reproducibility.
    """
    if _SKMULTILEARN:
        # Pre-shuffle for reproducibility (IterativeStratification ignores random_state)
        rng       = np.random.RandomState(seed)
        n         = len(df)
        perm      = rng.permutation(n)
        df_shuf   = df.iloc[perm].reset_index(drop=True)
        lm_shuf   = label_matrix[perm]

        stratifier = IterativeStratification(
            n_splits=2,
            order=2 if lm_shuf.shape[1] > 1 else 1,
            sample_distribution_per_fold=[test_size, 1 - test_size],
        )
        # split() yields (train_indices, test_indices) per fold — sklearn convention.
        # With sample_distribution_per_fold=[test_size, 1-test_size]:
        #   folds[0][0] = large partition indices (train, 1-test_size)
        #   folds[0][1] = small partition indices (test,  test_size)
        folds = list(stratifier.split(np.zeros((n, 1)), lm_shuf))
        train_idxs = sorted(folds[0][0])   # large partition → train
        test_idxs  = sorted(folds[0][1])   # small partition → test
        return df_shuf.iloc[train_idxs].reset_index(drop=True), \
               df_shuf.iloc[test_idxs].reset_index(drop=True)
    else:
        logger.warning(
            "scikit-multilearn not installed — using random split.\n"
            "Install with: pip install scikit-multilearn")
        return train_test_split(df, test_size=test_size, random_state=seed)


# ─── Main command ─────────────────────────────────────────────────────────────────

def command_multilabel_split(
    config: dict,
    dirs_dict: dict,
    log=None,
    csv_path: Optional[str] = None,
):
    """
    Generate multi-label dataset splits from a label CSV.
    """
    _log = log or logger

    ml_cfg    = config.get("multilabel", {})
    split_cfg = config.get("multilabel_split", config.get("split", {}))

    label_names    = ml_cfg.get("label_names", [])
    binary_columns = ml_cfg.get("binary_columns", True)
    labels_col     = ml_cfg.get("labels_string_col", None)
    task_name      = ml_cfg.get("task_name") or config.get("task", {}).get("name", "multilabel")

    if not label_names:
        _log.error(
            "multilabel.label_names must be set in config.yaml before splitting.\n"
            "  Example: label_names: [ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM]")
        return

    # ── Load CSV ──────────────────────────────────────────────────────────────
    ann_dir = config["paths"].get("annotations_dir", "dataset/annotations")
    if csv_path is None:
        candidates = [f for f in os.listdir(ann_dir) if f.endswith(".csv")]
        if not candidates:
            _log.error(f"No CSV found in {ann_dir}. Pass --csv to specify one.")
            return
        csv_path = os.path.join(ann_dir, candidates[0])

    _log.info(f"[MultiLabel Split] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    _log.info(f"  Loaded {len(df)} rows | columns: {list(df.columns)}")

    slide_col = ml_cfg.get("slide_col") or _auto_detect_slide_col(df)
    _log.info(f"  Slide column: '{slide_col}'")

    # ── Normalise to binary matrix ────────────────────────────────────────────
    df_clean, label_matrix = _normalise_to_binary_matrix(
        df, label_names, slide_col, labels_col, binary_columns)

    # Drop slides with zero positives
    mask    = label_matrix.sum(axis=1) > 0
    n_drop  = (~mask).sum()
    if n_drop:
        _log.warning(f"  Dropping {n_drop} slides with no positive labels.")
        df_clean     = df_clean[mask].reset_index(drop=True)
        label_matrix = label_matrix[mask]

    _log.info(f"  After filtering: {len(df_clean)} slides")

    # ── Split config ──────────────────────────────────────────────────────────
    split_type  = split_cfg.get("type", "train_val_test")
    test_size   = float(split_cfg.get("test_size", 0.20))
    val_size    = float(split_cfg.get("val_size", 0.10))
    train_size  = float(split_cfg.get("train_size", 0.70))
    seed        = int(split_cfg.get("random_seed", 42))

    _log.info(f"  Split mode: {split_type} | test={test_size} | seed={seed}")
    if not _SKMULTILEARN:
        _log.warning("  scikit-multilearn not found — using random splits "
                     "(install for multi-label stratification)")

    # ── Mode 1: train / test ──────────────────────────────────────────────────
    if split_type == "train_test":
        df_train, df_test = _iterative_train_test_split(
            df_clean, label_matrix, test_size=test_size, seed=seed)
        splits = {"train": df_train, "test": df_test}

    # ── Mode 2: train / val / test ────────────────────────────────────────────
    elif split_type == "train_val_test":
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-4:
            train_size /= total; val_size /= total; test_size /= total

        # Step 1: peel test
        lm_all      = label_matrix
        df_trainval, df_test = _iterative_train_test_split(
            df_clean, lm_all, test_size=test_size, seed=seed)

        # Re-extract label matrix for trainval
        lm_tv = df_trainval[label_names].values.astype(int)
        val_relative = val_size / (train_size + val_size)
        df_train, df_val = _iterative_train_test_split(
            df_trainval, lm_tv, test_size=val_relative, seed=seed)
        splits = {"train": df_train, "val": df_val, "test": df_test}

    else:
        _log.error(f"Unknown split type '{split_type}'. Use: train_test | train_val_test")
        return

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    splits_dir = os.path.join(config["paths"]["results_dir"],
                               "multilabel", "splits", task_name)
    os.makedirs(splits_dir, exist_ok=True)

    summary_lines = [
        "Multi-Label Split Configuration",
        "--------------------------------",
        f"CSV source   : {csv_path}",
        f"Task         : {task_name}",
        f"Labels       : {label_names}",
        f"Mode         : {split_type}",
        f"Stratified   : {'IterativeStratification' if _SKMULTILEARN else 'random'}",
        f"Random seed  : {seed}",
        f"Total slides : {len(df_clean)}",
        "",
        f"{'Split':<10} {'Count':>6}  Label distribution",
        f"{'-'*60}",
    ]

    for split_name, split_df in splits.items():
        out_path = os.path.join(splits_dir, f"{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        pct   = 100 * len(split_df) / len(df_clean)
        lbl_d = {lbl: int(split_df[lbl].sum()) for lbl in label_names}
        dist_str = "  ".join(f"{k}:{v}" for k, v in lbl_d.items())
        summary_lines.append(
            f"{split_name:<10} {len(split_df):>6}  ({pct:.1f}%)   {dist_str}")
        _log.info(f"  {split_name:>5}: {len(split_df):>4} slides ({pct:.0f}%)  → {out_path}")
        _log.info(f"          Label counts: {lbl_d}")

    with open(os.path.join(splits_dir, "split_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    _log.info(f"\n  Split summary → {splits_dir}/split_summary.txt")
    _log.info(f"  All splits    → {splits_dir}/")
