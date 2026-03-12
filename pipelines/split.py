"""
pipelines/split.py

Dataset splitting pipeline.

Modes
-----
train_test       : splits into train / test
train_val_test   : splits into train / val / test

All modes support optional stratification by class label.

Config example
--------------
split:
  type: train_test
  train_size: 0.8
  test_size: 0.2
  stratified: true
  random_seed: 42

or

split:
  type: train_val_test
  train_size: 0.7
  val_size: 0.1
  test_size: 0.2
  stratified: true
  random_seed: 42

Output
------
results/splits/<task_name>/
    train.csv
    test.csv
    val.csv          (train_val_test only)
    split_summary.txt
"""

import os
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _auto_detect_columns(df: pd.DataFrame):
    """Return (slide_col, label_col) by scanning column names."""
    cols = [c.lower() for c in df.columns]

    slide_col = None
    for cand in ("slide_id", "slide", "case_id", "filename", "id"):
        if cand in cols:
            slide_col = df.columns[cols.index(cand)]
            break
    if slide_col is None:
        slide_col = df.columns[0]

    label_col = None
    for cand in ("label", "class", "target", "diagnosis", "subtype"):
        if cand in cols:
            label_col = df.columns[cols.index(cand)]
            break
    if label_col is None:
        label_col = df.columns[-1]

    return slide_col, label_col


def command_split(config: dict, dirs_dict: dict, log=None, csv_path: str = None):
    """
    Generate dataset splits from an annotation CSV and save to results/splits/.

    Parameters
    ----------
    config      : full parsed YAML config
    dirs_dict   : directory dict from setup_directories
    log         : logger instance
    csv_path    : override annotation CSV path (auto-detect if None)
    """
    _log = log or logger

    # ── Load annotation CSV ──────────────────────────────────────────────────
    ann_dir = config["paths"].get("annotations_dir", "dataset/annotations")

    if csv_path is None:
        candidates = [f for f in os.listdir(ann_dir) if f.endswith(".csv")]
        if not candidates:
            _log.error(f"No CSV found in {ann_dir}. Pass --csv to specify one.")
            return
        csv_path = os.path.join(ann_dir, candidates[0])

    _log.info(f"Loading annotation CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    slide_col, label_col = _auto_detect_columns(df)
    _log.info(f"Slide column: '{slide_col}'  |  Label column: '{label_col}'")

    # Drop rows with missing labels
    before = len(df)
    df = df.dropna(subset=[label_col]).copy()
    if len(df) < before:
        _log.warning(f"Dropped {before - len(df)} rows with missing labels.")

    # ── Split config ─────────────────────────────────────────────────────────
    split_cfg   = config.get("split", {})
    split_type  = split_cfg.get("type", "train_test")
    test_size   = float(split_cfg.get("test_size",  0.20))
    val_size    = float(split_cfg.get("val_size",   0.10))
    train_size  = float(split_cfg.get("train_size", 0.80))
    stratified  = bool(split_cfg.get("stratified", True))
    seed        = int(split_cfg.get("random_seed", 42))

    strat_col = df[label_col] if stratified else None

    _log.info(
        f"Split mode: {split_type}  |  stratified={stratified}  |  seed={seed}")

    # ── Output directory ─────────────────────────────────────────────────────
    task_name  = config.get("task", {}).get("name", "task")
    splits_dir = os.path.join(config["paths"]["results_dir"], "splits", task_name)
    os.makedirs(splits_dir, exist_ok=True)

    # ── Mode 1: train / test ─────────────────────────────────────────────────
    if split_type == "train_test":
        # Validate
        if abs(train_size + test_size - 1.0) > 1e-6:
            _log.warning(
                f"train_size ({train_size}) + test_size ({test_size}) != 1.0. "
                f"Using test_size={test_size} and deriving train from remainder.")

        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=strat_col,
        )

        splits = {"train": df_train, "test": df_test}

    # ── Mode 2: train / val / test ────────────────────────────────────────────
    elif split_type == "train_val_test":
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-4:
            _log.warning(
                f"Split sizes sum to {total:.3f}, not 1.0. "
                f"Normalising automatically.")
            train_size /= total
            val_size   /= total
            test_size  /= total

        # Step 1: peel off test set
        df_trainval, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[label_col] if stratified else None,
        )

        # Step 2: split trainval into train and val
        val_relative = val_size / (train_size + val_size)
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=val_relative,
            random_state=seed,
            stratify=df_trainval[label_col] if stratified else None,
        )

        splits = {"train": df_train, "val": df_val, "test": df_test}

    else:
        _log.error(f"Unknown split type '{split_type}'. Choose: train_test | train_val_test")
        return

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    summary_lines = [
        f"Split configuration",
        f"-------------------",
        f"CSV source   : {csv_path}",
        f"Task         : {task_name}",
        f"Mode         : {split_type}",
        f"Stratified   : {stratified}",
        f"Random seed  : {seed}",
        f"Total slides : {len(df)}",
        f"",
        f"{'Split':<10} {'Count':>6}  Class distribution",
        f"{'-'*50}",
    ]

    for split_name, split_df in splits.items():
        out_path = os.path.join(splits_dir, f"{split_name}.csv")
        split_df.to_csv(out_path, index=False)

        dist = split_df[label_col].value_counts().to_dict()
        dist_str = "  ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        pct = 100 * len(split_df) / len(df)
        summary_lines.append(
            f"{split_name:<10} {len(split_df):>6}  ({pct:.1f}%)   {dist_str}")

        _log.info(
            f"  {split_name:>5}: {len(split_df):>4} slides  ({pct:.0f}%)  "
            f"-> {out_path}  |  {dist_str}")

    # ── Save summary text ─────────────────────────────────────────────────────
    summary_path = os.path.join(splits_dir, "split_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    _log.info(f"Split summary saved to: {summary_path}")
    _log.info(f"All splits saved to: {splits_dir}")
