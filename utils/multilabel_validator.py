"""
utils/multilabel_validator.py

Validation utilities for Multi-Label MIL workflows.

Validates:
  - Label CSV format (binary columns or string column)
  - All label names present in config
  - No slides with all-zero labels
  - Feature file availability for each slide in the CSV
  - Consistency between split CSVs and feature files

Usage:
    from utils.multilabel_validator import validate_multilabel_csv

    report = validate_multilabel_csv(
        csv_path    = "dataset/annotations/labels.csv",
        label_names = ["ADI", "DEB", "LYM", "MUC"],
        pt_dir      = "results/features/.../pt_files",
    )
    if not report["valid"]:
        print(report["errors"])
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_multilabel_csv(
    csv_path: str,
    label_names: List[str],
    pt_dir: Optional[str] = None,
    slide_col: Optional[str] = None,
    labels_col: Optional[str] = None,
    binary_columns: bool = True,
    log=None,
) -> Dict:
    """
    Validate a multi-label label CSV and optionally check feature file availability.

    Parameters
    ----------
    csv_path      : Path to the CSV file.
    label_names   : Expected label column names.
    pt_dir        : Directory of .pt feature files. If given, checks file existence.
    slide_col     : Slide ID column name (auto-detect if None).
    labels_col    : String labels column (used when binary_columns=False).
    binary_columns: If True, check for one column per label name.
    log           : Logger instance.

    Returns
    -------
    dict with keys:
        valid       : bool   — overall validity
        n_total     : int    — total rows in CSV
        n_valid     : int    — rows with >=1 positive label + feature file
        n_missing_pt: int    — rows with no .pt file (only if pt_dir given)
        n_zero_label: int    — rows with all-zero labels
        label_counts: dict   — {label: count_positive}
        warnings    : list of str
        errors      : list of str
    """
    _log   = log or logger
    errors : List[str] = []
    warnings: List[str] = []

    if not os.path.exists(csv_path):
        return {"valid": False, "errors": [f"CSV not found: {csv_path}"],
                "warnings": [], "n_total": 0, "n_valid": 0}

    df = pd.read_csv(csv_path)
    n_total = len(df)
    cols_lower = [c.lower() for c in df.columns]

    # ── Detect slide column ───────────────────────────────────────────────────
    if slide_col is None:
        for cand in ("slide_id", "slide", "case_id", "filename", "id"):
            if cand in cols_lower:
                slide_col = df.columns[cols_lower.index(cand)]
                break
        if slide_col is None:
            slide_col = df.columns[0]
            warnings.append(f"Slide column auto-selected as '{slide_col}'.")

    if slide_col not in df.columns:
        errors.append(f"Slide column '{slide_col}' not found in CSV.")
        return {"valid": False, "errors": errors, "warnings": warnings,
                "n_total": n_total, "n_valid": 0}

    # ── Check label columns ───────────────────────────────────────────────────
    if binary_columns:
        missing_cols = [l for l in label_names if l not in df.columns]
        if missing_cols:
            errors.append(
                f"Missing binary label columns: {missing_cols}.\n"
                f"  CSV columns: {list(df.columns)}\n"
                f"  Set multilabel.binary_columns: false to use string-column mode instead.")
        extra_cols = []
        label_matrix = None
        if not missing_cols:
            label_matrix = df[label_names].fillna(0).values.astype(int)
    else:
        if labels_col is None:
            for cand in ("labels", "label", "classes", "categories"):
                if cand in cols_lower:
                    labels_col = df.columns[cols_lower.index(cand)]
                    break
        if labels_col is None or labels_col not in df.columns:
            errors.append(
                f"String-label column '{labels_col}' not found. "
                "Set multilabel.labels_string_col to the correct column name.")
            label_matrix = None
        else:
            rows = []
            for raw in df[labels_col].astype(str):
                active = {s.strip() for s in raw.split(",") if s.strip()}
                unknown = active - set(label_names)
                if unknown:
                    warnings.append(f"Unknown labels found: {unknown}")
                rows.append([1 if lbl in active else 0 for lbl in label_names])
            label_matrix = np.array(rows, dtype=int)

    # ── Per-row validation ────────────────────────────────────────────────────
    n_zero_label = 0
    n_missing_pt = 0
    n_valid      = 0

    slide_ids = df[slide_col].astype(str).str.strip().tolist()

    if label_matrix is not None:
        for i, sid in enumerate(slide_ids):
            row_labels = label_matrix[i]
            if row_labels.sum() == 0:
                n_zero_label += 1
                continue

            if pt_dir is not None:
                pt_path = os.path.join(pt_dir, sid + ".pt")
                if not os.path.exists(pt_path):
                    n_missing_pt += 1
                    continue

            n_valid += 1

    # ── Label distribution ────────────────────────────────────────────────────
    label_counts: Dict[str, int] = {}
    if label_matrix is not None:
        for i, lbl in enumerate(label_names):
            label_counts[lbl] = int(label_matrix[:, i].sum())

    # ── Build warnings ────────────────────────────────────────────────────────
    if n_zero_label > 0:
        warnings.append(
            f"{n_zero_label} slides have no positive labels — they will be skipped.")

    if n_missing_pt > 0:
        warnings.append(
            f"{n_missing_pt} slides have no .pt feature file in {pt_dir} — they will be skipped.")

    # Check label imbalance
    if label_counts:
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        if max_count > 0 and min_count / max_count < 0.2:
            warnings.append(
                f"Severe label imbalance detected: "
                f"min={min_count} / max={max_count}.\n"
                "  Enable weighted_loss: true in multilabel_training config.")

    valid = len(errors) == 0 and n_valid > 0

    report = {
        "valid"       : valid,
        "n_total"     : n_total,
        "n_valid"     : n_valid,
        "n_zero_label": n_zero_label,
        "n_missing_pt": n_missing_pt,
        "label_counts": label_counts,
        "warnings"    : warnings,
        "errors"      : errors,
        "slide_col"   : slide_col,
        "labels_col"  : labels_col,
        "binary_columns": binary_columns,
    }

    # Print summary
    _log.info(f"MultiLabel CSV Validation: {csv_path}")
    _log.info(f"  Total slides  : {n_total}")
    _log.info(f"  Valid slides  : {n_valid}")
    _log.info(f"  Zero-label    : {n_zero_label}")
    if pt_dir:
        _log.info(f"  Missing .pt   : {n_missing_pt}")
    _log.info(f"  Label counts  : {label_counts}")
    for w in warnings:
        _log.warning(f"  ⚠ {w}")
    for e in errors:
        _log.error(f"  ✗ {e}")
    if valid:
        _log.info("  ✓ Validation PASSED")
    else:
        _log.error("  ✗ Validation FAILED")

    return report


def command_multilabel_validate(config: dict, dirs_dict: dict,
                                 csv_path=None, log=None) -> bool:
    """CLI entry-point: validate labels CSV and feature file availability."""
    _log  = log or logger
    ml_cfg = config.get("multilabel", {})

    label_names    = ml_cfg.get("label_names", [])
    binary_columns = ml_cfg.get("binary_columns", True)
    labels_col     = ml_cfg.get("labels_string_col")

    if not label_names:
        _log.error("multilabel.label_names not set in config.yaml")
        return False

    # Resolve CSV
    ann_dir = config["paths"].get("annotations_dir", "dataset/annotations")
    if csv_path is None:
        candidates = [f for f in os.listdir(ann_dir) if f.endswith(".csv")]
        if not candidates:
            _log.error(f"No CSV in {ann_dir}. Pass --csv to specify.")
            return False
        csv_path = os.path.join(ann_dir, candidates[0])

    # Resolve pt_dir
    feat_dir = dirs_dict.get("features")
    pt_dir   = os.path.join(feat_dir, "pt_files") if feat_dir else None

    report = validate_multilabel_csv(
        csv_path       = csv_path,
        label_names    = label_names,
        pt_dir         = pt_dir,
        binary_columns = binary_columns,
        labels_col     = labels_col,
        log            = _log,
    )
    return report["valid"]
