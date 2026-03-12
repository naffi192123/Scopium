"""
pipelines/analyse_annotations.py

Comprehensive annotation CSV analysis.
Reports:
  - CSV format validation
  - Class distribution and balance
  - Missing labels / duplicates
  - SVS file coverage (CSV vs. slides_dir)
  - Feature file coverage (for each extracted model)
"""

import os
import sys
import argparse

# Allow running from any CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from utils.config import load_config


# ─── ANSI colours (degrade gracefully on Windows if needed) ──────────────────
try:
    import colorama
    colorama.init()
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


def section(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


def ok(msg):   print(f"  {GREEN}[OK]{RESET}   {msg}")
def warn(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def err(msg):  print(f"  {RED}[ERR]{RESET}  {msg}")
def info(msg): print(f"         {msg}")


def run_analysis(config_path: str, csv_path: str = None,
                 slide_col: str = None, label_col: str = None):

    config = load_config(config_path)

    slides_dir  = config["paths"]["slides_dir"]
    results_dir = config["paths"]["results_dir"]
    slide_ext   = config["dataset"]["slide_extension"]
    ann_dir     = config["paths"].get("annotations_dir", "dataset/annotations")

    # ── Locate CSV ────────────────────────────────────────────────────────────
    if csv_path is None:
        # Auto-detect first CSV in annotations_dir
        candidates = [f for f in os.listdir(ann_dir) if f.endswith(".csv")]
        if not candidates:
            print(f"{RED}No CSV found in {ann_dir}. "
                  f"Pass --csv to specify one.{RESET}")
            sys.exit(1)
        csv_path = os.path.join(ann_dir, candidates[0])

    # ─────────────────────────────────────────────────────────────────────────
    section("1. CSV Format Validation")
    # ─────────────────────────────────────────────────────────────────────────

    try:
        df = pd.read_csv(csv_path)
        ok(f"CSV loaded: {csv_path}")
        info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        info(f"Columns: {list(df.columns)}")
    except Exception as e:
        err(f"Failed to read CSV: {e}")
        sys.exit(1)

    # Auto-detect column names
    cols = [c.lower() for c in df.columns]

    if slide_col is None:
        for candidate in ("slide_id", "slide", "case_id", "filename", "id"):
            if candidate in cols:
                slide_col = df.columns[cols.index(candidate)]
                break
        if slide_col is None:
            slide_col = df.columns[0]
            warn(f"Could not detect slide-ID column. Using first column: '{slide_col}'")
        else:
            ok(f"Slide-ID column: '{slide_col}'")

    if label_col is None:
        for candidate in ("label", "class", "target", "diagnosis", "subtype"):
            if candidate in cols:
                label_col = df.columns[cols.index(candidate)]
                break
        if label_col is None:
            label_col = df.columns[-1]
            warn(f"Could not detect label column. Using last column: '{label_col}'")
        else:
            ok(f"Label column: '{label_col}'")

    # Check required columns present
    missing_req = [c for c in [slide_col, label_col] if c not in df.columns]
    if missing_req:
        err(f"Required columns missing: {missing_req}")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    section("2. Label Analysis & Class Distribution")
    # ─────────────────────────────────────────────────────────────────────────

    total_rows = len(df)
    info(f"Total annotation rows: {total_rows}")

    # Missing labels
    missing_labels = df[label_col].isna().sum()
    if missing_labels == 0:
        ok(f"No missing labels")
    else:
        err(f"{missing_labels} rows have missing labels")
        info("Rows:\n" + str(df[df[label_col].isna()][slide_col].tolist()))

    df_valid = df.dropna(subset=[label_col])

    classes  = sorted(df_valid[label_col].unique().tolist())
    n_classes = len(classes)
    info(f"\nTask:          Binary Classification")
    info(f"Classes ({n_classes}):    {classes}")

    print()
    dist = df_valid[label_col].value_counts()
    for cls, cnt in dist.items():
        pct = 100 * cnt / len(df_valid)
        bar = "#" * int(pct / 2)
        print(f"    {str(cls):<20} {cnt:>4} ({pct:5.1f}%)  [{bar}]")

    # Balance
    if n_classes == 2:
        counts = dist.values
        ratio  = max(counts) / min(counts)
        balance = "Balanced" if ratio <= 1.5 else ("Mildly imbalanced" if ratio <= 3 else "Imbalanced")
        info(f"\nImbalance ratio (majority/minority): {ratio:.2f}x  -> {balance}")

    # ─────────────────────────────────────────────────────────────────────────
    section("3. Deduplication Check")
    # ─────────────────────────────────────────────────────────────────────────

    dup_slides = df[df[slide_col].duplicated(keep=False)]
    if dup_slides.empty:
        ok(f"No duplicate slide IDs")
    else:
        err(f"{df[slide_col].duplicated().sum()} duplicate slide IDs found:")
        info(str(dup_slides[[slide_col, label_col]].to_string(index=False)))

    # ─────────────────────────────────────────────────────────────────────────
    section("4. SVS File Coverage")
    # ─────────────────────────────────────────────────────────────────────────

    # Slides on disk
    on_disk = set(
        os.path.splitext(f)[0]
        for f in os.listdir(slides_dir)
        if f.endswith(slide_ext)
    )
    info(f"Slides in {slides_dir}: {len(on_disk)}")
    info(f"Slides in CSV:              {len(df_valid)}")

    # In CSV but not on disk
    csv_ids = set(df_valid[slide_col].astype(str))
    in_csv_not_disk = sorted(csv_ids - on_disk)
    in_disk_not_csv = sorted(on_disk - csv_ids)

    if not in_csv_not_disk:
        ok("All CSV slides found on disk")
    else:
        warn(f"{len(in_csv_not_disk)} slides in CSV but NOT found on disk:")
        for s in in_csv_not_disk:
            info(f"  - {s}{slide_ext}")

    if not in_disk_not_csv:
        ok("All on-disk slides are annotated in CSV")
    else:
        warn(f"{len(in_disk_not_csv)} slides on disk have NO annotation:")
        for s in in_disk_not_csv:
            info(f"  + {s}")

    # ─────────────────────────────────────────────────────────────────────────
    section("5. Extracted Feature File Coverage")
    # ─────────────────────────────────────────────────────────────────────────

    features_root = os.path.join(results_dir, "features")
    if not os.path.isdir(features_root):
        warn(f"No features directory found at {features_root}")
    else:
        model_dirs = [d for d in os.listdir(features_root)
                      if os.path.isdir(os.path.join(features_root, d))]
        if not model_dirs:
            warn("No model subdirectories found under features/")
        else:
            for model_dir in sorted(model_dirs):
                pt_dir  = os.path.join(features_root, model_dir, "pt_files")
                h5_dir  = os.path.join(features_root, model_dir, "h5_files")

                pt_files = set(os.path.splitext(f)[0]
                               for f in os.listdir(pt_dir)
                               if f.endswith(".pt")) if os.path.isdir(pt_dir) else set()
                h5_files = set(os.path.splitext(f)[0]
                               for f in os.listdir(h5_dir)
                               if f.endswith(".h5")) if os.path.isdir(h5_dir) else set()

                print(f"\n  Model configuration: {BOLD}{model_dir}{RESET}")
                info(f".pt files found: {len(pt_files)}")
                info(f".h5 files found: {len(h5_files)}")

                missing_pt = sorted(csv_ids & on_disk - pt_files)
                missing_h5 = sorted(csv_ids & on_disk - h5_files)

                if not missing_pt:
                    ok("All annotated slides have a .pt feature file")
                else:
                    err(f"{len(missing_pt)} annotated slides missing .pt:")
                    for s in missing_pt:
                        info(f"  - {s}.pt")

                if not missing_h5:
                    ok("All annotated slides have an .h5 feature file")
                else:
                    err(f"{len(missing_h5)} annotated slides missing .h5:")
                    for s in missing_h5:
                        info(f"  - {s}.h5")

    # ─────────────────────────────────────────────────────────────────────────
    section("Summary")
    # ─────────────────────────────────────────────────────────────────────────
    print(f"  CSV path           : {csv_path}")
    print(f"  Total annotations  : {total_rows}")
    print(f"  Task               : Binary Classification")
    print(f"  Classes            : {classes}")
    print(f"  Slides on disk     : {len(on_disk)}")
    print(f"  Missing labels     : {missing_labels}")
    print(f"  Duplicate IDs      : {df[slide_col].duplicated().sum()}")
    print(f"  In CSV / not disk  : {len(in_csv_not_disk)}")
    print(f"  On disk / not CSV  : {len(in_disk_not_csv)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation CSV analysis")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--csv",    default=None,
                        help="Path to annotation CSV (auto-detect if omitted)")
    parser.add_argument("--slide_col", default=None,
                        help="Column name for slide ID")
    parser.add_argument("--label_col", default=None,
                        help="Column name for label")
    args = parser.parse_args()
    run_analysis(args.config, args.csv, args.slide_col, args.label_col)
