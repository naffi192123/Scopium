"""
datasets/mil_dataset.py

MIL Bag Dataset — loads pre-extracted feature tensors + slide labels.

Each slide is treated as a "bag" of patch embeddings.
Labels are loaded from an annotation CSV and mapped to integer indices
using the class_names list from the task config.
"""

import os
import logging

import torch
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MILBagDataset(Dataset):
    """
    Multiple Instance Learning dataset.

    Parameters
    ----------
    csv_path      : str   Path to split CSV (slide_id, label).
    pt_dir        : str   Directory containing <slide_id>.pt feature files.
    class_names   : list  Class names in label-index order (e.g. ['benign', 'malignant']).
    slide_col     : str   Column name for slide ID.
    label_col     : str   Column name for label.
    max_patches   : int   Maximum patches to load per slide (None = all).
    """

    def __init__(self, csv_path: str, pt_dir: str, class_names: list,
                 slide_col: str = None, label_col: str = None,
                 max_patches: int = None):

        self.pt_dir        = pt_dir
        self.class_names   = class_names
        self.class_to_idx  = {c: i for i, c in enumerate(class_names)}
        self.max_patches   = max_patches

        df = pd.read_csv(csv_path)
        cols = [c.lower() for c in df.columns]

        # Auto-detect columns
        if slide_col is None:
            for cand in ("slide_id", "slide", "case_id", "filename", "id"):
                if cand in cols:
                    slide_col = df.columns[cols.index(cand)]
                    break
            if slide_col is None:
                slide_col = df.columns[0]

        if label_col is None:
            for cand in ("label", "class", "target", "diagnosis", "subtype"):
                if cand in cols:
                    label_col = df.columns[cols.index(cand)]
                    break
            if label_col is None:
                label_col = df.columns[-1]

        df = df.dropna(subset=[label_col]).copy()

        # Filter to slides that have a .pt feature file
        self.slide_ids = []
        self.labels    = []
        self.pt_paths  = []
        missing = 0

        for _, row in df.iterrows():
            sid   = str(row[slide_col])
            label = str(row[label_col]).strip()
            pt    = os.path.join(pt_dir, sid + '.pt')
            if not os.path.exists(pt):
                missing += 1
                continue
            if label not in self.class_to_idx:
                logger.warning(f"Unknown label '{label}' for slide {sid} — skipping.")
                continue
            self.slide_ids.append(sid)
            self.labels.append(self.class_to_idx[label])
            self.pt_paths.append(pt)

        if missing:
            logger.warning(
                f"MILBagDataset: {missing} slides have no .pt file in {pt_dir}.")
        logger.info(
            f"MILBagDataset loaded: {len(self)} slides | "
            f"classes={class_names} | "
            f"dist={self._class_dist()}")

    def _class_dist(self):
        from collections import Counter
        c = Counter(self.labels)
        return {self.class_names[k]: v for k, v in sorted(c.items())}

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        feats = torch.load(self.pt_paths[idx], map_location='cpu', weights_only=False)
        if self.max_patches and len(feats) > self.max_patches:
            perm  = torch.randperm(len(feats))[:self.max_patches]
            feats = feats[perm]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feats, label, self.slide_ids[idx]


def mil_collate_fn(batch):
    """
    Custom collate — keeps each bag as a 2-D tensor (N_i, D).
    Returns a list of feature tensors, a stacked label tensor, and slide_ids.
    """
    features  = [item[0] for item in batch]
    labels    = torch.stack([item[1] for item in batch])
    slide_ids = [item[2] for item in batch]
    return features, labels, slide_ids


def build_mil_datasets(config: dict, dirs_dict: dict):
    """
    Build train / val (optional) / test MILBagDataset instances.

    Returns
    -------
    dict of {split_name: MILBagDataset}  e.g. {'train': ..., 'test': ...}
    """
    task_cfg  = config.get('task', {})
    feat_cfg  = config.get('feature_extraction', {})
    paths_cfg = config.get('paths', {})
    split_cfg = config.get('split', {})

    class_names = task_cfg.get('class_names', None)
    task_name   = task_cfg.get('name', 'task')
    model_key   = feat_cfg.get('model', 'rn50')

    if class_names is None:
        # fall back: infer from training CSV
        import pandas as pd
        train_csv = os.path.join(paths_cfg['results_dir'], 'splits', task_name, 'train.csv')
        df = pd.read_csv(train_csv)
        class_names = sorted(df.iloc[:, -1].astype(str).unique().tolist())
        logger.warning(f"class_names not in config — inferred: {class_names}")

    # Resolve feature pt_dir
    p_size  = config['tiling'].get('patch_size', 512)
    s_size  = config['tiling'].get('step_size', 512)
    lvl     = config['tiling'].get('patch_level', 0)
    pt_dir  = os.path.join(paths_cfg['results_dir'], 'features',
                           f"patch{p_size}_step{s_size}_level{lvl}__{model_key}",
                           'pt_files')

    splits_dir = os.path.join(paths_cfg['results_dir'], 'splits', task_name)

    datasets = {}
    for split_name in ('train', 'val', 'test'):
        csv_path = os.path.join(splits_dir, f'{split_name}.csv')
        if not os.path.exists(csv_path):
            continue
        datasets[split_name] = MILBagDataset(
            csv_path    = csv_path,
            pt_dir      = pt_dir,
            class_names = class_names,
        )

    return datasets, class_names
