"""
pipelines/extract.py

Feature extraction orchestrator.

Design
------
* Mirrors reference extract_features_fp.py compute_w_loader logic.
* One slide at a time (sequential outer loop).
* The openslide handle is opened in the MAIN process and passed directly to
  WSIPatchDataset — DataLoader uses num_workers=0 to avoid Windows ctypes
  pickling errors.
* Features are saved as .pt tensors (one file per slide).
* Supports auto-transform selection per model (transforms_cfg: 'auto').
"""

import os
import time
import logging

import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.slide_dataset import WSIPatchDataset, collate_features
from models.feature_models import load_backbone, pool_features
from utils.transforms import get_transforms

logger = logging.getLogger(__name__)


def command_extract(config: dict, dirs_dict: dict, log=None):
    """
    Run feature extraction over all slides that have a matching .h5 patch file.

    Parameters
    ----------
    config    : dict  Full parsed YAML config.
    dirs_dict : dict  Directory paths set up by utils.config.setup_directories.
    log       : Logger (falls back to module logger if None).
    """
    _log = log or logger

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _log.info(f"Feature Extraction initialised | device: {device}")
    if device.type == 'cuda':
        _log.info(f"Available GPUs: {torch.cuda.device_count()}")

    # ── Config ───────────────────────────────────────────────────────────────
    feat_cfg     = config.get('feature_extraction', {})
    model_type   = feat_cfg.get('model', 'rn50')
    batch_size   = feat_cfg.get('batch_size', 256)
    transforms_cfg = feat_cfg.get('transforms', 'auto')
    weights_path = feat_cfg.get('weights_path', None)

    # ── Build transform ───────────────────────────────────────────────────────
    transform = get_transforms(transforms_cfg, model_type)
    _log.info(f"Transform pipeline built for model '{model_type}'.")

    # ── Load backbone ─────────────────────────────────────────────────────────
    try:
        model, feat_dim, _ = load_backbone(model_type,
                                           weights_path=weights_path,
                                           device=device)
    except Exception as exc:
        _log.error(f"Failed to load model '{model_type}': {exc}")
        return

    # ── Directories ───────────────────────────────────────────────────────────
    slides_dir  = config['paths']['slides_dir']
    slide_ext   = config['dataset']['slide_extension']
    patches_dir = dirs_dict['patches']
    features_dir = dirs_dict['features']

    pt_dir = os.path.join(features_dir, 'pt_files')
    os.makedirs(pt_dir, exist_ok=True)

    # ── Find patch files ──────────────────────────────────────────────────────
    h5_files = sorted(f for f in os.listdir(patches_dir) if f.endswith('.h5'))
    if not h5_files:
        _log.error(f"No .h5 patch files found in {patches_dir}. Run `segment` first.")
        return
    _log.info(f"Found {len(h5_files)} slides to embed.")

    success = failed = 0
    wall_start = time.time()

    # ── Per-slide loop ────────────────────────────────────────────────────────
    for h5_file in h5_files:
        slide_name = os.path.splitext(h5_file)[0]
        h5_path    = os.path.join(patches_dir, h5_file)
        slide_path = os.path.join(slides_dir, slide_name + slide_ext)
        pt_path    = os.path.join(pt_dir, slide_name + '.pt')

        if not os.path.exists(slide_path):
            _log.warning(f"[{slide_name}] Slide file not found: {slide_path} — skipping.")
            failed += 1
            continue

        if os.path.exists(pt_path):
            _log.info(f"[{slide_name}] Already embedded — skipping.")
            success += 1
            continue

        # Open the WSI in the MAIN process so ctypes never crosses a fork boundary
        try:
            wsi     = openslide.open_slide(slide_path)
            dataset = WSIPatchDataset(h5_path, wsi, transform)
        except Exception as exc:
            _log.error(f"[{slide_name}] Could not open slide/h5: {exc}")
            failed += 1
            continue

        if len(dataset) == 0:
            _log.warning(f"[{slide_name}] No patches — saving empty tensor.")
            torch.save(torch.empty(0, feat_dim), pt_path)
            wsi.close()
            success += 1
            continue

        # num_workers=0: patches are loaded in the main thread — no pickling issues
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_features,
        )

        _log.info(
            f"[{slide_name}] Extracting {len(dataset)} patches "
            f"| {len(loader)} batches | batch_size={batch_size}")

        all_features = []
        slide_start  = time.time()

        try:
            with torch.no_grad():
                for batch_imgs, _coords in tqdm(loader, desc=slide_name, leave=False):
                    batch_imgs = batch_imgs.to(device, non_blocking=True)
                    out        = model(batch_imgs)
                    feats      = pool_features(out, model_type)
                    all_features.append(feats.cpu())

        except KeyboardInterrupt:
            _log.critical("Interrupted by user.")
            wsi.close()
            return
        except Exception as exc:
            _log.error(f"[{slide_name}] Extraction error: {exc}")
            wsi.close()
            failed += 1
            continue

        wsi.close()

        # Concatenate [N, D] and save
        slide_features = torch.cat(all_features, dim=0)
        torch.save(slide_features, pt_path)

        elapsed = time.time() - slide_start
        _log.info(
            f"[{slide_name}] Done | Shape: {slide_features.shape} | "
            f"Time: {elapsed:.1f}s | Saved: {pt_path}")
        success += 1

    total = time.time() - wall_start
    _log.info(
        f"Extraction complete | success={success} | failed={failed} | "
        f"total_time={total:.1f}s")
    _log.info(f"Features directory: {pt_dir}")
