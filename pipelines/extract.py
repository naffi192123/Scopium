"""
pipelines/extract.py

GPU-optimised feature extraction orchestrator.

Architecture
------------
Multi-process, one process per GPU (``torch.multiprocessing.spawn``).
Each process receives a disjoint slice of slides and runs them independently,
giving true GPU parallelism with no communication overhead.

Within each worker process:
* DataLoader with ``num_workers=dataloader_workers`` overlaps CPU patch reading
  with GPU compute (the key fix for near-0% GPU utilisation).
* ``torch.cuda.amp.autocast`` (FP16) doubles throughput on A100 / H100.
* The WSI handle is opened once per DataLoader worker via ``worker_init_fn``
  so openslide ctypes objects are never pickled.

Outputs per slide (unchanged from v1):
  features/h5_files/<slide>.h5   HDF5: 'features' (N, D) + 'coords' (N, 2)
  features/pt_files/<slide>.pt   PyTorch FloatTensor (N, D)

Config keys (feature_extraction section)
-----------------------------------------
  model                 : str   backbone key
  batch_size            : int   per-GPU batch size (default 256 on A100)
  transforms            : str   transform preset
  weights_path          : str   path to local weights
  num_extraction_workers: int   number of GPU processes (default = GPU count)
  dataloader_workers    : int   DataLoader workers per GPU process (default 4)
  pin_memory            : bool  pin host tensors for faster H↔D transfer (default true)
  use_amp               : bool  FP16 autocast on A100/H100 (default true)
"""

import os
import time
import logging
import traceback

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.slide_dataset import WSIPatchDataset, collate_features, worker_init_fn
from models.feature_models import load_backbone, pool_features
from utils.transforms import get_transforms
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDF5 incremental writer
# ---------------------------------------------------------------------------

def _save_hdf5(output_path: str, asset_dict: dict, mode: str = 'a'):
    """
    Append or create datasets inside an HDF5 file.
    First call (mode='w') creates; subsequent calls (mode='a') extend.
    """
    with h5py.File(output_path, mode) as f:
        for key, val in asset_dict.items():
            if key not in f:
                maxshape = (None,) + val.shape[1:]
                f.create_dataset(
                    key, data=val,
                    maxshape=maxshape,
                    chunks=(1,) + val.shape[1:])
            else:
                dset = f[key]
                dset.resize(len(dset) + len(val), axis=0)
                dset[-len(val):] = val


# ---------------------------------------------------------------------------
# Per-GPU worker — runs inside each spawned process
# ---------------------------------------------------------------------------

def _extract_worker(
    rank: int,
    all_subsets: list,
    config: dict,
    dirs_dict: dict,
    log_path: str,
):
    """
    Entry point for each GPU worker process.

    Parameters
    ----------
    rank        : int   GPU index injected by mp.spawn as the first argument.
    all_subsets : list  List-of-lists; this worker processes all_subsets[rank].
    config      : dict  Full parsed YAML config.
    dirs_dict   : dict  Result paths from setup_directories().
    log_path    : str   Shared rotating log file path.
    """
    slides_subset = all_subsets[rank]
    # Per-process logger (appends to shared rotating file)
    log = setup_logger(log_path, worker_id=rank)
    log.info(f"[GPU {rank}] Worker started | {len(slides_subset)} slides assigned")

    # ── Device ──────────────────────────────────────────────────────────────
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # ── Config ───────────────────────────────────────────────────────────────
    feat_cfg          = config.get('feature_extraction', {})
    model_type        = feat_cfg.get('model', 'rn50')
    batch_size        = feat_cfg.get('batch_size', 256)
    transforms_cfg    = feat_cfg.get('transforms', 'auto')
    weights_path      = feat_cfg.get('weights_path', None)
    dl_workers        = feat_cfg.get('dataloader_workers', 4)
    pin_memory        = feat_cfg.get('pin_memory', True)
    use_amp           = feat_cfg.get('use_amp', True)

    slides_dir = config['paths']['slides_dir']
    slide_ext  = config['dataset']['slide_extension']

    features_dir = dirs_dict['features']
    patches_dir  = dirs_dict['patches']
    pt_dir = os.path.join(features_dir, 'pt_files')
    h5_dir = os.path.join(features_dir, 'h5_files')
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)

    # ── Transform & model ────────────────────────────────────────────────────
    transform = get_transforms(transforms_cfg, model_type)
    try:
        # use_data_parallel=False — each process owns exactly one GPU
        model, feat_dim, _ = load_backbone(
            model_type, weights_path=weights_path,
            device=device, use_data_parallel=False)
    except Exception as exc:
        log.error(f"[GPU {rank}] Failed to load model: {exc}")
        return

    # AMP scaler — only needed for training; for inference just use autocast
    amp_ctx = torch.cuda.amp.autocast if use_amp else _no_amp

    success = failed = 0
    wall_start = time.time()

    for h5_file in slides_subset:
        slide_name = os.path.splitext(h5_file)[0]
        h5_path    = os.path.join(patches_dir, h5_file)
        slide_path = os.path.join(slides_dir, slide_name + slide_ext)
        pt_path    = os.path.join(pt_dir, slide_name + '.pt')
        h5_out     = os.path.join(h5_dir, slide_name + '.h5')

        if not os.path.exists(slide_path):
            log.warning(f"[GPU {rank}][{slide_name}] Slide not found — skipping.")
            failed += 1
            continue

        if os.path.exists(pt_path) and os.path.exists(h5_out):
            log.info(f"[GPU {rank}][{slide_name}] Already embedded — skipping.")
            success += 1
            continue

        # ── Build dataset ─────────────────────────────────────────────────
        # Pass slide PATH (not handle) — worker_init_fn opens in each worker
        try:
            dataset = WSIPatchDataset(h5_path, slide_path, transform)
        except Exception as exc:
            log.error(f"[GPU {rank}][{slide_name}] Dataset error: {exc}")
            failed += 1
            continue

        if len(dataset) == 0:
            log.warning(f"[GPU {rank}][{slide_name}] No patches — saving empty files.")
            empty = torch.empty(0, feat_dim)
            torch.save(empty, pt_path)
            _save_hdf5(
                h5_out,
                {'features': np.empty((0, feat_dim), dtype=np.float32),
                 'coords':   np.empty((0, 2), dtype=np.int64)},
                mode='w')
            success += 1
            continue

        # num_workers > 0:  DataLoader workers pre-fetch batches while GPU runs
        # worker_init_fn :  each worker opens its own slide handle (no pickling)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dl_workers,
            pin_memory=pin_memory,
            persistent_workers=(dl_workers > 0),
            prefetch_factor=2 if dl_workers > 0 else None,
            worker_init_fn=worker_init_fn if dl_workers > 0 else None,
            collate_fn=collate_features,
        )

        log.info(
            f"[GPU {rank}][{slide_name}] "
            f"{len(dataset)} patches | {len(loader)} batches | "
            f"batch_size={batch_size} | dl_workers={dl_workers} | "
            f"AMP={'on' if use_amp else 'off'}")

        all_features = []
        h5_mode      = 'w'
        slide_start  = time.time()

        try:
            with torch.no_grad():
                for batch_imgs, coords in tqdm(
                        loader, desc=f"GPU{rank}:{slide_name}", leave=False):
                    batch_imgs = batch_imgs.to(device, non_blocking=True)
                    with amp_ctx():
                        out = model(batch_imgs)
                    feats = pool_features(out, model_type).float().cpu()

                    all_features.append(feats)
                    _save_hdf5(
                        h5_out,
                        {'features': feats.numpy(),
                         'coords':   coords},
                        mode=h5_mode)
                    h5_mode = 'a'

        except KeyboardInterrupt:
            log.critical(f"[GPU {rank}] Interrupted.")
            return
        except Exception as exc:
            log.error(
                f"[GPU {rank}][{slide_name}] Extraction error: {exc}\n"
                + traceback.format_exc())
            failed += 1
            continue

        dataset.close()

        slide_features = torch.cat(all_features, dim=0)
        torch.save(slide_features, pt_path)

        elapsed = time.time() - slide_start
        log.info(
            f"[GPU {rank}][{slide_name}] Done | "
            f"Shape: {slide_features.shape} | Time: {elapsed:.1f}s")
        success += 1

    total = time.time() - wall_start
    log.info(
        f"[GPU {rank}] Finished | success={success} | failed={failed} | "
        f"wall_time={total:.1f}s")


# ---------------------------------------------------------------------------
# No-op context manager (fallback when use_amp=False)
# ---------------------------------------------------------------------------

class _no_amp:
    """Drop-in replacement for torch.cuda.amp.autocast when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ---------------------------------------------------------------------------
# Main entry point — called by main.py
# ---------------------------------------------------------------------------

def command_extract(config: dict, dirs_dict: dict, log=None):
    """
    Run feature extraction over all slides that have a matching .h5 patch file.

    Spawns one process per GPU (or fewer if there are fewer slides than GPUs).
    Falls back to a single-process loop when no CUDA device is available.

    Parameters
    ----------
    config    : dict  Full parsed YAML config.
    dirs_dict : dict  Directory paths set up by utils.config.setup_directories.
    log       : Logger (falls back to module logger if None).
    """
    _log = log or logger

    # ── GPU inventory ────────────────────────────────────────────────────────
    n_gpus_available = torch.cuda.device_count()
    feat_cfg         = config.get('feature_extraction', {})
    n_workers_cfg    = feat_cfg.get('num_extraction_workers', n_gpus_available or 1)

    if n_gpus_available == 0:
        _log.warning("No CUDA GPUs found — running on CPU (single process).")
        n_procs = 1
    else:
        n_procs = min(n_workers_cfg, n_gpus_available)
        _log.info(
            f"Feature Extraction | GPUs available: {n_gpus_available} | "
            f"Processes: {n_procs} (num_extraction_workers={n_workers_cfg})")

    # ── Find patch coordinate files ───────────────────────────────────────────
    patches_dir = dirs_dict['patches']
    h5_files = sorted(f for f in os.listdir(patches_dir) if f.endswith('.h5'))
    if not h5_files:
        _log.error(f"No .h5 patch files found in {patches_dir}. Run `segment` first.")
        return
    _log.info(f"Found {len(h5_files)} slides to embed.")

    # ── Log path for workers ──────────────────────────────────────────────────
    log_path = os.path.join(config['paths']['results_dir'], 'wsi_framework.log')

    # ── Single-process path (CPU or 1 GPU) ────────────────────────────────────
    if n_procs == 1:
        # Wrap in list-of-lists so all_subsets[rank] indexing works uniformly
        _extract_worker(0, [h5_files], config, dirs_dict, log_path)
        return

    # ── Multi-process path: split slides across workers ───────────────────────
    # Round-robin distribution so each GPU gets roughly equal work
    subsets = [[] for _ in range(n_procs)]
    for i, f in enumerate(h5_files):
        subsets[i % n_procs].append(f)

    for rank, subset in enumerate(subsets):
        _log.info(f"  GPU {rank} assigned {len(subset)} slides")

    _log.info("Spawning GPU workers…")
    try:
        mp.spawn(
            _extract_worker,
            # rank is injected as first arg by spawn; all workers receive the same
            # args tuple and each indexes all_subsets[rank] to get its slice.
            args=(subsets, config, dirs_dict, log_path),
            nprocs=n_procs,
            join=True,
        )
    except Exception as exc:
        _log.error(f"Multi-process extraction failed: {exc}")
        raise

    _log.info(f"All GPU workers complete. Features at: {dirs_dict['features']}")
