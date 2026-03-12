"""
pipelines/visualize.py

Attention Heatmap Pipeline.

Usage:
    python main.py heatmap --config config/config.yaml
    python main.py heatmap --config config/config.yaml --experiment results/experiments/metastasis/abmil_...

For each slide in the test split:
  - Loads pre-extracted .h5 file (features + coords)
  - Runs model forward pass (attention_only) to get per-patch attention scores
  - Renders an alpha-blended heatmap on the WSI thumbnail
  - Saves top-20 tiles (highest attention) from the raw WSI

Outputs (inside experiment_dir/heatmaps/<slide_id>/):
    <slide_id>_heatmap.png
    <slide_id>_attention_scores.csv    (coord_x, coord_y, attention)
    top20_tiles/
        <rank>_<slide_id>_x<x>_y<y>_a<score>.png
"""

import os
import logging

import numpy as np
import torch
import h5py
import pandas as pd
import openslide
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.mil_models import build_mil_model, has_attention

logger = logging.getLogger(__name__)


def _find_latest_experiment(config):
    task_name   = config['task']['name']
    results_dir = config['paths']['results_dir']
    exp_root    = os.path.join(results_dir, 'experiments', task_name)
    if not os.path.isdir(exp_root):
        return None
    dirs = sorted([d for d in os.listdir(exp_root)
                   if os.path.isdir(os.path.join(exp_root, d))],
                  reverse=True)
    return os.path.join(exp_root, dirs[0]) if dirs else None


def _render_heatmap(thumbnail: Image.Image, coords: np.ndarray,
                    scores: np.ndarray, patch_size: int,
                    level_downsample: float, alpha: float = 0.5) -> Image.Image:
    """
    Overlay per-patch attention scores on WSI thumbnail.

    Parameters
    ----------
    thumbnail      : PIL thumbnail at some downsample level
    coords         : (N, 2) patch top-left coords in level-0 pixels
    scores         : (N,) normalised attention [0, 1]
    patch_size     : patch size in level-0 pixels
    level_downsample : downsample factor of the thumbnail vs level-0
    alpha          : blending opacity for the overlay
    """
    W, H     = thumbnail.size
    heat_arr = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.int32)

    # Project patches onto thumbnail coordinates
    scale = 1.0 / level_downsample
    p_w   = max(1, int(patch_size * scale))
    p_h   = max(1, int(patch_size * scale))

    for (x, y), s in zip(coords, scores):
        tx = int(x * scale)
        ty = int(y * scale)
        x1, x2 = max(0, tx), min(W, tx + p_w)
        y1, y2 = max(0, ty), min(H, ty + p_h)
        if x2 > x1 and y2 > y1:
            heat_arr[y1:y2, x1:x2] += s
            count[y1:y2, x1:x2]    += 1

    mask       = count > 0
    heat_norm  = np.zeros_like(heat_arr)
    heat_norm[mask] = heat_arr[mask] / count[mask]
    heat_norm  = (heat_norm - heat_norm.min()) / (heat_norm.max() - heat_norm.min() + 1e-8)

    cmap_fn    = cm.get_cmap('jet')
    overlay    = (cmap_fn(heat_norm)[:, :, :3] * 255).astype(np.uint8)
    overlay_im = Image.fromarray(overlay).convert('RGBA')
    overlay_im.putalpha(int(alpha * 255 * mask.astype(np.uint8)))

    base_rgba  = thumbnail.convert('RGBA')
    merged     = Image.alpha_composite(base_rgba, overlay_im)
    return merged.convert('RGB')


def command_heatmap(config: dict, dirs_dict: dict, log=None,
                    experiment_dir: str = None, top_k: int = 20):
    _log = log or logger

    # ── Find experiment ────────────────────────────────────────────────────────
    if experiment_dir is None:
        experiment_dir = _find_latest_experiment(config)
    if experiment_dir is None:
        _log.error("No experiment directory found. Train first.")
        return

    best_ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    if not os.path.exists(best_ckpt_path):
        _log.error(f"best_model.pt not found in {experiment_dir}")
        return

    # ── Load model ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get('config', config)
    class_names = ckpt.get('class_names', config['task'].get('class_names', []))

    model, _ = build_mil_model(ckpt_config)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    if not has_attention(model):
        _log.warning(
            f"Model '{ckpt_config['mil']['model']}' does not produce attention scores. "
            f"Heatmaps are only supported for: abmil, clam_sb, clam_mb, dsmil.")
        return

    # ── Resolve paths ──────────────────────────────────────────────────────────
    feat_cfg  = ckpt_config.get('feature_extraction', {})
    tiling    = ckpt_config.get('tiling', {})
    paths_cfg = ckpt_config.get('paths', {})

    model_key = feat_cfg.get('model', 'rn50')
    p_size    = tiling.get('patch_size', 512)
    s_size    = tiling.get('step_size', 512)
    lvl       = tiling.get('patch_level', 0)
    patch_level = int(lvl)

    feat_dir  = os.path.join(paths_cfg['results_dir'], 'features',
                             f"patch{p_size}_step{s_size}_level{lvl}__{model_key}")
    h5_dir    = os.path.join(feat_dir, 'h5_files')
    slides_dir = paths_cfg['slides_dir']
    slide_ext  = config['dataset'].get('slide_extension', '.svs')

    # Load test slides from split CSV
    task_name = ckpt_config['task']['name']
    test_csv  = os.path.join(paths_cfg['results_dir'], 'splits', task_name, 'test.csv')
    if not os.path.exists(test_csv):
        _log.error(f"test.csv not found: {test_csv}")
        return

    test_df = pd.read_csv(test_csv)
    slide_col = test_df.columns[0]
    label_col = test_df.columns[1] if len(test_df.columns) > 1 else None

    # ── Per-slide heatmap ──────────────────────────────────────────────────────
    heatmap_root = os.path.join(experiment_dir, 'heatmaps')
    os.makedirs(heatmap_root, exist_ok=True)

    processed, skipped = 0, 0

    for _, row in test_df.iterrows():
        sid        = str(row[slide_col])
        true_label = str(row[label_col]) if label_col else 'unknown'

        h5_path    = os.path.join(h5_dir,    sid + '.h5')
        slide_path = os.path.join(slides_dir, sid + slide_ext)

        if not os.path.exists(h5_path):
            _log.warning(f"  [SKIP] {sid} — no .h5 file: {h5_path}")
            skipped += 1
            continue
        if not os.path.exists(slide_path):
            _log.warning(f"  [SKIP] {sid} — no WSI: {slide_path}")
            skipped += 1
            continue

        _log.info(f"  Processing: {sid}  (label={true_label})")

        # Load features + coords from H5
        with h5py.File(h5_path, 'r') as f:
            features = torch.tensor(f['features'][:], dtype=torch.float32).to(device)
            coords   = f['coords'][:]                # (N, 2) int

        # Forward pass — attention only
        with torch.no_grad():
            attn = model(features, attention_only=True)
        if attn is None:
            _log.warning(f"  {sid}: model returned None attention — skipping")
            skipped += 1
            continue

        attn_np = attn.detach().cpu().numpy().flatten()
        # Normalise to [0, 1]
        attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

        # Output dir per slide
        slide_out = os.path.join(heatmap_root, sid)
        os.makedirs(slide_out, exist_ok=True)

        # Save attention scores CSV
        pd.DataFrame({
            'coord_x':   coords[:, 0],
            'coord_y':   coords[:, 1],
            'attention': attn_norm,
        }).to_csv(os.path.join(slide_out, f'{sid}_attention_scores.csv'), index=False)

        # Build heatmap on thumbnail
        try:
            wsi = openslide.OpenSlide(slide_path)
            # Get best thumbnail level
            thumb_level = wsi.get_best_level_for_downsample(32)
            ds           = wsi.level_downsamples[thumb_level]
            thumb        = wsi.get_thumbnail((
                wsi.dimensions[0] // int(ds),
                wsi.dimensions[1] // int(ds)))

            heatmap_img = _render_heatmap(
                thumbnail        = thumb,
                coords           = coords,
                scores           = attn_norm,
                patch_size       = p_size,
                level_downsample = ds,
                alpha            = 0.45)

            heatmap_img.save(os.path.join(slide_out, f'{sid}_heatmap.png'))

            # ── Top-K tiles ────────────────────────────────────────────────────
            top_dir = os.path.join(slide_out, 'top20_tiles')
            os.makedirs(top_dir, exist_ok=True)
            top_idx = np.argsort(attn_norm)[::-1][:top_k]

            for rank, idx in enumerate(top_idx):
                x, y   = int(coords[idx, 0]), int(coords[idx, 1])
                score  = attn_norm[idx]
                try:
                    tile  = wsi.read_region((x, y), patch_level, (p_size, p_size))
                    tile  = tile.convert('RGB')
                    fname = f"{rank+1:02d}_{sid}_x{x}_y{y}_a{score:.3f}.png"
                    tile.save(os.path.join(top_dir, fname))
                except Exception as te:
                    _log.warning(f"    Could not extract tile at ({x},{y}): {te}")

            wsi.close()
            _log.info(f"    Heatmap saved | Top-{top_k} tiles saved")
            processed += 1

        except Exception as e:
            _log.error(f"  Failed for {sid}: {e}")
            skipped += 1

    _log.info(f"Heatmaps complete | processed={processed} | skipped={skipped}")
    _log.info(f"Output directory: {heatmap_root}")
    return heatmap_root
