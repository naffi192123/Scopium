"""
pipelines/visualize.py

Attention Heatmap Pipeline.

Usage:
    python main.py heatmap --config config/config.yaml
    python main.py heatmap --config config/config.yaml --experiment results/experiments/metastasis/abmil_...

For each slide in the test split:
  - Loads pre-extracted .h5 file (features + coords)
  - Runs model forward pass (attention_only=True) to get per-patch attention scores
  - Renders an alpha-blended jet-colourmap heatmap on the WSI thumbnail
  - Adds a colorbar strip and slide-ID text annotation
  - Saves top-20 tiles (highest attention) from the raw WSI

Outputs (inside experiment_dir/heatmaps/<slide_id>/):
    <slide_id>_heatmap.png
    <slide_id>_attention_scores.csv    (coord_x, coord_y, attention)
    top20_tiles/
        <rank>_<slide_id>_x<x>_y<y>_a<score>.png
"""

import os
import logging
import csv

import numpy as np
import torch
import h5py
import pandas as pd
import openslide
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.mil_models import build_mil_model, has_attention

logger = logging.getLogger(__name__)


# ── Pure-NumPy jet colourmap (no matplotlib dependency for rendering) ──────────

def _jet(v):
    """v in [0,1] → (R,G,B) uint8 tuple — classic jet colour scale."""
    v = float(np.clip(v, 0, 1))
    r = np.clip(1.5 - abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * v - 1), 0, 1)
    return (int(r * 255), int(g * 255), int(b * 255))


def _apply_jet(scores):
    """scores: (N,) float [0,1] → (N, 3) uint8 RGB array."""
    return np.array([_jet(s) for s in scores], dtype=np.uint8)


def _render_heatmap(thumbnail: Image.Image, coords: np.ndarray,
                    scores: np.ndarray, patch_size: int,
                    level_downsample: float, alpha: float = 0.45) -> Image.Image:
    """
    Build and return the blended heatmap image with colorbar + slide text.

    Parameters
    ----------
    thumbnail        : PIL image at thumbnail resolution
    coords           : (N, 2) level-0 patch coords
    scores           : (N,) normalised attention [0, 1]
    patch_size       : patch size in level-0 pixels
    level_downsample : downsample factor of thumbnail vs level-0
    alpha            : blending opacity for the heatmap overlay
    """
    W, H = thumbnail.size
    thumb_np  = np.array(thumbnail.convert('RGB'), dtype=np.float32)

    heatmap   = np.zeros((H, W, 3), dtype=np.float32)
    count_map = np.zeros((H, W),    dtype=np.float32)
    rgb_scores = _apply_jet(scores).astype(np.float32)  # (N, 3)

    scale  = 1.0 / level_downsample
    ps_vis = max(1, int(patch_size * scale))

    for i, (cx, cy) in enumerate(coords):
        x  = int(cx * scale)
        y  = int(cy * scale)
        x2 = min(x + ps_vis, W)
        y2 = min(y + ps_vis, H)
        if x >= W or y >= H:
            continue
        heatmap[y:y2, x:x2]   += rgb_scores[i]
        count_map[y:y2, x:x2] += 1.0

    mask = count_map > 0
    for c in range(3):
        heatmap[:, :, c][mask] /= count_map[mask]
    heatmap = np.clip(heatmap, 0, 255)

    # Alpha blend
    alpha_map = (alpha * mask.astype(np.float32))[:, :, np.newaxis]
    blended   = ((1 - alpha_map) * thumb_np + alpha_map * heatmap).astype(np.uint8)

    # ── Colorbar strip on the right ───────────────────────────────────────────
    bar_w = max(20, W // 30)
    bar   = np.zeros((H, bar_w, 3), dtype=np.uint8)
    for row in range(H):
        bar[row, :] = _jet(1.0 - row / H)

    combined = np.concatenate([blended, bar], axis=1)
    img = Image.fromarray(combined)
    return img


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
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt        = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get('config', config)
    class_names = ckpt.get('class_names', config['task'].get('class_names', []))
    model_type  = ckpt.get('model_type', ckpt_config['mil']['model'])

    model, _ = build_mil_model(ckpt_config)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    if not has_attention(model):
        _log.warning(
            f"Model '{model_type}' does not produce attention scores. "
            f"Heatmaps only supported for: abmil, clam_sb, clam_mb, dsmil.")
        return

    # ── Resolve feature h5 directory ──────────────────────────────────────────
    # Priority: dirs_dict['features'] (set by --features CLI / config override)
    #           → falls back to ckpt_config paths for backward compat
    tiling      = ckpt_config.get('tiling', {})
    p_size      = tiling.get('patch_size', 512)
    lvl         = tiling.get('patch_level', 0)
    patch_level = int(lvl)

    if dirs_dict.get('features'):
        feat_dir = dirs_dict['features']
    else:
        # Backward compat: derive from checkpoint config
        feat_cfg  = ckpt_config.get('feature_extraction', {})
        paths_ckpt = ckpt_config.get('paths', {})
        model_key = feat_cfg.get('model', 'rn50')
        s_size    = tiling.get('step_size', 512)
        feat_dir  = os.path.join(
            paths_ckpt['results_dir'], 'features',
            f"patch{p_size}_step{s_size}_level{lvl}__{model_key}")
        _log.warning(
            f"dirs_dict has no 'features' key — falling back to: {feat_dir}")

    h5_dir     = os.path.join(feat_dir, 'h5_files')
    slides_dir = config['paths']['slides_dir']
    slide_ext  = config['dataset'].get('slide_extension', '.svs')

    if not os.path.isdir(h5_dir):
        _log.error(
            f"Feature h5_files directory not found: {h5_dir}\n"
            "  Run `python main.py extract --config ...` first, or use "
            "--features to point to an existing feature set.")
        return

    _log.info(f"  Feature dir  : {feat_dir}")

    # Load test slides from split CSV
    task_name = ckpt_config['task']['name']
    paths_cfg = config.get('paths', {})
    test_csv  = os.path.join(paths_cfg['results_dir'], 'splits', task_name, 'test.csv')
    if not os.path.exists(test_csv):
        _log.error(f"test.csv not found: {test_csv}")
        return

    test_df   = pd.read_csv(test_csv)
    slide_col = test_df.columns[0]
    label_col = test_df.columns[1] if len(test_df.columns) > 1 else None

    # ── Per-slide heatmap ──────────────────────────────────────────────────────
    heatmap_root = os.path.join(experiment_dir, 'heatmaps')
    os.makedirs(heatmap_root, exist_ok=True)
    _log.info(f"{'─'*60}")
    _log.info(f"  HEATMAP GENERATION  ({model_type})")
    _log.info(f"{'─'*60}")
    _log.info(f"  Slides    : {len(test_df)}")
    _log.info(f"  Output    : {heatmap_root}")

    processed, skipped = 0, 0

    for _, row in test_df.iterrows():
        sid        = str(row[slide_col])
        true_label = str(row[label_col]) if label_col else 'unknown'

        h5_path    = os.path.join(h5_dir,     sid + '.h5')
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
            features    = torch.tensor(f['features'][:],
                                       dtype=torch.float32).to(device)
            coords      = f['coords'][:]               # (N, 2) int
            patch_level = int(f['coords'].attrs.get('patch_level', patch_level))

        # Guard: features and coords may differ in count
        n = min(features.shape[0], coords.shape[0])
        if features.shape[0] != coords.shape[0]:
            _log.warning(f"  {sid}: #features ({features.shape[0]}) ≠ "
                         f"#coords ({coords.shape[0]}), using {n}.")
        features = features[:n]
        coords   = coords[:n]

        # Forward pass — attention only
        with torch.no_grad():
            attn = model(features, attention_only=True)
        if attn is None:
            _log.warning(f"  {sid}: model returned None attention — skipping")
            skipped += 1
            continue

        attn_np   = attn.detach().cpu().numpy().flatten()[:n]
        # Normalise to [0, 1]
        a_min, a_max = attn_np.min(), attn_np.max()
        attn_norm = ((attn_np - a_min) / (a_max - a_min + 1e-8))

        # Output dir per slide
        slide_out = os.path.join(heatmap_root, sid)
        os.makedirs(slide_out, exist_ok=True)

        # Save attention scores CSV
        csv_path = os.path.join(slide_out, f'{sid}_attention_scores.csv')
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['coord_x', 'coord_y', 'attention'])
            for (x, y), sc in zip(coords, attn_norm):
                writer.writerow([int(x), int(y), round(float(sc), 6)])

        # Build heatmap on thumbnail
        try:
            wsi = openslide.OpenSlide(slide_path)

            # Pick the best thumbnail level (target ≤ 2048 px on longest side)
            vis_level = wsi.level_count - 1
            for lv in range(wsi.level_count):
                if max(wsi.level_dimensions[lv]) <= 2048:
                    vis_level = lv
                    break
            ds    = wsi.level_downsamples[vis_level]
            thumb = wsi.get_thumbnail(wsi.level_dimensions[vis_level])

            heatmap_img = _render_heatmap(
                thumbnail        = thumb,
                coords           = coords,
                scores           = attn_norm,
                patch_size       = p_size,
                level_downsample = ds,
                alpha            = 0.45)

            # ── Slide ID text overlay ──────────────────────────────────────────
            try:
                draw = ImageDraw.Draw(heatmap_img)
                draw.text((10, 10),
                          f"{sid}  |  {true_label}  |  attention heatmap",
                          fill=(255, 255, 255))
            except Exception:
                pass

            heatmap_img.save(os.path.join(slide_out, f'{sid}_heatmap.png'))

            # ── Top-K tiles ────────────────────────────────────────────────────
            top_dir = os.path.join(slide_out, 'top20_tiles')
            os.makedirs(top_dir, exist_ok=True)
            top_idx = np.argsort(attn_norm)[::-1][:top_k]

            for rank, idx in enumerate(top_idx):
                x, y  = int(coords[idx, 0]), int(coords[idx, 1])
                score = attn_norm[idx]
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

    _log.info(f"\nHeatmaps complete | processed={processed} | skipped={skipped}")
    _log.info(f"Output directory: {heatmap_root}")
    return heatmap_root
