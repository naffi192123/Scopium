"""
pipelines/classify_heatmap.py

Tile-level prediction heatmap visualisation.

Reads prediction CSVs produced by ``pipelines/classify.py`` and overlays
per-patch classification colours on the WSI thumbnail.

Two visualisation modes
-----------------------
category_map
    One PNG per slide. Each tile is rendered in its predicted class colour.
    A colour-coded legend is appended below the image.

confidence
    One PNG per slide per category. Tile colour = jet(confidence score).
    A jet colorbar is appended to the right. Top-K tiles are also saved.

Commands
--------
    python main.py classify-heatmap --config config/config.yaml
    python main.py classify-heatmap --config config/config.yaml --slide CMU-1

Config (under patch_classifier.heatmap)
----------------------------------------
    slides      : all | "CMU-1" | [CMU-1, CMU-2]
    categories  : all | "TUM"   | [TUM, STR, LYM]
    mode        : category_map | confidence
    alpha       : 0.50
    top_k_tiles : 10

Outputs  (results/patch_predictions/{subfolder}/heatmaps/{slide_id}/)
----------------------------------------------------------------------
    {slide_id}_category_map.png
    {slide_id}_confidence_{CAT}.png
    top{K}_{CAT}/{rank}_{slide_id}_x{x}_y{y}_conf{score}.png
"""

import os
import logging

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')   # non-interactive backend

logger = logging.getLogger(__name__)


# ── Fixed category colour palette (RGB uint8) ─────────────────────────────────
CATEGORY_COLORS = {
    "ADI": (210, 180, 140),   # tan
    "DEB": (160, 160, 160),   # grey
    "LYM": (170,  80, 200),   # purple
    "MUC": ( 60, 190, 170),   # teal
    "MUS": (220, 130,  50),   # orange
    "NOR": ( 80, 180,  80),   # green
    "STR": ( 70, 120, 220),   # blue
    "TUM": (210,  50,  50),   # red
}


# ---------------------------------------------------------------------------
# Low-level rendering helpers
# ---------------------------------------------------------------------------

def _jet(v: float):
    """v ∈ [0, 1] → (R, G, B) uint8 — classic jet colour scale."""
    v = float(np.clip(v, 0, 1))
    r = np.clip(1.5 - abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * v - 1), 0, 1)
    return (int(r * 255), int(g * 255), int(b * 255))


def _render_heatmap(
    thumbnail: Image.Image,
    coords: np.ndarray,
    patch_colors: np.ndarray,
    patch_size: int,
    level_downsample: float,
    alpha: float = 0.50,
) -> Image.Image:
    """
    Alpha-blend a per-patch colour overlay onto a WSI thumbnail.

    Parameters
    ----------
    thumbnail        : PIL Image at thumbnail resolution.
    coords           : (N, 2) int — level-0 patch (x, y) coordinates.
    patch_colors     : (N, 3) uint8 RGB — colour for each patch.
    patch_size       : patch size in level-0 pixels.
    level_downsample : thumbnail downsample factor vs level-0.
    alpha            : overlay opacity in [0, 1].
    """
    W, H      = thumbnail.size
    thumb_np  = np.array(thumbnail.convert("RGB"), dtype=np.float32)
    heatmap   = np.zeros((H, W, 3), dtype=np.float32)
    count_map = np.zeros((H, W),    dtype=np.float32)

    scale  = 1.0 / level_downsample
    ps_vis = max(1, int(patch_size * scale))

    for i, (cx, cy) in enumerate(coords):
        x  = int(cx * scale)
        y  = int(cy * scale)
        x2 = min(x + ps_vis, W)
        y2 = min(y + ps_vis, H)
        if x >= W or y >= H:
            continue
        heatmap[y:y2, x:x2]   += patch_colors[i].astype(np.float32)
        count_map[y:y2, x:x2] += 1.0

    mask = count_map > 0
    for c in range(3):
        heatmap[:, :, c][mask] /= count_map[mask]

    alpha_map = (alpha * mask.astype(np.float32))[:, :, np.newaxis]
    blended   = ((1 - alpha_map) * thumb_np + alpha_map * heatmap).astype(np.uint8)
    return Image.fromarray(blended)


def _add_legend(img: Image.Image, categories: list) -> Image.Image:
    """Append a colour-coded legend strip below the image."""
    W, H     = img.size
    swatch   = max(18, H // 25)
    pad      = 6
    legend_h = len(categories) * (swatch + pad) + pad
    legend   = Image.new("RGB", (W, legend_h), color=(30, 30, 30))
    draw     = ImageDraw.Draw(legend)
    for i, cat in enumerate(categories):
        color = CATEGORY_COLORS.get(cat.upper(), (200, 200, 200))
        y0    = pad + i * (swatch + pad)
        draw.rectangle([pad, y0, pad + swatch, y0 + swatch], fill=color)
        draw.text((pad + swatch + 6, y0 + 2), cat.upper(), fill=(230, 230, 230))
    out = Image.new("RGB", (W, H + legend_h))
    out.paste(img,    (0, 0))
    out.paste(legend, (0, H))
    return out


def _add_jet_colorbar(img: Image.Image) -> Image.Image:
    """Append a vertical jet colorbar strip to the right of the image."""
    W, H  = img.size
    bar_w = max(25, W // 30)
    bar   = np.zeros((H, bar_w, 3), dtype=np.uint8)
    for row in range(H):
        bar[row, :] = _jet(1.0 - row / H)
    bar_img  = Image.fromarray(bar)
    out      = Image.new("RGB", (W + bar_w, H))
    out.paste(img,     (0, 0))
    out.paste(bar_img, (W, 0))
    draw = ImageDraw.Draw(out)
    draw.text((W + 2, 4),          "1.0", fill=(255, 255, 255))
    draw.text((W + 2, H - 14),     "0.0", fill=(255, 255, 255))
    return out


# ---------------------------------------------------------------------------
# Slide / category resolution helpers
# ---------------------------------------------------------------------------

def _resolve_slides(hm_cfg: dict, pred_dir: str,
                    slide_override: str = None) -> list:
    """Return sorted list of slide IDs to process."""
    if slide_override:
        return [slide_override]
    slides_cfg = hm_cfg.get("slides", "all")
    if slides_cfg == "all" or not slides_cfg:
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(pred_dir) if f.endswith(".csv"))
    if isinstance(slides_cfg, str):
        return [slides_cfg]
    return list(slides_cfg)


def _resolve_categories(hm_cfg: dict) -> list:
    """Return list of upper-cased category names to visualise."""
    cats = hm_cfg.get("categories", "all")
    all_cats = list(CATEGORY_COLORS.keys())
    if cats == "all" or not cats:
        return all_cats
    if isinstance(cats, str):
        return [cats.upper()]
    return [c.upper() for c in cats]


def _open_wsi_with_thumb(slide_path: str, max_px: int = 2048):
    """Open slide, pick best thumbnail level, return (wsi, thumb, ds)."""
    wsi       = openslide.OpenSlide(slide_path)
    vis_level = wsi.level_count - 1
    for lv in range(wsi.level_count):
        if max(wsi.level_dimensions[lv]) <= max_px:
            vis_level = lv
            break
    ds    = wsi.level_downsamples[vis_level]
    thumb = wsi.get_thumbnail(wsi.level_dimensions[vis_level])
    return wsi, thumb, ds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def command_classify_heatmap(config: dict, dirs_dict: dict,
                              log=None, slide_override: str = None):
    """
    Render tile-level prediction heatmaps onto WSI thumbnails.

    Parameters
    ----------
    config         : full parsed YAML config
    dirs_dict      : from utils.config.setup_directories
    log            : optional logger
    slide_override : process only this slide (from --slide CLI flag)
    """
    _log = log or logger

    # ── Config ────────────────────────────────────────────────────────────────
    clf_cfg    = config.get("patch_classifier", {})
    hm_cfg     = clf_cfg.get("heatmap", {})
    mode       = hm_cfg.get("mode",        "category_map")
    alpha      = float(hm_cfg.get("alpha", 0.50))
    top_k      = int(hm_cfg.get("top_k_tiles", 10))
    slides_dir = config["paths"]["slides_dir"]
    slide_ext  = config["dataset"].get("slide_extension", ".svs")
    results_root = config["paths"]["results_dir"]
    p_size     = int(config.get("tiling", {}).get("patch_size", 512))

    feat_subfolder = dirs_dict.get("_feature_subfolder",
                                   os.path.basename(dirs_dict["features"]))
    pred_dir = os.path.join(results_root, "patch_predictions", feat_subfolder)

    if not os.path.isdir(pred_dir):
        _log.error(
            f"No prediction directory: {pred_dir}\n"
            "  Run `python main.py classify ...` first.")
        return

    heatmap_dir = os.path.join(pred_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    slides     = _resolve_slides(hm_cfg, pred_dir, slide_override)
    categories = _resolve_categories(hm_cfg)

    _log.info("─" * 60)
    _log.info(f"  CLASSIFY HEATMAP  |  mode={mode}")
    _log.info(f"  Slides     : {len(slides)}")
    _log.info(f"  Categories : {categories}")
    _log.info(f"  Alpha      : {alpha}  |  Top-K : {top_k}")
    _log.info(f"  Output     : {heatmap_dir}")
    _log.info("─" * 60)

    processed = skipped = 0

    for slide_id in slides:
        pred_csv   = os.path.join(pred_dir, slide_id + ".csv")
        slide_path = os.path.join(slides_dir, slide_id + slide_ext)

        if not os.path.exists(pred_csv):
            _log.warning(f"[{slide_id}] Prediction CSV not found — skipping.")
            skipped += 1
            continue
        if not os.path.exists(slide_path):
            _log.warning(
                f"[{slide_id}] WSI not found: {slide_path} — skipping.")
            skipped += 1
            continue

        try:
            df = pd.read_csv(pred_csv)
        except Exception as exc:
            _log.error(f"[{slide_id}] Cannot read CSV: {exc}")
            skipped += 1
            continue

        required = {"coord_x", "coord_y", "predicted_label", "confidence"}
        if not required.issubset(df.columns):
            _log.error(
                f"[{slide_id}] CSV missing columns {required - set(df.columns)}. "
                "Re-run classify.")
            skipped += 1
            continue

        coords      = df[["coord_x", "coord_y"]].values.astype(np.int64)
        pred_labels = df["predicted_label"].values

        try:
            wsi, thumb, ds = _open_wsi_with_thumb(slide_path)
        except Exception as exc:
            _log.error(f"[{slide_id}] Cannot open WSI: {exc}")
            skipped += 1
            continue

        slide_out = os.path.join(heatmap_dir, slide_id)
        os.makedirs(slide_out, exist_ok=True)

        try:
            # ── Mode 1: category_map ─────────────────────────────────────────
            if mode == "category_map":
                vis_mask   = np.array([lbl in categories for lbl in pred_labels])
                vis_coords = coords[vis_mask]
                vis_labels = pred_labels[vis_mask]

                if len(vis_coords) == 0:
                    _log.warning(
                        f"  [{slide_id}] No patches match categories {categories}.")
                else:
                    patch_colors = np.array([
                        CATEGORY_COLORS.get(lbl, (200, 200, 200))
                        for lbl in vis_labels], dtype=np.uint8)

                    img  = _render_heatmap(
                        thumb, vis_coords, patch_colors, p_size, ds, alpha)
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10),
                              f"{slide_id}  |  category map  |  "
                              f"{', '.join(categories)}",
                              fill=(255, 255, 255))
                    img = _add_legend(img, categories)
                    out = os.path.join(slide_out, f"{slide_id}_category_map.png")
                    img.save(out)
                    _log.info(f"  [{slide_id}] Category map → {out}")

            # ── Mode 2: confidence ───────────────────────────────────────────
            elif mode == "confidence":
                for cat in categories:
                    if cat not in df.columns:
                        _log.warning(
                            f"  [{slide_id}] Column '{cat}' missing in CSV.")
                        continue

                    confs  = df[cat].values.astype(np.float32)
                    colors = np.array([_jet(c) for c in confs], dtype=np.uint8)

                    img  = _render_heatmap(
                        thumb, coords, colors, p_size, ds, alpha)
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10),
                              f"{slide_id}  |  confidence: {cat}",
                              fill=(255, 255, 255))
                    img = _add_jet_colorbar(img)
                    out = os.path.join(
                        slide_out, f"{slide_id}_confidence_{cat}.png")
                    img.save(out)
                    _log.info(f"  [{slide_id}] Confidence [{cat}] → {out}")

                    # Top-K tiles
                    if top_k > 0:
                        top_dir = os.path.join(slide_out, f"top{top_k}_{cat}")
                        os.makedirs(top_dir, exist_ok=True)
                        for rank, idx in enumerate(
                                np.argsort(confs)[::-1][:top_k]):
                            x, y  = int(coords[idx, 0]), int(coords[idx, 1])
                            score = float(confs[idx])
                            try:
                                tile = wsi.read_region(
                                    (x, y), 0, (p_size, p_size)).convert("RGB")
                                fname = (f"{rank+1:02d}_{slide_id}"
                                         f"_x{x}_y{y}_conf{score:.3f}.png")
                                tile.save(os.path.join(top_dir, fname))
                            except Exception as te:
                                _log.warning(
                                    f"    [{slide_id}] Tile ({x},{y}): {te}")
                        _log.info(
                            f"    Top-{top_k} [{cat}] tiles → {top_dir}")

            else:
                _log.error(
                    f"Unknown mode '{mode}'. Use 'category_map' or 'confidence'.")
                wsi.close()
                return

        finally:
            wsi.close()

        processed += 1

    _log.info(
        f"\nHeatmaps complete | processed={processed} | skipped={skipped}")
    _log.info(f"Output: {heatmap_dir}")
