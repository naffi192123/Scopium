import argparse
import os
import glob
from utils.config import load_config, setup_directories
from core.wsi_reader import process_wsi, WSIReader
from pipelines.preprocess import run_segment_and_patch
from pipelines.debug import command_debug_segmentation, command_extract_tile
from pipelines.extract import command_extract
from pipelines.analyse_annotations import run_analysis
from pipelines.split import command_split
from pipelines.train import command_train
from pipelines.evaluate import command_evaluate
from pipelines.visualize import command_heatmap
from pipelines.classify import command_classify
from pipelines.classify_heatmap import command_classify_heatmap
from pipelines.hpo import command_hpo, load_best_config
from pipelines.crossval import command_crossval
from utils.logger import setup_logger
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="WSI Classification Framework")
    parser.add_argument("command", choices=[
        "process", "stats", "segment", "debug-segmentation", "extract-tile",
        "extract", "analyse", "split", "train", "evaluate", "heatmap",
        "classify", "classify-heatmap", "hpo", "crossval"],
        help="Command to execute")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file")
    parser.add_argument("--csv", type=str, default=None, help="[analyse/split] Path to annotation CSV")
    parser.add_argument("--experiment", type=str, default=None,
                        help="[evaluate/heatmap] Path to experiment dir (auto-detect latest if omitted)")
    parser.add_argument(
        "--patches", type=str, default=None,
        metavar="SUBFOLDER",
        help="[segment/extract] Override the patch subfolder name (or full path) inside "
             "results/patches/. Takes priority over tiling.patches_subfolder_override in config.yaml. "
             "Example: --patches patch256_step256_level0")
    parser.add_argument(
        "--features", type=str, default=None,
        metavar="SUBFOLDER",
        help="[extract/train/evaluate/heatmap/classify] Exact feature subfolder name inside "
             "results/features/. This value is used verbatim — the model key is NOT auto-appended. "
             "Overrides feature_extraction.features_subfolder_override / features_dir_override in config.yaml. "
             "Example: --features patch512_step512_level0__uni")
    parser.add_argument(
        "--feature_dir", type=str, default=None,
        metavar="SUBFOLDER",
        help="Alias for --features (both are treated identically as exact subfolder name).")
    parser.add_argument(
        "--slide", type=str, default=None,
        metavar="SLIDE_ID",
        help="[classify-heatmap] Process only this single slide ID "
             "(without file extension). Overrides patch_classifier.heatmap.slides in config.")
    parser.add_argument(
        "--use_best_config", action="store_true",
        help="[train] Load best_config.yaml from the HPO results directory and "
             "merge it into config before training. Run `hpo` first to generate it.")
    return parser.parse_args()

def get_slide_paths(config):
    slides_dir = config['paths']['slides_dir']
    extensions = ['*.svs', '*.tif', '*.ndpi', '*.vms', '*.vmu', '*.scn', '*.mrxs', '*.tiff', '*.bif']
    slide_paths = []
    for ext in extensions:
        search_pattern = os.path.join(slides_dir, ext)
        slide_paths.extend(glob.glob(search_pattern))
    return slides_dir, slide_paths

def command_process(config, dirs_dict, logger):
    """
    The 'process' command reads all WSIs in the dataset/slides directory,
    extracts metadata, and generates thumbnails.
    """
    slides_dir, slide_paths = get_slide_paths(config)

    if not slide_paths:
        logger.error(f"No slides found in {slides_dir}. Please place some WSIs there to test.")
        return
        
    logger.info(f"Starting processing for {len(slide_paths)} slides...")
    for slide_path in slide_paths:
        try:
            process_wsi(slide_path, dirs_dict, logger)
            logger.info(f"Successfully processed {slide_path}")
        except Exception as e:
            logger.error(f"Error processing {slide_path}: {e}")

def command_stats(config, logger):
    """
    Generates dataset_stats.csv indicating the slides found.
    """
    slides_dir, slide_paths = get_slide_paths(config)
    
    if not slide_paths:
        logger.warning(f"No slides found in {slides_dir}. Cannot generate stats.")
        return
        
    logger.info(f"Found {len(slide_paths)} slides in {slides_dir}.")
    
    # Generate basic stats dataframe
    data = []
    for path in slide_paths:
        filename = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        slide_info = {
            "slide_name": os.path.splitext(filename)[0],
            "filename": filename,
            "size_mb": round(size_mb, 2),
            "dimensions": "Error",
            "level_count": "Error",
            "mpp_x": "Error",
            "mpp_y": "Error"
        }
        
        try:
            reader = WSIReader(path)
            meta = reader.get_metadata()
            slide_info["dimensions"] = str(meta.get("dimensions", ""))
            slide_info["level_count"] = meta.get("level_count", "")
            slide_info["mpp_x"] = meta.get("mpp_x", "")
            slide_info["mpp_y"] = meta.get("mpp_y", "")
        except Exception as e:
            logger.error(f"Failed to read metadata for {filename}: {e}")
            
        data.append(slide_info)
        
    df = pd.DataFrame(data)
    out_path = os.path.join(config['paths']['results_dir'], "dataset_stats.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved dataset statistics to {out_path}")

def command_segment(config, dirs_dict, logger):
    """
    Runs the segmentation and patch coordinate extraction based on YAML configs.
    """
    slides_dir, slide_paths = get_slide_paths(config)
    if not slide_paths:
        logger.error(f"No slides found in {slides_dir}. Cannot run segmentation.")
        return
        
    run_segment_and_patch(config, dirs_dict, slide_paths, logger)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Both --features and --feature_dir are treated as exact subfolder names
    # (model is NOT auto-appended). --feature_dir takes priority if both given.
    features_override = args.feature_dir or args.features

    dirs_dict = setup_directories(
        config,
        patches_override=args.patches,
        features_override=features_override,
    )

    # Initialize global logger
    logger = setup_logger("wsi_framework", results_dir=config['paths']['results_dir'])
    logger.info(f"Initialized WSI Framework. Running command: {args.command}")
    if args.patches:
        logger.info(f"  Patch subfolder    : {dirs_dict['patches']}")
    feat_ovr = getattr(args, 'feature_dir', None) or getattr(args, 'features', None)
    if feat_ovr:
        logger.info(f"  Feature dir override → {dirs_dict['features']}")
    else:
        logger.info(f"  Feature dir (auto)   : {dirs_dict['features']}")
    
    if args.command == "process":
        command_process(config, dirs_dict, logger)
    elif args.command == "stats":
        command_stats(config, logger)
    elif args.command == "segment":
        command_segment(config, dirs_dict, logger)
    elif args.command == "debug-segmentation":
        command_debug_segmentation(config, dirs_dict, logger)
    elif args.command == "extract-tile":
        command_extract_tile(config, dirs_dict, logger)
    elif args.command == "extract":
        command_extract(config, dirs_dict, logger)
    elif args.command == "analyse":
        run_analysis(args.config, csv_path=args.csv)
    elif args.command == "split":
        command_split(config, dirs_dict, logger, csv_path=args.csv)
    elif args.command == "train":
        # Optionally merge best HPO config before training
        if getattr(args, 'use_best_config', False):
            config = load_best_config(config, log=logger)
        command_train(config, dirs_dict, logger)
    elif args.command == "evaluate":
        command_evaluate(config, dirs_dict, logger, experiment_dir=args.experiment)
    elif args.command == "heatmap":
        command_heatmap(config, dirs_dict, logger, experiment_dir=args.experiment)
    elif args.command == "classify":
        command_classify(config, dirs_dict, logger)
    elif args.command == "classify-heatmap":
        command_classify_heatmap(config, dirs_dict, logger,
                                 slide_override=args.slide)
    elif args.command == "hpo":
        command_hpo(config, dirs_dict, logger)
    elif args.command == "crossval":
        # Optionally merge best HPO config before cross-validation
        if getattr(args, 'use_best_config', False):
            config = load_best_config(config, log=logger)
        command_crossval(config, dirs_dict, logger)

if __name__ == "__main__":
    main()
