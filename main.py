import argparse
import os
import glob
from utils.config import load_config, setup_directories
from core.wsi_reader import process_wsi
from utils.logger import setup_logger
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="WSI Classification Framework")
    parser.add_argument("command", choices=["process", "stats", "segment", "extract", "train", "evaluate", "heatmap"],
                        help="Command to execute")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file")
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
        slide_id = os.path.splitext(filename)[0]
        size_mb = os.path.getsize(path) / (1024 * 1024)
        data.append({"slide_id": slide_id, "filename": filename, "size_mb": round(size_mb, 2)})
        
    df = pd.DataFrame(data)
    out_path = os.path.join(config['paths']['results_dir'], "dataset_stats.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved dataset statistics to {out_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    dirs_dict = setup_directories(config)
    
    # Initialize global logger
    logger = setup_logger("wsi_framework", results_dir=config['paths']['results_dir'])
    logger.info(f"Initialized WSI Framework. Running command: {args.command}")
    
    if args.command == "process":
        command_process(config, dirs_dict, logger)
    elif args.command == "stats":
        command_stats(config, logger)
    elif args.command == "segment":
        logger.warning("Segment command not yet implemented.")
    elif args.command == "extract":
        logger.warning("Extract command not yet implemented.")
    elif args.command == "train":
        logger.warning("Train command not yet implemented.")
    elif args.command == "evaluate":
        logger.warning("Evaluate command not yet implemented.")
    elif args.command == "heatmap":
        logger.warning("Heatmap command not yet implemented.")

if __name__ == "__main__":
    main()
