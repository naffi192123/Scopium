import argparse
import os
import glob
from utils.config import load_config, setup_directories
from core.wsi_reader import process_wsi

def parse_args():
    parser = argparse.ArgumentParser(description="WSI Classification Framework")
    parser.add_argument("command", choices=["process", "stats", "segment", "extract", "train", "evaluate", "heatmap"],
                        help="Command to execute")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the config file")
    return parser.parse_args()

def command_process(config, dirs_dict):
    """
    The 'process' command reads all WSIs in the dataset/slides directory,
    extracts metadata, and generates thumbnails.
    """
    slides_dir = config['dataset']['slides_dir']
    
    # Accept common slide extensions
    extensions = ['*.svs', '*.tif', '*.ndpi', '*.vms', '*.vmu', '*.scn', '*.mrxs', '*.tiff', '*.bif']
    slide_paths = []
    
    # Handle paths cross-platform and glob pattern matching
    for ext in extensions:
        search_pattern = os.path.join(slides_dir, ext)
        slide_paths.extend(glob.glob(search_pattern))
    
    if not slide_paths:
        print(f"No slides found in {slides_dir}. Please place some WSIs there to test.")
        return
        
    for slide_path in slide_paths:
        try:
            process_wsi(slide_path, dirs_dict)
        except Exception as e:
            print(f"Error processing {slide_path}: {e}")

def main():
    args = parse_args()
    config = load_config(args.config)
    dirs_dict = setup_directories(config)
    
    if args.command == "process":
        command_process(config, dirs_dict)
    elif args.command == "stats":
        print("Stats command not yet implemented.")
    elif args.command == "segment":
        print("Segment command not yet implemented.")
    elif args.command == "extract":
        print("Extract command not yet implemented.")
    elif args.command == "train":
        print("Train command not yet implemented.")
    elif args.command == "evaluate":
        print("Evaluate command not yet implemented.")
    elif args.command == "heatmap":
        print("Heatmap command not yet implemented.")

if __name__ == "__main__":
    main()
