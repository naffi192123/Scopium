import argparse
import os
import glob
from utils.config import load_config, setup_directories
from core.wsi_reader import process_wsi
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

def command_process(config, dirs_dict):
    """
    The 'process' command reads all WSIs in the dataset/slides directory,
    extracts metadata, and generates thumbnails.
    """
    slides_dir, slide_paths = get_slide_paths(config)

    
    if not slide_paths:
        print(f"No slides found in {slides_dir}. Please place some WSIs there to test.")
        return
        
    for slide_path in slide_paths:
        try:
            process_wsi(slide_path, dirs_dict)
        except Exception as e:
            print(f"Error processing {slide_path}: {e}")

def command_stats(config):
    """
    Generates dataset_stats.csv indicating the slides found.
    """
    slides_dir, slide_paths = get_slide_paths(config)
    
    if not slide_paths:
        print(f"No slides found in {slides_dir}. Cannot generate stats.")
        return
        
    print(f"Found {len(slide_paths)} slides in {slides_dir}.")
    
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
    print(f"Saved dataset statistics to {out_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    dirs_dict = setup_directories(config)
    
    if args.command == "process":
        command_process(config, dirs_dict)
    elif args.command == "stats":
        command_stats(config)
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
