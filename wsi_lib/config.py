import yaml
import os
from pathlib import Path

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """
    Creates the necessary directories for results as per the config.
    Returns the paths.
    """
    # Create dataset dirs if they don't exist (helpful for local test setup)
    os.makedirs(config['dataset']['slides_dir'], exist_ok=True)
    os.makedirs(config['dataset']['annotations_dir'], exist_ok=True)
    
    # Create results dirs
    results_dir = config['results_dir']
    subdirs = ['thumbnails', 'metadata', 'segmentation', 'patches', 'features', 'models', 'heatmaps', 'logs']
    
    dirs_dict = {}
    for subdir in subdirs:
        path = os.path.join(results_dir, subdir)
        os.makedirs(path, exist_ok=True)
        dirs_dict[subdir] = path
        
    return dirs_dict
