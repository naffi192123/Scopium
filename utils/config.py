import yaml
import os
import sys

def load_config(config_path):
    """
    Parse a YAML configuration file to a Python dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    validate_config(config)
    return config

def validate_config(config):
    """
    Validates the presence of required fields in the config.
    """
    missing_fields = []
    
    # Check paths
    if 'paths' not in config:
        missing_fields.append('paths')
    else:
        if 'slides_dir' not in config['paths']:
            missing_fields.append('paths.slides_dir')
        if 'results_dir' not in config['paths']:
            missing_fields.append('paths.results_dir')
            
    # Check task definition
    if 'task' not in config:
        missing_fields.append('task')
    else:
        if 'name' not in config['task']:
            missing_fields.append('task.name')

    if missing_fields:
        print("ERROR: Missing required fields in configuration YAML:", file=sys.stderr)
        for field in missing_fields:
            print(f"  - {field}", file=sys.stderr)
        sys.exit(1)

def setup_directories(config):
    """
    Creates the necessary directories for results as dictated by the config.
    Returns a dictionary of the created paths for easy access across modules.
    """
    slides_dir = config['paths']['slides_dir']
    results_root = config['paths']['results_dir']
    
    # Create dataset dirs if they don't exist
    os.makedirs(slides_dir, exist_ok=True)
    if 'annotations_dir' in config['paths'] and config['paths']['annotations_dir']:
        os.makedirs(config['paths']['annotations_dir'], exist_ok=True)
    
    # Create main results root
    os.makedirs(results_root, exist_ok=True)
    
    # Define variables for dynamic paths
    p_size = config['tiling']['patch_size']
    s_size = config['tiling']['step_size']
    lvl = config['tiling']['level']
    f_model = config['feature_extraction']['model']
    
    patch_str = f"patch{p_size}_step{s_size}_level{lvl}"
    feature_str = f"{patch_str}__{f_model}"
    
    # Define and create all required subdirectories conceptually mapped to pipeline stages
    subdirs = {
        'thumbnails': 'thumbnails',
        'metadata': 'metadata',
        'masks': 'masks',
        'patches': os.path.join('patches', patch_str),
        'features': os.path.join('features', feature_str),
        'experiments': 'experiments',
        'heatmaps': 'heatmaps'
    }
    
    dirs_dict = {}
    for key, subdir in subdirs.items():
        path = os.path.join(results_root, subdir)
        os.makedirs(path, exist_ok=True)
        dirs_dict[key] = path
        
    return dirs_dict
