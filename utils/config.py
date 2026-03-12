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
    
    # Define and create all required subdirectories conceptually mapped to pipeline stages
    subdirs = [
        'thumbnails', 
        'metadata', 
        'segmentation', 
        'patches', 
        'features', 
        'splits',
        'models', 
        'heatmaps', 
        'logs'
    ]
    
    dirs_dict = {}
    for subdir in subdirs:
        path = os.path.join(results_root, subdir)
        os.makedirs(path, exist_ok=True)
        dirs_dict[subdir] = path
        
    # Example for experiment-specific nested folders
    if 'experiment' in config and config['experiment'].get('name'):
        exp_name = config['experiment']['name']
        exp_models_dir = os.path.join(dirs_dict['models'], exp_name)
        exp_logs_dir = os.path.join(dirs_dict['logs'], exp_name)
        os.makedirs(exp_models_dir, exist_ok=True)
        os.makedirs(exp_logs_dir, exist_ok=True)
        dirs_dict['exp_models'] = exp_models_dir
        dirs_dict['exp_logs'] = exp_logs_dir
        
    return dirs_dict
