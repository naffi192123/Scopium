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

def setup_directories(config, patches_override=None, features_override=None):
    """
    Creates the necessary directories for results as dictated by the config.
    Returns a dictionary of the created paths for easy access across modules.

    Subfolder Naming Rules
    ----------------------
    **Patches** subfolder (used by ``segment`` and ``extract``):

      Default auto-name:  ``patches/patch{size}_step{step}_level{lvl}/``
      Override (YAML):    ``tiling.patches_subfolder_override: "my_name"``
      Override (CLI):     ``--patches my_name``  (highest priority)

    **Masks** subfolder mirrors the patches subfolder so each tiling run
    keeps its own mask set:

      Default:  ``masks/patch{size}_step{step}_level{lvl}/``

    **Features** subfolder (used by ``extract``, ``train``, ``evaluate``, ``heatmap``):

      The model name is **always appended** (``__{model}``) to both the
      auto-derived name and any override value, ensuring each
      (patches_config, model) combination gets its own directory.

      Default auto-name:  ``features/patch{size}_step{step}_level{lvl}__{model}/``
      Override (YAML):    ``feature_extraction.features_subfolder_override: "my_base"``
                          → resolves to ``features/my_base__{model}/``
      Override (CLI):     ``--features my_base``
                          → resolves to ``features/my_base__{model}/``

    Override priorities (highest → lowest):
      1. CLI keyword argument (``patches_override`` / ``features_override``)
      2. YAML config key     (``patches_subfolder_override`` / ``features_subfolder_override``)
      3. Auto-derived name
    """
    slides_dir   = config['paths']['slides_dir']
    results_root = config['paths']['results_dir']

    # Create dataset dirs if they don't exist
    os.makedirs(slides_dir, exist_ok=True)
    if 'annotations_dir' in config['paths'] and config['paths']['annotations_dir']:
        os.makedirs(config['paths']['annotations_dir'], exist_ok=True)

    # Create main results root
    os.makedirs(results_root, exist_ok=True)

    # ── Auto-derive canonical subfolder names ──────────────────────────────────
    p_size  = config['tiling'].get('patch_size', 512)
    s_size  = config['tiling'].get('step_size', 512)
    lvl     = config['tiling'].get('patch_level', 0)
    f_model = config['feature_extraction']['model']

    auto_patch_str = f"patch{p_size}_step{s_size}_level{lvl}"

    # ── Resolve patches subfolder ──────────────────────────────────────────────
    # Priority: CLI arg > config key > auto-derived
    if patches_override:
        patch_str = os.path.basename(patches_override.rstrip("\\/"))
    else:
        patch_str = config['tiling'].get('patches_subfolder_override') or auto_patch_str

    # ── Resolve features subfolder ─────────────────────────────────────────────
    # The model name is ALWAYS appended so different backbones never share a dir.
    # The override (CLI or config key) sets the BASE name; model is auto-appended.
    if features_override:
        feature_base = os.path.basename(features_override.rstrip("\\/"))
    else:
        feature_base = config['feature_extraction'].get('features_subfolder_override') or patch_str

    feature_str = f"{feature_base}__{f_model}"

    # ── Build and create all result subdirectories ─────────────────────────────
    subdirs = {
        'thumbnails': 'thumbnails',
        'metadata':   'metadata',
        # Masks are stored alongside their corresponding patch config
        'masks':      os.path.join('masks', patch_str),
        'patches':    os.path.join('patches', patch_str),
        'features':   os.path.join('features', feature_str),
        'experiments':'experiments',
        'heatmaps':   'heatmaps',
        'debug':      'debug',
    }

    dirs_dict = {}
    for key, subdir in subdirs.items():
        path = os.path.join(results_root, subdir)
        os.makedirs(path, exist_ok=True)
        dirs_dict[key] = path

    # Expose the resolved subfolder names so downstream code can log them
    dirs_dict['_patch_subfolder']   = patch_str
    dirs_dict['_feature_subfolder'] = feature_str

    return dirs_dict
