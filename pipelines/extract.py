import os
import time
import torch
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.feature_models import get_feature_extractor, forward_features
from datasets.slide_dataset import WSIPatchDataset

def command_extract(config, dirs_dict, logger):
    """
    Orchestrates the feature extraction pipeline over all WSI patches.
    """
    # 1. Setup Environment
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Initialized Feature Extraction. Hardware detected: {device}")
    if device.type == "cuda":
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
    feat_config = config.get('feature_extraction', {})
    model_name = feat_config.get('model', 'rn50')
    batch_size = feat_config.get('batch_size', 256)
    num_workers = feat_config.get('num_workers', 4)
    
    # Check for custom weights folder in the ENV
    weights_dir = os.environ.get("WSI_WEIGHTS_DIR", None)
    
    # 2. Load the specific architecture
    try:
        model = get_feature_extractor(model_name, device, weights_dir=weights_dir)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return
        
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"Model wrapped in DataParallel across {torch.cuda.device_count()} GPUs.")
        
    # Directories
    slides_dir = config['paths']['slides_dir']
    patches_dir = dirs_dict['patches']
    features_dir = dirs_dict['features']
    
    pt_dir = os.path.join(features_dir, 'pt_files')
    os.makedirs(pt_dir, exist_ok=True)
    
    # 3. Locate processing files
    h5_files = [f for f in os.listdir(patches_dir) if f.endswith('.h5')]
    if not h5_files:
        logger.error(f"No patch files found in {patches_dir}. Run `segment` first.")
        return
        
    logger.info(f"Found {len(h5_files)} chunked WSI files to embed.")
    
    success, failed = 0, 0
    total_time = 0.0
    
    # 4. Processing Loop
    for h5_file in h5_files:
        slide_name = os.path.splitext(h5_file)[0]
        h5_path = os.path.join(patches_dir, h5_file)
        
        # We need the original slide path to map coordinates natively
        slide_ext = config['dataset']['slide_extension']
        slide_path = os.path.join(slides_dir, slide_name + slide_ext)
        if not os.path.exists(slide_path):
            logger.warning(f"Could not find matching slide {slide_path} for coordinates. Skipping.")
            failed += 1
            continue
            
        pt_path = os.path.join(pt_dir, slide_name + '.pt')
        if os.path.exists(pt_path):
            logger.info(f"Skipping {slide_name}, already embedded.")
            continue
            
        # Initialize the lazy Dataset loader
        logger.info(f"[{slide_name}] Creating Dataset mapping...")
        try:
            dataset = WSIPatchDataset(h5_path, slide_path, config)
        except Exception as e:
            logger.error(f"[{slide_name}] Failed to open dataset natively: {e}")
            failed += 1
            continue
            
        if len(dataset) == 0:
            logger.warning(f"[{slide_name}] Coordinate dataset is empty (no valid tissue).")
            # Save empty tensor tracker
            torch.save(torch.empty((0,1)), pt_path)
            continue
            
        loader_kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
        
        logger.info(f"[{slide_name}] Beginning Extraction | Patches: {len(dataset)} | Batches: {len(loader)}")
        
        slide_features = []
        slide_start_time = time.time()
        
        try:
            # Iterating through dataloader pushes batches to GPU and calls eval
            for batch_img, coords in tqdm(loader, desc=f"{slide_name} GPU", leave=False):
                # Images loaded natively via WSIReader/PIL
                batch_img = batch_img.to(device, non_blocking=True)
                
                # Squeeze through the loaded model architecture
                batch_features = forward_features(model, batch_img, model_name)
                slide_features.append(batch_features)
                
        except KeyboardInterrupt:
            logger.critical("Process interrupted by user.")
            return
        except Exception as e:
            logger.error(f"[{slide_name}] GPU exception during batch forward pass: {e}")
            failed += 1
            continue
            
        slide_elapsed = time.time() - slide_start_time
        total_time += slide_elapsed
        
        # Concatenate and save the NxD feature tensor to disk
        slide_features = torch.cat(slide_features, dim=0)
        torch.save(slide_features, pt_path)
        
        logger.info(f"[{slide_name}] Finished! Shape: {slide_features.shape} | Time: {slide_elapsed:.2f}s")
        success += 1
        
    logger.info(f"Extraction Complete. Success: {success} | Failed: {failed} | Total Time: {total_time:.2f}s")
    logger.info(f"Embeddings saved to: {pt_dir}")
