import os
import h5py
import cv2
import numpy as np
from core.wsi_reader import WSIReader

def command_debug_segmentation(config, dirs_dict, logger):
    """
    Reads corresponding .h5 coordinate file and plots the extracted tiles 
    onto a WSI thumbnail for visualization.
    """
    slides_dir = config['paths']['slides_dir']
    slide_files = [f for f in os.listdir(slides_dir) if f.endswith(config['dataset']['slide_extension'])]
    
    if not slide_files:
        logger.error("No slides found in the dataset directory.")
        return
        
    # We will just debug the first slide found for simplicity
    slide_name = os.path.splitext(slide_files[0])[0]
    slide_path = os.path.join(slides_dir, slide_files[0])
    
    # Locate the H5 file
    h5_path = os.path.join(dirs_dict['patches'], f"{slide_name}.h5")
    if not os.path.exists(h5_path):
        logger.error(f"Cannot find patch coordinate file at {h5_path}. Did you run 'segment' first?")
        return
        
    logger.info(f"Reconstructing tissue coverage map for {slide_name}...")
    
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_size = f['coords'].attrs['patch_size']
        patch_level = f['coords'].attrs['patch_level']
        
    if len(coords) == 0:
        logger.warning(f"No coordinates found in {h5_path}.")
        return

    # Load thumbnail
    reader = WSIReader(slide_path)
    meta = reader.get_metadata()
    
    # 64x downsample is generally good for visualization
    vis_level = reader.wsi.get_best_level_for_downsample(64)
    img_rgb = np.array(reader.wsi.read_region((0, 0), vis_level, reader.wsi.level_dimensions[vis_level]).convert("RGB"))
    
    # Coordinates in H5 are at Level 0 (Raw). We need to scale them down to vis_level
    downsample_x = meta['level_downsamples'][vis_level][0]
    downsample_y = meta['level_downsamples'][vis_level][1]
    
    vis_patch_w = int(patch_size * meta['level_downsamples'][patch_level][0] / downsample_x)
    vis_patch_h = int(patch_size * meta['level_downsamples'][patch_level][1] / downsample_y)

    # Draw Rectangles
    for coord in coords:
        start_x = int(coord[0] / downsample_x)
        start_y = int(coord[1] / downsample_y)
        cv2.rectangle(img_rgb, (start_x, start_y), (start_x + vis_patch_w, start_y + vis_patch_h), (0, 255, 0), 1)
        
    out_path = os.path.join(dirs_dict['debug'], f"segmented_tiles_thumbnail_{slide_name}.png")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved segmentation debug map to {out_path}")


def command_extract_tile(config, dirs_dict, logger):
    """
    Extracts a single tile from the first found WSI using its H5 coordinates.
    """
    slides_dir = config['paths']['slides_dir']
    slide_files = [f for f in os.listdir(slides_dir) if f.endswith(config['dataset']['slide_extension'])]
    
    if not slide_files:
        logger.error("No slides found in the dataset directory.")
        return
        
    slide_name = os.path.splitext(slide_files[0])[0]
    slide_path = os.path.join(slides_dir, slide_files[0])
    
    h5_path = os.path.join(dirs_dict['patches'], f"{slide_name}.h5")
    if not os.path.exists(h5_path):
        logger.error(f"Cannot find patch coordinate file at {h5_path}. Did you run 'segment' first?")
        return
        
    logger.info(f"Extracting a sample tile from {slide_name}...")
    
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_size = f['coords'].attrs['patch_size']
        patch_level = f['coords'].attrs['patch_level']

    if len(coords) == 0:
        logger.warning(f"No coordinates found in {h5_path}.")
        return

    # Grab a coordinate from the middle of the array to increase chances 
    # of grabbing a dense tissue spot instead of a border.
    mid_idx = len(coords) // 2
    coord = coords[mid_idx]
    
    reader = WSIReader(slide_path)
    
    try:
        patch_pil = reader.wsi.read_region(tuple(coord), patch_level, (patch_size, patch_size)).convert("RGB")
        out_path = os.path.join(dirs_dict['debug'], f"tile_{slide_name}_{coord[0]}_{coord[1]}.png")
        patch_pil.save(out_path)
        logger.info(f"Saved sample tile to {out_path}")
    except Exception as e:
        logger.error(f"Failed to extract tile at {coord}: {e}")
