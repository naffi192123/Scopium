import os
import concurrent.futures
from core.wsi_reader import WSIReader
from core.segmenter import TissueSegmenter
from core.patcher import Patcher
import cv2
import numpy as np

def _process_single_slide(slide_path, config, dirs_dict, logger):
    """
    Worker function to process a single WSI: Read -> Segment -> Patch -> Save.
    """
    try:
        reader = WSIReader(slide_path)
        logger.info(f"[{reader.slide_name}] Initializing Processing...")
        
        # Determine downsample scales
        seg_param = config.get('segmentation', {})
        tiling_param = config.get('tiling', {})
        filter_param = config.get('filter', {})

        seg_level = seg_param.get('seg_level', -1)
        if seg_level < 0:
            # -1 = auto ~64x downsample
            seg_level = reader.wsi.get_best_level_for_downsample(64)
            
        vis_level = seg_param.get('vis_level', -1)
        if vis_level < 0:
            vis_level = reader.wsi.get_best_level_for_downsample(64)
            
        meta = reader.get_metadata()
        
        # 1. Segmentation
        segmenter = TissueSegmenter(
            mthresh=seg_param.get('mthresh', 7), 
            sthresh=seg_param.get('sthresh', 8), 
            close=seg_param.get('close', 4),
            use_otsu=seg_param.get('use_otsu', False), 
            filter_params=filter_param,
            ref_patch_size=filter_param.get('ref_patch_size', 512),
            seg_level_downsample=meta['level_downsamples'][seg_level]
        )
        
        # Get thumbnail for segmentation
        img_rgb = np.array(reader.wsi.read_region((0,0), seg_level, reader.wsi.level_dimensions[seg_level]).convert('RGB'))
        foreground_contours, hole_contours = segmenter.segment(img_rgb)
        logger.debug(f"[{reader.slide_name}] Tissue segmented at level {seg_level}. Found {len(foreground_contours)} tissue regions.")
        
        # Draw and Save Mask
        # Use vis_level for the output mask to save on memory
        img_vis = np.array(reader.wsi.read_region((0,0), vis_level, reader.wsi.level_dimensions[vis_level]).convert('RGB'))
        mask_path = os.path.join(dirs_dict['masks'], f"{reader.slide_name}_mask.png")
        
        scale_vis = [
            meta['level_downsamples'][seg_level][0] / meta['level_downsamples'][vis_level][0],
            meta['level_downsamples'][seg_level][1] / meta['level_downsamples'][vis_level][1]
        ]
        
        thickness = seg_param.get('line_thickness', 25)
        if len(foreground_contours) > 0:
            vis_contours = [np.array(c * scale_vis, dtype=np.int32) for c in foreground_contours]
            cv2.drawContours(img_vis, vis_contours, -1, (0, 255, 0), thickness)
            for holes in hole_contours:
                vis_holes = [np.array(h * scale_vis, dtype=np.int32) for h in holes]
                cv2.drawContours(img_vis, vis_holes, -1, (0, 0, 255), thickness)
        cv2.imwrite(mask_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        
        # 2. Patching
        p_size = tiling_param.get('patch_size', 512)
        s_size = tiling_param.get('step_size', 512)
        p_lvl = tiling_param.get('patch_level', 0)
        
        meta = reader.get_metadata()
        
        patcher = Patcher(
            patch_size=p_size,
            step_size=s_size,
            patch_level=p_lvl,
            level_dim=reader.wsi.level_dimensions[p_lvl],
            level_downsample=meta['level_downsamples'][p_lvl],
            raw_dim=reader.wsi.level_dimensions[0],
            contour_fn=tiling_param.get('contour_fn', 'four_pt'),
            use_padding=tiling_param.get('use_padding', True),
            filter_blank=tiling_param.get('filter_blank', True),
            white_thresh=tiling_param.get('white_thresh', 15),
            black_thresh=tiling_param.get('black_thresh', 40),
            wsi_reader=reader  # Passing reader down for blank filtering if needed
        )
        
        # Convert contours from seg_level downsample back to level 0 (raw) downsample
        # so Patcher can process them against raw coords reliably.
        scale_to_raw = [
            meta['level_downsamples'][seg_level][0],
             meta['level_downsamples'][seg_level][1]
        ]
        raw_contours = [np.array(c * scale_to_raw, dtype=np.int32) for c in foreground_contours]
        raw_holes = [[np.array(h * scale_to_raw, dtype=np.int32) for h in holes] for holes in hole_contours]
        
        coords = patcher.get_patch_coordinates(raw_contours, raw_holes)
        
        if len(coords) == 0:
            logger.warning(f"[{reader.slide_name}] No valid patches found inside the tissue contours.")
            return

        # 3. Patch Extraction / Metadata Saving
        # Instead of unpacking the whole slide to .jpg files, we pack the coordinates into .h5
        logger.debug(f"[{reader.slide_name}] Extracted {len(coords)} patch coordinates. Saving to HDF5.")
        h5_path = patcher.save_hdf5(
            coords=coords,
            slide_path=slide_path,
            save_path=dirs_dict['patches'],
            wsi_name=reader.slide_name,
            extract_patches=False, # Standard is keeping them light as coordinates
            wsi_reader=reader
        )
        
        logger.info(f"[{reader.slide_name}] Finished successfully. Saved patches to {h5_path}")
        
    except Exception as e:
        logger.error(f"[{os.path.basename(slide_path)}] Failed: {str(e)}")


def run_segment_and_patch(config, dirs_dict, slide_paths, logger):
    """
    Executes the segmentation and patching pipeline in either sequential or parallel mode.
    """
    mode = config['tiling'].get('mode', 'sequential').lower()
    
    if mode == 'parallel':
        num_workers = config['tiling'].get('num_workers', 4)
        logger.info(f"Running Segmentation & Patching Pipeline in PARALLEL mode with {num_workers} workers.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for sp in slide_paths:
                futures.append(executor.submit(_process_single_slide, sp, config, dirs_dict, logger))
            
            for future in concurrent.futures.as_completed(futures):
                future.result() # Will raise any unhandled exceptions from the threads

    else:
        logger.info(f"Running Segmentation & Patching Pipeline in SEQUENTIAL mode.")
        for sp in slide_paths:
            _process_single_slide(sp, config, dirs_dict, logger)
