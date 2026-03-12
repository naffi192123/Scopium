import os
import cv2
import h5py
import numpy as np
from PIL import Image

class Patcher:
    """
    Extracts patch coordinates within tissue contours and (optionally) saves them to an HDF5 file.
    Uses robust bounding box and polygon validation for accurate coordinate generation.
    """
    def __init__(self, patch_size, step_size, patch_level, 
                 level_dim, level_downsample, raw_dim,
                 contour_fn="four_pt", use_padding=True, 
                 filter_blank=True, white_thresh=15, black_thresh=40, wsi_reader=None):
        self.patch_size = patch_size
        self.step_size = step_size
        self.patch_level = patch_level
        self.level_dim = level_dim
        self.level_downsample = level_downsample
        self.raw_dim = raw_dim
        
        self.contour_fn = contour_fn
        self.use_padding = use_padding
        self.filter_blank = filter_blank
        self.white_thresh = white_thresh
        self.black_thresh = black_thresh
        self.wsi_reader = wsi_reader
        
        # Calculate exactly how many full-res pixels one patch corresponds to
        self.ref_patch_size = (
            int(self.patch_size * self.level_downsample[0]), 
            int(self.patch_size * self.level_downsample[1])
        )
        self.step_size_x = int(self.step_size * self.level_downsample[0])
        self.step_size_y = int(self.step_size * self.level_downsample[1])

    def get_patch_coordinates(self, contours, holes):
        """
        Calculates all valid top-left (x, y) coordinates of patches contained within contours.
        """
        all_coords = []
        
        for idx, cont in enumerate(contours):
            cont_holes = holes[idx] if holes else []
            
            # Get bounding box of the contour (x, y, w, h)
            start_x, start_y, w, h = cv2.boundingRect(cont)
            
            # Ensure we don't extract patches past the slide boundaries if not padded
            if self.use_padding:
                stop_x = start_x + w
                stop_y = start_y + h
            else:
                stop_x = min(start_x + w, self.raw_dim[0] - self.ref_patch_size[0] + 1)
                stop_y = min(start_y + h, self.raw_dim[1] - self.ref_patch_size[1] + 1)
            
            # Generate grid
            x_range = np.arange(start_x, stop_x, step=self.step_size_x)
            y_range = np.arange(start_y, stop_y, step=self.step_size_y)
            x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
            coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
            
            # Validate each coordinate
            for c in coord_candidates:
                if self._is_valid_contour(c, cont, cont_holes):
                    if self.filter_blank and self.wsi_reader is not None:
                        if not self._is_blank(c):
                            all_coords.append(c)
                    else:
                        all_coords.append(c)
            
        return np.array(all_coords)

    def _is_valid_contour(self, pt, contour, holes):
        """
        Checks if a patch originating at `pt` is primarily inside `contour` and not inside `holes`.
        Supports 'four_pt', 'center', and 'basic' checking modes.
        """
        pt_x, pt_y = int(pt[0]), int(pt[1])
        w, h = self.ref_patch_size[0], self.ref_patch_size[1]
        
        points_to_check = []
        
        if self.contour_fn == 'four_pt' or self.contour_fn == 'four_pt_hard':
            # Check the 4 corners at 0.5 center shift inward
            shift_x, shift_y = w // 4, h // 4
            points_to_check = [
                (pt_x + shift_x, pt_y + shift_y),               # Top-Left
                (pt_x + w - shift_x, pt_y + shift_y),           # Top-Right
                (pt_x + shift_x, pt_y + h - shift_y),           # Bottom-Left
                (pt_x + w - shift_x, pt_y + h - shift_y)        # Bottom-Right
            ]
        elif self.contour_fn == 'center':
            points_to_check = [(pt_x + w // 2, pt_y + h // 2)]
        elif self.contour_fn == 'basic':
            points_to_check = [(pt_x, pt_y)]
        else:
            points_to_check = [(pt_x + w // 2, pt_y + h // 2)] # Default to center
            
        # Count how many points are inside the contour and outside holes
        valid_points = 0
        for test_pt in points_to_check:
            if cv2.pointPolygonTest(contour, test_pt, False) >= 0:
                in_hole = False
                for hole in holes:
                    if cv2.pointPolygonTest(hole, test_pt, False) > 0:
                        in_hole = True
                        break
                if not in_hole:
                    valid_points += 1
                    
        if self.contour_fn == 'four_pt_hard':
            return valid_points == 4
        elif self.contour_fn == 'four_pt':
            return valid_points >= 1
        return valid_points >= 1

    def _is_blank(self, coord):
        """
        Checks if a patch is mostly blank Background (white) or Pen marks/Artifacts (black).
        """
        try:
            patch_pil = self.wsi_reader.wsi.read_region(tuple(coord), self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
            patch_arr = np.array(patch_pil)
            
            # Check White Space (Low Saturation)
            patch_hsv = cv2.cvtColor(patch_arr, cv2.COLOR_RGB2HSV)
            if np.mean(patch_hsv[:,:,1]) < self.white_thresh:
                return True
                
            # Check Black Space (Low RGB mean)
            if np.all(np.mean(patch_arr, axis=(0,1)) < self.black_thresh):
                return True
                
        except Exception:
            return True # If it fails to read, consider it bad/blank
            
        return False

    def save_hdf5(self, coords, slide_path, save_path, wsi_name, extract_patches=False, wsi_reader=None):
        """
        Saves coordinates (and optionally extracted image data) to an HDF5 file.
        """
        file_path = os.path.join(save_path, f"{wsi_name}.h5")
        
        # If there are no coordinates, safely exit
        if len(coords) == 0:
            return file_path
            
        with h5py.File(file_path, "w") as f:
            dset = f.create_dataset('coords', shape=coords.shape, dtype=np.int32)
            dset[:] = coords
            
            # Store metadata
            dset.attrs['patch_size'] = self.patch_size
            dset.attrs['patch_level'] = self.patch_level
            dset.attrs['wsi_name'] = wsi_name
            dset.attrs['downsample'] = self.level_downsample
            dset.attrs['level_dim'] = self.level_dim
            dset.attrs['raw_dim'] = self.raw_dim
            dset.attrs['slide_path'] = slide_path
            
            # Optional: Extract actual patches (.jpg/.png equivalent stored inside the HDF5)
            if extract_patches and wsi_reader is not None:
                img_shape = (self.patch_size, self.patch_size, 3)
                img_dset = f.create_dataset('imgs', shape=(len(coords),) + img_shape, 
                                            chunks=(1,) + img_shape, dtype=np.uint8)
                
                for idx, coord in enumerate(coords):
                    try:
                        patch_pil = wsi_reader.wsi.read_region(tuple(coord), self.patch_level, (self.patch_size, self.patch_size)).convert("RGB")
                        img_dset[idx] = np.array(patch_pil)
                    except Exception as e:
                        pass # Typically handle boundaries gracefully
                        
        return file_path
