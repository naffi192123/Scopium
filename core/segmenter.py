import cv2
import numpy as np

class TissueSegmenter:
    """
    Handles extracting tissue contours from a Whole Slide Image thumbnail.
    Uses HSV color space conversion -> Median Blur -> Otsu/Binary Thresholding.
    """
    def __init__(self, mthresh=7, sthresh=8, sthresh_up=255, close=4, use_otsu=False, 
                 filter_params={'a_t':100, 'a_h':16, 'max_n_holes':8},
                 ref_patch_size=512, seg_level_downsample=(1.0, 1.0)):
        self.mthresh = mthresh
        self.sthresh = sthresh
        self.sthresh_up = sthresh_up
        self.close = close
        self.use_otsu = use_otsu
        
        # Scale the area thresholds by the downsample to get accurate pixel measurements 
        # for filtering directly on the segmentation thumbnail.
        scaled_ref_patch_area = int(ref_patch_size**2 / (seg_level_downsample[0] * seg_level_downsample[1]))
        
        self.filter_params = filter_params.copy()
        self.filter_params['a_t'] = filter_params.get('a_t', 100) * scaled_ref_patch_area
        self.filter_params['a_h'] = filter_params.get('a_h', 16) * scaled_ref_patch_area
        self.filter_params['max_n_holes'] = filter_params.get('max_n_holes', 8)

    def segment(self, img_rgb):
        """
        Segments a low-resolution RGB image of the WSI.
        Returns:
            foreground_contours: list of np.array (valid tissue regions)
            hole_contours: list of lists (holes inside each tissue region)
        """
        # Convert to HSV and extract Saturation channel
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:,:,1], self.mthresh)
        
        # Thresholding
        if self.use_otsu:
            _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing to fill small gaps
        if self.close > 0:
            kernel = np.ones((self.close, self.close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        # Find all contours and hierarchy
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return [], []
            
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        return self._filter_contours(contours, hierarchy)

    def _filter_contours(self, contours, hierarchy):
        """
        Filters contours based on area parameters.
        """
        filtered_foreground = []
        all_holes = []
        
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        
        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            
            a = cv2.contourArea(cont)
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            
            a = a - np.array(hole_areas).sum()
            if a == 0: continue
            
            if a > self.filter_params.get('a_t', 0):
                filtered_foreground.append(cont)
                
                # Process holes for this contour
                unfiltered_holes = [contours[idx] for idx in holes]
                unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfiltered_holes = unfiltered_holes[:self.filter_params.get('max_n_holes', 8)]
                
                valid_holes = [h for h in unfiltered_holes if cv2.contourArea(h) > self.filter_params.get('a_h', 0)]
                all_holes.append(valid_holes)

        return filtered_foreground, all_holes
