import os
import json
import openslide
from PIL import Image

class WSIReader:
    def __init__(self, slide_path):
        self.slide_path = slide_path
        self.slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        try:
            self.wsi = openslide.open_slide(slide_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open {slide_path}: {e}")
            
    def get_metadata(self):
        """
        Extracts metadata from the WSI.
        """
        properties = dict(self.wsi.properties)
        
        # Calculate level downsamples similar to original Ovarian_Features
        level_downsamples = []
        dim_0 = self.wsi.level_dimensions[0]
        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            # Keep original precision if very close
            if abs(estimated_downsample[0] - downsample) < 0.01:
                level_downsamples.append((downsample, downsample))
            else:
                level_downsamples.append(estimated_downsample)

        metadata = {
            "slide_name": self.slide_name,
            "dimensions": self.wsi.dimensions,
            "level_count": self.wsi.level_count,
            "level_dimensions": self.wsi.level_dimensions,
            "level_downsamples": level_downsamples,
            "vendor": properties.get(openslide.PROPERTY_NAME_VENDOR, "unknown"),
            "mpp_x": properties.get(openslide.PROPERTY_NAME_MPP_X, "unknown"),
            "mpp_y": properties.get(openslide.PROPERTY_NAME_MPP_Y, "unknown")
        }
        return metadata

    def generate_thumbnail(self, max_size=1024):
        """
        Generates a thumbnail of the WSI.
        """
        return self.wsi.get_thumbnail((max_size, max_size))

def process_wsi(slide_path, results_dirs):
    """
    Given a single slide path, extracts metadata and thumbnail, saving them to results_dirs.
    """
    print(f"Processing {slide_path}...")
    reader = WSIReader(slide_path)
    
    # Extract metadata
    metadata = reader.get_metadata()
    metadata_path = os.path.join(results_dirs['metadata'], f"{reader.slide_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Extract thumbnail
    thumbnail = reader.generate_thumbnail()
    thumbnail_path = os.path.join(results_dirs['thumbnails'], f"{reader.slide_name}.png")
    thumbnail.save(thumbnail_path)
    
    print(f"[{reader.slide_name}] Saved metadata and thumbnail.")
