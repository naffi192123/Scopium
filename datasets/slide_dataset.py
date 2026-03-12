import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image

from core.wsi_reader import WSIReader
from utils.transforms import build_transform_pipeline

class WSIPatchDataset(Dataset):
    """
    A PyTorch Dataset that maps directly over the extracted HDF5 patch coordinates.
    Natively opens the corresponding WSI slide, extracts the high-resolution PIL tile, 
    and applies the composed user-defined Torchvision transformations.
    """
    def __init__(self, h5_path, slide_path, config):
        """
        Args:
            h5_path (str): Path to the generated {slide_name}.h5 coordinate file.
            slide_path (str): Path to the raw WSI image (e.g. .svs).
            config (dict): The entire loaded YAML configuration dictionary.
        """
        self.h5_path = h5_path
        self.slide_path = slide_path
        
        # Load the reader (OpenSlide obj inside)
        self.reader = WSIReader(self.slide_path)
        
        # Parse the transform pipeline from config
        self.transforms = build_transform_pipeline(config)
        
        # Extract attributes off the H5
        with h5py.File(self.h5_path, "r") as f:
            self.coords = f['coords'][:]
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            
        self.length = len(self.coords)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        coord = self.coords[idx]
        
        # Crop directly from OpenSlide
        img = self.reader.wsi.read_region(
            (coord[0], coord[1]), 
            self.patch_level, 
            (self.patch_size, self.patch_size)
        ).convert('RGB')
        
        # Important: The transform pipeline we built automatically handles
        # converting the PIL image to a tensor and normalizing it.
        img_tensor = self.transforms(img)
        
        return img_tensor, coord
