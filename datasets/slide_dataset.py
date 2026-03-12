"""
datasets/slide_dataset.py

PyTorch Dataset that streams patches out of a WSI on-the-fly.

Key design decisions
--------------------
* The openslide.OpenSlide handle is passed IN to the constructor (not opened
  inside it). This means the dataset is created AFTER the slide is opened in
  the MAIN process, and the handle is then used from a single thread.
* For Windows compatibility (ctypes cannot cross process boundaries) set
  num_workers=0 in the DataLoader so patches are read in the main thread.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class WSIPatchDataset(Dataset):
    """
    Streams patches from a WSI using coordinates stored in an HDF5 file.

    Parameters
    ----------
    h5_path   : str             Path to the HDF5 file containing 'coords'.
    wsi       : openslide.OpenSlide  Already-opened slide object.
    transform : callable        Preprocessing pipeline to apply to each PIL image.
    """

    def __init__(self, h5_path: str, wsi, transform):
        self.wsi       = wsi
        self.transform = transform

        with h5py.File(h5_path, 'r') as f:
            self.coords      = f['coords'][:]
            self.patch_level = int(f['coords'].attrs['patch_level'])
            self.patch_size  = int(f['coords'].attrs['patch_size'])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img   = self.wsi.read_region(
            (int(coord[0]), int(coord[1])),
            self.patch_level,
            (self.patch_size, self.patch_size)
        ).convert('RGB')
        return self.transform(img), coord


def collate_features(batch):
    """Custom collate that stacks tensors and keeps coords as numpy."""
    imgs   = torch.stack([item[0] for item in batch])
    coords = np.stack([item[1] for item in batch])
    return imgs, coords
