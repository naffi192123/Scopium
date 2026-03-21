"""
datasets/slide_dataset.py

PyTorch Dataset that streams patches out of a WSI on-the-fly.

Key design decisions
--------------------
* The dataset stores the **slide path** (string), NOT an open OpenSlide handle.
  Each DataLoader worker opens its own handle inside worker_init_fn, stored in
  a process-local dict keyed by worker PID.  This allows num_workers > 0 on
  both Linux and Windows without any pickling of ctypes objects.

* worker_init_fn  is exported so callers can pass it directly to DataLoader.
  It receives the worker integer id from PyTorch automatically.

* For single-threaded use (num_workers=0) the handle is opened lazily on the
  first __getitem__ call and stored on the dataset instance.

Backward compatibility
----------------------
The old API accepted an open ``wsi`` handle as the second argument.  To keep
old code working, the constructor checks whether the second argument is a
string (new) or an OpenSlide handle (old/legacy).  In legacy mode num_workers
must remain 0.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

# Thread-local storage for per-worker slide handles
_WORKER_HANDLES: dict = {}


def worker_init_fn(worker_id: int):
    """
    Called once per DataLoader worker at startup.

    Reads 'slide_path' and 'patch_level'/'patch_size' from the worker's copy
    of the dataset and opens an OpenSlide handle local to that worker process.
    """
    import openslide
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    pid = os.getpid()
    if hasattr(dataset, 'slide_path') and dataset.slide_path is not None:
        _WORKER_HANDLES[pid] = openslide.open_slide(dataset.slide_path)


class WSIPatchDataset(Dataset):
    """
    Streams patches from a WSI using coordinates stored in an HDF5 file.

    Parameters
    ----------
    h5_path    : str             Path to the HDF5 file containing 'coords'.
    wsi_source : str | OpenSlide Slide path (preferred) or an already-opened
                                 OpenSlide object (legacy, forces num_workers=0).
    transform  : callable        Preprocessing pipeline to apply to each PIL image.
    """

    def __init__(self, h5_path: str, wsi_source, transform):
        import openslide as _osl

        self.transform = transform

        with h5py.File(h5_path, 'r') as f:
            self.coords      = f['coords'][:]
            self.patch_level = int(f['coords'].attrs['patch_level'])
            self.patch_size  = int(f['coords'].attrs['patch_size'])

        # New API: store path, open lazily per worker
        if isinstance(wsi_source, str):
            self.slide_path = wsi_source
            self._wsi       = None          # opened lazily or by worker_init_fn
            self._legacy    = False
        else:
            # Legacy: caller passed an open handle — only safe with num_workers=0
            self.slide_path = None
            self._wsi       = wsi_source
            self._legacy    = True

    def _get_handle(self):
        """Return the correct slide handle for the current thread/process."""
        if self._legacy:
            return self._wsi

        pid = os.getpid()
        if pid in _WORKER_HANDLES:
            return _WORKER_HANDLES[pid]

        # Main process / num_workers=0: open lazily and cache on instance
        if self._wsi is None:
            import openslide
            self._wsi = openslide.open_slide(self.slide_path)
        return self._wsi

    def close(self):
        """Close the main-process handle (no-op in worker processes)."""
        if self._wsi is not None:
            try:
                self._wsi.close()
            except Exception:
                pass
            self._wsi = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        wsi   = self._get_handle()
        img   = wsi.read_region(
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
