"""
utils/transforms.py

Preprocessing transform registry for the WSI feature extraction pipeline.

Design notes
------------
* All transforms are EAGERLY buildable (no lazy imports in the registry itself).
* Stain-normalisation transforms (reinhard, macenko) use the pattern
      ToTensor  →  Lambda(x * 255)  →  NormClass()
  so the normaliser ALWAYS receives a float32 tensor in the [0, 255] range,
  matching what torchstain's PyTorch backend expects.
* Classes defined at module scope so PyTorch DataLoader can pickle them on Windows.
"""

import torch
from torchvision import transforms


# ---------------------------------------------------------------------------
# Module-level normaliser classes (must be picklable → no local class defs)
# ---------------------------------------------------------------------------

class ReinhardNorm:
    """Reinhard colour normalisation (torchstain PyTorch backend)."""

    def __init__(self):
        try:
            import torchstain
        except ImportError:
            raise ImportError(
                "Please install torchstain:  pip install torchstain")
        self.normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
        # Standard target statistics used for H&E normalisation
        self.normalizer.target_means = torch.tensor([79.2929, 11.2809, -5.9533])
        self.normalizer.target_stds  = torch.tensor([17.3957,  8.6891, 10.5019])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image arrives as float32 (C, H, W) in range [0, 255].
        # normalizer.normalize() returns (H, W, C) → permute back to (C, H, W) / 255
        norm = self.normalizer.normalize(I=image)
        return norm.permute(2, 0, 1) / 255.0


class MacenkoNorm:
    """Macenko stain normalisation (torchstain PyTorch backend)."""

    def __init__(self):
        try:
            import torchstain
        except ImportError:
            raise ImportError(
                "Please install torchstain:  pip install torchstain")
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image arrives as float32 (C, H, W) in range [0, 255].
        try:
            norm, _, _ = self.normalizer.normalize(I=image, stains=False)
            return norm.permute(2, 0, 1) / 255.0
        except Exception:
            # Rare degenerate tissue patches — pass through normalised to [0,1]
            return image / 255.0


# ---------------------------------------------------------------------------
# Named transform registry
# ---------------------------------------------------------------------------

# All transforms except reinhard/macenko are pure torchvision — no extra deps
_TRANSFORM_REGISTRY = {

    # ── Standard ImageNet baseline ────────────────────────────────────────
    'none': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),

    # ── Pathology foundation model defaults ───────────────────────────────
    'uni_default': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),

    'gigapath_default': transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),

    'gpfm_default': transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),

    'hibou_default': transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.7068, 0.5755, 0.7220),
                             std=(0.1950, 0.2316, 0.1816))]),

    'kaiko_default': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))]),

    'optimus_default': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.707223, 0.578729, 0.703617),
                             std=(0.211883, 0.230117, 0.177517))]),

    'resnet50lunit_default': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.70322989, 0.53606487, 0.66096631),
                             std=(0.21716536, 0.26081574, 0.20723464))]),

    'vitslunit_default': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.70322989, 0.53606487, 0.66096631),
                             std=(0.21716536, 0.26081574, 0.20723464))]),

    'histo_resnet18': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))]),

    'histo_resnet18_224': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))]),

    # ── Augmentation presets ──────────────────────────────────────────────
    'colourjitter': transforms.Compose([
        transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
        transforms.ToTensor()]),

    'colourjitternorm': transforms.Compose([
        transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),

    'spatial': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=0.1),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),

    'all': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.1),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))]),
}

# Canonical (auto) transform per model type
_AUTO_TRANSFORM = {
    'resnet18':           'none',
    'resnet50':           'none',
    'resnet50lunit':      'resnet50lunit_default',
    'vitslunit':          'vitslunit_default',
    'uni':                'uni_default',
    'vit_l':              'uni_default',
    'ctranspath':         'uni_default',
    'provgigapath':       'gigapath_default',
    'phikon':             'uni_default',
    'hibou_b':            'hibou_default',
    'hibou_l':            'hibou_default',
    'kaiko_b8':           'kaiko_default',
    'optimus':            'optimus_default',
    'virchow':            'gigapath_default',
    'virchow2cls':        'gigapath_default',
    'gpfm':               'gpfm_default',
    # legacy short names used by config.yaml
    'rn18':               'none',
    'rn50':               'none',
    'rn18_histo':         'histo_resnet18',
    'rn50_histo':         'histo_resnet18',
}


def build_transform(name: str) -> transforms.Compose:
    """Return a single named transform from the registry."""
    name = name.lower().strip()

    if name in _TRANSFORM_REGISTRY:
        return _TRANSFORM_REGISTRY[name]

    # Stain normalisation — built lazily so torchstain not required at import
    if name == 'reinhard':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            ReinhardNorm(),
        ])

    if name == 'macenko':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            MacenkoNorm(),
        ])

    # Fallback — imagenet is an alias for 'none'
    if name == 'imagenet':
        return _TRANSFORM_REGISTRY['none']

    raise ValueError(
        f"Unknown transform '{name}'. Available: "
        + ", ".join(sorted(_TRANSFORM_REGISTRY.keys()))
        + ", reinhard, macenko, imagenet"
    )


def get_transforms(transforms_cfg, model_type: str):
    """
    Build the full preprocessing pipeline from config.

    Parameters
    ----------
    transforms_cfg : str | list[str] | None
        * None / 'auto'     → canonical pipeline for model_type
        * single string     → named preset from the registry
        * list of strings   → first item used as named preset (legacy style)
    model_type : str
        Model key, used only when transforms_cfg is 'auto'.
    """
    key = model_type.lower().strip()

    if transforms_cfg in (None, 'auto'):
        name = _AUTO_TRANSFORM.get(key, 'none')
        return build_transform(name)

    if isinstance(transforms_cfg, list):
        # List style: treat first element as the named transform
        # e.g. transforms: [reinhard, imagenet]  →  use 'reinhard' preset
        transforms_cfg = transforms_cfg[0]

    return build_transform(transforms_cfg)
