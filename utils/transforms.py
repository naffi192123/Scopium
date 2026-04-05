"""
utils/transforms.py

Preprocessing transform registry and cascade builder for WSI feature extraction.

Design notes
------------
* Every named preset is stored as a pair (pre_ops, post_ops):
    pre_ops  — torchvision transforms that operate on a PIL Image (before ToTensor)
    post_ops — torchvision transforms that operate on a float Tensor (after ToTensor)

  This split enables the **cascade builder** (build_cascade) to compose
  multiple presets correctly:

    - The FIRST step in the cascade contributes:   pre_ops + [ToTensor] + post_ops
    - Every SUBSEQUENT step contributes:            post_ops only
      (the tensor is already in [0, 1] float32 — no second ToTensor needed)

  Stain normalisers (reinhard, macenko) have empty pre_ops because they
  operate on the tensor domain. Their full logic lives in post_ops:
      [Lambda(x*255), NormClass()]   →  receives a tensor in [0,1] and returns [0,1]

* A single string in config selects the full transform (pre+ToTensor+post combined),
  which is exactly the behaviour that existed before.  Full backward compatibility.

* Classes are defined at module scope so PyTorch DataLoader can pickle them on Windows.

Usage in config.yaml
--------------------
  # Single preset (original behaviour)
  transforms: auto
  transforms: optimus_default
  transforms: reinhard

  # Sequential cascade (new)
  transforms:
    - reinhard            # Step 1: colour normalisation  (PIL → normalised Tensor)
    - optimus_default     # Step 2: model-specific crop + pixel normalise

  # Common cascade recipes — see config.yaml for annotated examples.
"""

import torch
from torchvision import transforms


# ---------------------------------------------------------------------------
# Module-level normaliser classes  (picklable — no local class defs)
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
        # Standard H&E target statistics (Reinhard 2001 / Macenko target)
        self.normalizer.target_means = torch.tensor([79.2929, 11.2809, -5.9533])
        self.normalizer.target_stds  = torch.tensor([17.3957,  8.6891, 10.5019])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Receives float32 (C, H, W) in [0, 255] (pre-scaled by Lambda).
        # normalizer.normalize() returns (H, W, C) → permute + scale to [0, 1].
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
        try:
            norm, _, _ = self.normalizer.normalize(I=image, stains=False)
            return norm.permute(2, 0, 1) / 255.0
        except Exception:
            # Degenerate tissue patches — pass through as [0, 1]
            return image / 255.0


# ---------------------------------------------------------------------------
# Split-step registry:  name → (pre_ops, post_ops)
#
#   pre_ops  : list of torchvision transforms that run on a PIL Image
#   post_ops : list of torchvision transforms that run on a float Tensor
#
# The full single-preset transform is:
#   Compose(pre_ops + [ToTensor()] + post_ops)
#
# In a cascade, only the FIRST step includes ToTensor(); subsequent steps
# receive the tensor produced by the previous step and apply only post_ops.
# ---------------------------------------------------------------------------

_IMAGENET_NORM = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225))

_HIBOU_NORM = transforms.Normalize(
    mean=(0.7068, 0.5755, 0.7220),
    std=(0.1950, 0.2316, 0.1816))

_KAIKO_NORM = transforms.Normalize(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5))

_OPTIMUS_NORM = transforms.Normalize(
    mean=(0.707223, 0.578729, 0.703617),
    std=(0.211883, 0.230117, 0.177517))

_LUNIT_NORM = transforms.Normalize(
    mean=(0.70322989, 0.53606487, 0.66096631),
    std=(0.21716536, 0.26081574, 0.20723464))

_HALF_NORM = transforms.Normalize(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5))


# Each entry: (pre_ops_on_PIL, post_ops_on_Tensor)
# Stain normaliser steps that require torchstain are marked with a sentinel string
# and resolved lazily inside _get_step() so importing this module NEVER fails if
# torchstain is absent.
_STAIN_SENTINEL = '__stain__'

_STEP_REGISTRY: dict = {

    # ── Standard ImageNet ─────────────────────────────────────────────────
    'none':       ([], [_IMAGENET_NORM]),
    'imagenet':   ([], [_IMAGENET_NORM]),

    # ── Stain normalisers (lazy — resolved in _get_step to avoid import errors) ─
    # Sentinels: the actual classes are built only when the step is first used.
    'reinhard': _STAIN_SENTINEL,
    'macenko':  _STAIN_SENTINEL,

    # ── Pathology foundation model defaults ───────────────────────────────
    'uni_default': (
        [transforms.Resize(224)],
        [_IMAGENET_NORM]),

    'gigapath_default': (
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(224)],
        [_IMAGENET_NORM]),

    'gpfm_default': (
        [transforms.Resize((224, 224), interpolation=3)],
        [_IMAGENET_NORM]),

    'hibou_default': (
        [transforms.Resize((224, 224),
                           interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop((224, 224))],
        [_HIBOU_NORM]),

    'kaiko_default': (
        [transforms.Resize(224),
         transforms.CenterCrop(224)],
        [_KAIKO_NORM]),

    'optimus_default': (
        [transforms.Resize(224)],
        [_OPTIMUS_NORM]),

    'resnet50lunit_default': ([], [_LUNIT_NORM]),

    'vitslunit_default': (
        [transforms.Resize(224)],
        [_LUNIT_NORM]),

    'histo_resnet18': ([], [_HALF_NORM]),

    'histo_resnet18_224': (
        [transforms.Resize(224)],
        [_HALF_NORM]),

    # ── Size-only steps (useful as cascade building blocks) ───────────────
    'resize_224': (
        [transforms.Resize(224)],
        []),

    'resize_256_crop_224': (
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(224)],
        []),

    'centercrop_224': (
        [transforms.CenterCrop(224)],
        []),

    # ── Normalisation-only steps (on tensor, no spatial ops) ──────────────
    'imagenet_norm':   ([], [_IMAGENET_NORM]),
    'hibou_norm':      ([], [_HIBOU_NORM]),
    'kaiko_norm':      ([], [_KAIKO_NORM]),
    'optimus_norm':    ([], [_OPTIMUS_NORM]),
    'lunit_norm':      ([], [_LUNIT_NORM]),
    'half_norm':       ([], [_HALF_NORM]),   # mean/std = 0.5

    # ── Augmentation presets ──────────────────────────────────────────────
    'colourjitter': (
        [transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)],
        []),

    'colourjitternorm': (
        [transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)],
        [_IMAGENET_NORM]),

    'spatial': ([], [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=90, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=0.1),
        _IMAGENET_NORM]),

    'all': (
        [transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                saturation=0.1, hue=0.1)],
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomAffine(degrees=90, translate=(0.1, 0.1),
                                 scale=(0.9, 1.1), shear=0.1),
         _IMAGENET_NORM]),
}


# Canonical (auto) transform per model type
_AUTO_TRANSFORM: dict = {
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
    # legacy short names
    'rn18':               'none',
    'rn50':               'none',
    'rn18_histo':         'histo_resnet18',
    'rn50_histo':         'histo_resnet18',
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_step(name: str) -> tuple:
    """Return (pre_ops, post_ops) for a named step. Raises ValueError if unknown."""
    key = name.lower().strip()
    if key not in _STEP_REGISTRY:
        available = ", ".join(sorted(_STEP_REGISTRY.keys()))
        raise ValueError(
            f"Unknown transform step '{name}'. Available steps:\n  {available}")

    entry = _STEP_REGISTRY[key]

    # Lazy resolution for stain normalisers (require torchstain — not installed everywhere)
    if entry is _STAIN_SENTINEL:
        if key == 'reinhard':
            return ([], [
                transforms.Lambda(lambda x: x * 255.0),
                ReinhardNorm(),        # raises ImportError if torchstain absent
            ])
        if key == 'macenko':
            return ([], [
                transforms.Lambda(lambda x: x * 255.0),
                MacenkoNorm(),
            ])

    return entry



def _build_single(name: str) -> transforms.Compose:
    """Build a full standalone transform (pre + ToTensor + post) for a single name."""
    pre_ops, post_ops = _get_step(name)
    return transforms.Compose(pre_ops + [transforms.ToTensor()] + post_ops)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cascade(steps: list) -> transforms.Compose:
    """
    Build a sequential transform cascade from an ordered list of step names.

    The cascade rules are:
      - The FIRST step contributes:  pre_ops + [ToTensor()] + post_ops
      - Every subsequent step contributes: post_ops only
        (the tensor is already in [0, 1] float32 from the previous step)

    This means you can write, for example::

        transforms:
          - reinhard           # PIL → colour-normalised [0,1] Tensor
          - optimus_default    # crop (skipped — already tensor) + channel normalise

    The composed pipeline will be:
        [ToTensor, Lambda(*255), Reinhard, Normalize(optimus stats)]

    Note that spatial pre_ops (Resize, CenterCrop) from steps 2+ are ALSO applied
    because they now receive a float Tensor.  The torchvision transforms that do
    spatial ops (Resize, CenterCrop, etc.) accept both PIL and Tensor inputs in
    modern torchvision, so this works correctly.

    Parameters
    ----------
    steps : list[str]
        Ordered list of named presets.  Must contain at least one entry.

    Returns
    -------
    transforms.Compose
    """
    if not steps:
        raise ValueError("build_cascade received an empty steps list.")

    all_ops = []

    for i, name in enumerate(steps):
        pre_ops, post_ops = _get_step(name)

        if i == 0:
            # First step: PIL → Tensor conversion happens here
            all_ops.extend(pre_ops)
            all_ops.append(transforms.ToTensor())
            all_ops.extend(post_ops)
        else:
            # Subsequent steps: already on a float32 Tensor
            # Include pre_ops too — torchvision spatial transforms accept Tensor
            all_ops.extend(pre_ops)
            all_ops.extend(post_ops)

    return transforms.Compose(all_ops)


def list_transforms() -> list:
    """Return all registered transform step names (sorted)."""
    return sorted(_STEP_REGISTRY.keys())


def build_transform(name: str) -> transforms.Compose:
    """Return a full standalone transform for a single named preset.

    This is the original single-step API, retained for backward compatibility.
    """
    return _build_single(name)


def get_transforms(transforms_cfg, model_type: str) -> transforms.Compose:
    """
    Build the full preprocessing pipeline from config.

    Parameters
    ----------
    transforms_cfg : None | str | list[str]
        * None / 'auto'     → canonical pipeline for model_type
        * single string     → named preset (full single-step transform)
        * list of strings   → sequential CASCADE via build_cascade()
    model_type : str
        Model key, used only when transforms_cfg is 'auto' / None.

    Returns
    -------
    transforms.Compose
    """
    key = model_type.lower().strip()

    # Auto / None: use canonical preset for this model
    if transforms_cfg in (None, 'auto'):
        auto_name = _AUTO_TRANSFORM.get(key, 'none')
        return _build_single(auto_name)

    # List → cascade
    if isinstance(transforms_cfg, list):
        if len(transforms_cfg) == 0:
            raise ValueError("transforms list in config must not be empty.")
        if len(transforms_cfg) == 1:
            # Single-element list: same as a single string
            return _build_single(str(transforms_cfg[0]).lower().strip())
        return build_cascade([str(s).lower().strip() for s in transforms_cfg])

    # Single string
    if isinstance(transforms_cfg, str):
        return _build_single(transforms_cfg.lower().strip())

    raise TypeError(
        f"transforms config must be None, str, or list[str]; got {type(transforms_cfg)}")
