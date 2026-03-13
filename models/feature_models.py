"""
models/feature_models.py

Model factory for all supported feature extraction backbones.

Design
------
* ALL model-specific imports are LAZY (inside load_backbone).
  Plain `import models.feature_models` never fails even if timm/transformers
  are absent.
* Returns (model, feat_dim, input_size) so callers know the embedding
  dimensionality without hard-coding it.
"""

import os
import sys
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_backbone(model_type: str, weights_path: str = None, device=None):
    """
    Load a backbone model.

    Parameters
    ----------
    model_type   : str   Key from the supported list below.
    weights_path : str   Path to local weights file (required for some models).
    device       : torch.device  Defaults to CUDA if available.

    Returns
    -------
    model      : nn.Module in eval mode, on device, DataParallel if >1 GPU
    feat_dim   : int   Output embedding dimensionality
    input_size : int   Square patch dimension the model expects
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = model_type.lower().strip()
    logger.info(f"Loading model: {model_type}")

    # ── Standard ResNets (torchvision) ────────────────────────────────────────
    if model_type in ('rn18', 'resnet18'):
        import torchvision.models as tv
        model = tv.resnet18(pretrained=True)
        model.fc = nn.Identity()
        feat_dim, input_size = 512, 224

    elif model_type in ('rn18_histo', 'resnet18_histo'):
        # ResNet-18 with pretrained=False and normalised differently
        import torchvision.models as tv
        model = tv.resnet18(pretrained=False)
        model.fc = nn.Identity()
        feat_dim, input_size = 512, 224

    elif model_type in ('rn50', 'resnet50'):
        import torchvision.models as tv
        model = tv.resnet50(pretrained=True)
        model.fc = nn.Identity()
        feat_dim, input_size = 2048, 224

    elif model_type in ('rn50_histo', 'resnet50_histo'):
        import torchvision.models as tv
        model = tv.resnet50(pretrained=False)
        model.fc = nn.Identity()
        feat_dim, input_size = 1024, 224

    # ── ViT-L (ImageNet, via timm) ────────────────────────────────────────────
    elif model_type == 'vit_l':
        import timm
        model = timm.create_model('vit_large_patch16_224',
                                  num_classes=0, pretrained=True)
        feat_dim, input_size = 1024, 224

    # ── UNI ───────────────────────────────────────────────────────────────────
    elif model_type == 'uni':
        import timm
        model = timm.create_model(
            'vit_large_patch16_224', img_size=224, patch_size=16,
            init_values=1e-5, num_classes=0, dynamic_img_size=True)
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(
                torch.load(weights_path, map_location='cpu'), strict=True)
            logger.info(f"UNI: loaded weights from {weights_path}")
        else:
            logger.warning(
                "UNI: weights_path not provided or missing. "
                "Trying HF hub (MahmoodLab/UNI)…")
            model = timm.create_model(
                'hf-hub:MahmoodLab/UNI', pretrained=True,
                init_values=1e-5, dynamic_img_size=True)
        feat_dim, input_size = 1024, 224

    # ── Prov-GigaPath ─────────────────────────────────────────────────────────
    elif model_type == 'provgigapath':
        import timm
        logger.info("Loading prov-gigapath (requires HF token)…")
        model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath',
                                  pretrained=True)
        feat_dim, input_size = 1536, 224

    # ── Phikon ────────────────────────────────────────────────────────────────
    elif model_type == 'phikon':
        from transformers import ViTModel
        model = ViTModel.from_pretrained('owkin/phikon', add_pooling_layer=False)
        feat_dim, input_size = 768, 224

    # ── Hibou-B ───────────────────────────────────────────────────────────────
    elif model_type == 'hibou_b':
        from transformers import AutoModel
        logger.info("Loading hibou-b from HuggingFace hub (histai/hibou-b)…")
        model = AutoModel.from_pretrained('histai/hibou-b', trust_remote_code=True)
        feat_dim, input_size = 768, 224

    # ── Hibou-L ───────────────────────────────────────────────────────────────
    elif model_type == 'hibou_l':
        from transformers import AutoModel
        logger.info("Loading hibou-l from HuggingFace hub (histai/hibou-l)…")
        model = AutoModel.from_pretrained('histai/hibou-l', trust_remote_code=True)
        feat_dim, input_size = 1024, 224

    # ── H-Optimus-0 ───────────────────────────────────────────────────────────
    elif model_type == 'optimus':
        import timm
        model = timm.create_model('hf_hub:bioptimus/H-optimus-0', pretrained=True)
        feat_dim, input_size = 1536, 224

    # ── Virchow ───────────────────────────────────────────────────────────────
    elif model_type == 'virchow':
        import timm
        model = timm.create_model(
            'hf-hub:paige-ai/Virchow', pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        feat_dim, input_size = 2560, 224   # CLS + mean-patch concat

    elif model_type == 'virchow2cls':
        import timm
        model = timm.create_model(
            'hf-hub:paige-ai/Virchow-2', pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
        feat_dim, input_size = 1280, 224   # CLS token only

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: "
            "rn18, rn50, rn18_histo, rn50_histo, vit_l, uni, provgigapath, "
            "phikon, hibou_b, hibou_l, optimus, virchow, virchow2cls"
        )

    model = model.to(device)

    # Multi-GPU via DataParallel (mirrors reference behaviour)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"DataParallel across {torch.cuda.device_count()} GPUs.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb  = sum(
        p.nelement() * p.element_size()
        for p in list(model.parameters()) + list(model.buffers())
    ) / (1024 ** 2)
    logger.info(
        f"Model '{model_type}' | Params: {n_params:,} | "
        f"Size: {size_mb:.2f} MB | feat_dim: {feat_dim}")

    model.eval()
    return model, feat_dim, input_size


def pool_features(features, model_type: str):
    """
    Apply model-specific pooling / head selection to the raw model output.

    Parameters
    ----------
    features   : raw output from model(batch)
    model_type : str

    Returns
    -------
    torch.Tensor  shape (N, feat_dim)
    """
    model_type = model_type.lower().strip()

    if model_type == 'phikon':
        return features.last_hidden_state[:, 0, :]

    if model_type == 'virchow':
        cls   = features[:, 0]
        patch = features[:, 1:]
        return torch.cat([cls, patch.mean(1)], dim=-1)

    if model_type == 'virchow2cls':
        return features[:, 0]

    if model_type in ('hibou_l', 'hibou_b'):
        return features.pooler_output

    # Default: model already returns a flat (N, D) feature tensor
    return features
