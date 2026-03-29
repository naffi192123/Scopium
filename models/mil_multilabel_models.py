"""
models/mil_multilabel_models.py

Multi-Label MIL Model Library.

Wraps all existing MIL backbones with a sigmoid (instead of softmax) output
head, enabling multi-label classification (each label is independent binary).

All models share the same forward signature:
    forward(h) -> (logits, probs, preds, attention, extras)

where:
    h         : (N, D)  float tensor — bag of patch embeddings
    logits    : (1, n_labels)  — raw score per label
    probs     : (1, n_labels)  — sigmoid(logits), independent per-label prob
    preds     : (1, n_labels)  — bool (probs > threshold)
    attention : (N,) per-patch attention weights (None for pool models)
    extras    : dict (e.g. instance_loss for CLAM)

Supported models
----------------
mean_pool  : Mean Pooling -> FC -> sigmoid
max_pool   : Max Pooling  -> FC -> sigmoid
abmil      : Gated Attention MIL  (Ilse 2018)
clam_sb    : CLAM Single-Branch   (Lu 2021)
clam_mb    : CLAM Multi-Branch    (Lu 2021)
transmil   : Transformer MIL      (Shao 2021)
dsmil      : Dual-Stream MIL      (Li 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List

# Reuse weight init from single-label models
from models.mil_models import (
    initialize_weights,
    _GatedAttn,
    _AttnNetGated,
    _NystromAttention,
    _MODEL_REGISTRY as _SL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helper: sigmoid multi-label head
# ---------------------------------------------------------------------------

def _sigmoid_preds(logits: torch.Tensor, threshold: float = 0.5):
    """logits (1, n_labels) -> probs, preds."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    return probs, preds


# ---------------------------------------------------------------------------
# 1. Mean Pooling Multi-Label
# ---------------------------------------------------------------------------

class MLMeanPoolMIL(nn.Module):
    """Global average pool -> FC -> sigmoid for multi-label."""

    def __init__(self, encoding_size=1024, n_labels=8, dropout=0.25,
                 threshold=0.5, **kw):
        super().__init__()
        self.threshold = threshold
        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, 512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_labels))
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h = h.mean(dim=0, keepdim=True)              # (1, D)
        logits = self.classifier(h)                   # (1, n_labels)
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, None, {}


# ---------------------------------------------------------------------------
# 2. Max Pooling Multi-Label
# ---------------------------------------------------------------------------

class MLMaxPoolMIL(nn.Module):
    """Global max pool -> FC -> sigmoid for multi-label."""

    def __init__(self, encoding_size=1024, n_labels=8, dropout=0.25,
                 threshold=0.5, **kw):
        super().__init__()
        self.threshold = threshold
        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, 512), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_labels))
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h = h.max(dim=0, keepdim=True).values         # (1, D)
        logits = self.classifier(h)
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, None, {}


# ---------------------------------------------------------------------------
# 3. ABMIL Multi-Label
# ---------------------------------------------------------------------------

class MLABMIL(nn.Module):
    """Gated Attention MIL — multi-label sigmoid output."""

    def __init__(self, encoding_size=1024, n_labels=8,
                 hidden_dim=256, dropout=0.25, threshold=0.5,
                 feature_proj_dim=512, **kw):
        super().__init__()
        self.threshold  = threshold
        proj_dim        = feature_proj_dim or 512
        self.projection = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(),
            nn.Dropout(dropout))
        self.attn       = _GatedAttn(proj_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(proj_dim, n_labels)
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h     = self.projection(h)                     # (N, proj_dim)
        A     = self.attn(h)                           # (N, 1)
        A_soft = F.softmax(A, dim=0)                  # (N, 1)
        if attention_only:
            return A_soft.squeeze()
        M      = (A_soft * h).sum(dim=0, keepdim=True) # (1, proj_dim)
        logits = self.classifier(M)                    # (1, n_labels)
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, A_soft.squeeze(), {}


# ---------------------------------------------------------------------------
# 4. CLAM-SB Multi-Label
# ---------------------------------------------------------------------------

class MLCLAM_SB(nn.Module):
    """CLAM Single-Branch — multi-label sigmoid output."""

    def __init__(self, encoding_size=1024, n_labels=8,
                 hidden_dim=256, dropout=0.25, k_sample=8,
                 threshold=0.5, feature_proj_dim=512, **kw):
        super().__init__()
        self.n_labels  = n_labels
        self.k_sample  = k_sample
        self.threshold = threshold
        proj_dim       = feature_proj_dim or 512

        self.proj  = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn  = _AttnNetGated(proj_dim, hidden_dim, dropout, n_classes=1)
        self.clf   = nn.Linear(proj_dim, n_labels)
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, attention_only=False):
        h      = self.proj(h)
        A, h   = self.attn(h)                          # (N,1), (N, proj_dim)
        A      = A.squeeze(1)                          # (N,)
        if attention_only:
            return F.softmax(A, dim=0)
        A_soft = F.softmax(A, dim=0)
        M      = (A_soft.unsqueeze(1) * h).sum(0, keepdim=True)  # (1, proj_dim)
        logits = self.clf(M)                           # (1, n_labels)
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, A_soft, {}


# ---------------------------------------------------------------------------
# 5. CLAM-MB Multi-Label
# ---------------------------------------------------------------------------

class MLCLAM_MB(nn.Module):
    """CLAM Multi-Branch — multi-label sigmoid output.
    Uses one attention branch, shared across all labels.
    """

    def __init__(self, encoding_size=1024, n_labels=8,
                 hidden_dim=256, dropout=0.25, k_sample=8,
                 threshold=0.5, feature_proj_dim=512, **kw):
        super().__init__()
        self.n_labels  = n_labels
        self.threshold = threshold
        proj_dim       = feature_proj_dim or 512

        self.proj  = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        # single shared attention branch (branch per class is ill-defined for ML)
        self.attn  = _AttnNetGated(proj_dim, hidden_dim, dropout, n_classes=1)
        self.clf   = nn.Linear(proj_dim, n_labels)
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, attention_only=False):
        h      = self.proj(h)
        A, h   = self.attn(h)                          # (N,1), (N, proj_dim)
        A      = A.squeeze(1)
        if attention_only:
            return F.softmax(A, dim=0)
        A_soft = F.softmax(A, dim=0)
        M      = (A_soft.unsqueeze(1) * h).sum(0, keepdim=True)
        logits = self.clf(M)
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, A_soft, {}


# ---------------------------------------------------------------------------
# 6. TransMIL Multi-Label
# ---------------------------------------------------------------------------

class MLTransMIL(nn.Module):
    """Transformer-based MIL — multi-label sigmoid output."""

    def __init__(self, encoding_size=1024, n_labels=8,
                 hidden_dim=512, dropout=0.25, threshold=0.5, **kw):
        super().__init__()
        self.threshold  = threshold
        self.proj       = nn.Sequential(nn.Linear(encoding_size, hidden_dim), nn.ReLU())
        self.pos_enc    = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.layer1     = _NystromAttention(hidden_dim, heads=8, dropout=dropout)
        self.layer2     = _NystromAttention(hidden_dim, heads=8, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_labels))
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h = self.proj(h).unsqueeze(0)                  # (1, N, hidden_dim)
        h = h + self.pos_enc
        h = self.layer1(h)
        h = self.layer2(h)
        z = h.squeeze(0).mean(dim=0, keepdim=True)     # (1, hidden_dim)
        logits = self.classifier(z)                    # (1, n_labels)
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, None, {}


# ---------------------------------------------------------------------------
# 7. DSMIL Multi-Label
# ---------------------------------------------------------------------------

class MLDSMIL(nn.Module):
    """Dual-Stream MIL — multi-label sigmoid output."""

    def __init__(self, encoding_size=1024, n_labels=8,
                 hidden_dim=256, dropout=0.25, threshold=0.5,
                 feature_proj_dim=512, **kw):
        super().__init__()
        self.threshold = threshold
        proj_dim       = feature_proj_dim or 512

        self.proj     = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        self.inst_clf = nn.Linear(proj_dim, n_labels)
        self.q        = nn.Linear(proj_dim, hidden_dim)
        self.k        = nn.Linear(proj_dim, hidden_dim)
        self.bag_clf  = nn.Linear(proj_dim, n_labels)
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h           = self.proj(h)                     # (N, proj_dim)
        inst_logits = self.inst_clf(h)                 # (N, n_labels)
        # critical instance: highest mean sigmoid across labels
        crit_idx    = torch.sigmoid(inst_logits).mean(dim=1).argmax()
        crit        = h[crit_idx].unsqueeze(0)

        scores = (self.q(crit) * self.k(h)).sum(-1)   # (N,)
        A      = F.softmax(scores, dim=0)
        if attention_only:
            return A
        M = (A.unsqueeze(1) * h).sum(0, keepdim=True) # (1, proj_dim)

        bag_logits = self.bag_clf(M)                   # (1, n_labels)
        # ensemble averaged
        logits = (inst_logits[crit_idx].unsqueeze(0) + bag_logits) / 2.0
        probs, preds = _sigmoid_preds(logits, self.threshold)
        return logits, probs, preds, A, {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ML_MODEL_REGISTRY = {
    "mean_pool" : MLMeanPoolMIL,
    "max_pool"  : MLMaxPoolMIL,
    "abmil"     : MLABMIL,
    "clam_sb"   : MLCLAM_SB,
    "clam_mb"   : MLCLAM_MB,
    "transmil"  : MLTransMIL,
    "dsmil"     : MLDSMIL,
}


def build_multilabel_model(config: dict) -> Tuple[nn.Module, int]:
    """
    Build and return a multi-label MIL model from config.

    Uses:
        config.mil.model              → model key
        config.mil.encoding_size      → input feature dimension
        config.mil.hidden_dim         → attention hidden size
        config.mil.dropout            → dropout rate
        config.multilabel.label_names → output size (n_labels)
        config.multilabel.threshold   → sigmoid decision threshold
    """
    mil_cfg  = config.get("mil", {})
    ml_cfg   = config.get("multilabel", {})

    model_key      = mil_cfg.get("model", "abmil").lower().strip()
    encoding_size  = mil_cfg.get("encoding_size", 1024)
    hidden_dim     = mil_cfg.get("hidden_dim", 256)
    dropout        = mil_cfg.get("dropout", 0.25)
    k_sample       = mil_cfg.get("k_sample", 8)
    feature_proj   = mil_cfg.get("feature_proj_dim", 512)
    label_names    = ml_cfg.get("label_names", [])
    n_labels       = len(label_names)
    threshold      = float(ml_cfg.get("threshold", 0.5))

    if model_key not in _ML_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MIL model '{model_key}'. "
            f"Available: {list(_ML_MODEL_REGISTRY.keys())}")

    if n_labels < 1:
        raise ValueError(
            "multilabel.label_names must contain at least one label name.")

    model = _ML_MODEL_REGISTRY[model_key](
        encoding_size   = encoding_size,
        n_labels        = n_labels,
        hidden_dim      = hidden_dim,
        dropout         = dropout,
        k_sample        = k_sample,
        threshold       = threshold,
        feature_proj_dim= feature_proj,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_params


def ml_has_attention(model: nn.Module) -> bool:
    """True if the model produces per-patch attention scores."""
    return isinstance(model, (MLABMIL, MLCLAM_SB, MLCLAM_MB, MLDSMIL))
