"""
models/mil_models.py

Modular MIL model library.

All models share the same forward signature:
    forward(h) -> (logits, Y_prob, Y_hat, attention, extras)

where:
    h          : (N, D) float tensor — bag of patch embeddings
    logits     : (1, n_classes)
    Y_prob     : (1, n_classes) softmax probabilities
    Y_hat      : (1,) predicted class index
    attention  : (N,) per-patch attention weights (None for pool models)
    extras     : dict (e.g. instance_loss for CLAM)

Design note — auto-detected projection dimension
-------------------------------------------------
  encoding_size : the ACTUAL dimension of the .pt feature vectors on disk.
                  Must match the feature extractor. NEVER changed by HPO.

  proj_dim      : the REQUESTED internal projection dimension. HPO can tune
                  this via the 'proj_dim' search-space parameter.

  _d            : the ACTUAL output dimension of self.proj, detected by
                  probing the projection layer with a dummy tensor immediately
                  after it is built.  ALL downstream layers (attn, clf,
                  inst_clfs, bag_clf, q, k) are built from _d, never from
                  proj_dim directly.

  This design makes a dimension mismatch architecturally impossible:
  even if a stale or partially-updated version of self.proj uses a different
  hardcoded value, all subsequent layers will automatically match it.

Supported models
----------------
  mean_pool : Mean Pooling  -> linear classifier
  max_pool  : Max Pooling   -> linear classifier
  abmil     : Gated Attention-Based MIL (Ilse et al. 2018)
  clam_sb   : CLAM single-branch (Lu et al. 2021)
  clam_mb   : CLAM multi-branch
  transmil  : Transformer MIL (Shao et al. 2021)
  dsmil     : Dual-Stream MIL (Li et al. 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def initialize_weights(module):
    """Kaiming uniform for Linear layers, zero bias. Ones/zeros for LayerNorm."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def _probe_dim(projection: nn.Module, encoding_size: int) -> int:
    """Return the actual output dimension of *projection* for an input of
    shape (1, encoding_size).  This is called once at construction time so
    that all downstream layers are built from the *real* output dimension,
    not from the proj_dim argument — making dimension mismatches impossible.
    """
    with torch.no_grad():
        dummy = torch.zeros(1, encoding_size)
        out   = projection(dummy)
    return out.shape[-1]


# ---------------------------------------------------------------------------
# 1. Mean Pooling MIL
# ---------------------------------------------------------------------------
class MeanPoolMIL(nn.Module):
    """Global average pool -> hidden projection -> linear classifier."""

    def __init__(self, encoding_size=1024, n_classes=2,
                 dropout=0.25, proj_dim=512, **kw):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        _d = _probe_dim(self.proj, encoding_size)           # auto-detect actual dim
        self.classifier = nn.Linear(_d, n_classes)
        initialize_weights(self)

    def forward(self, h):
        h      = h.mean(dim=0, keepdim=True)               # (1, encoding_size)
        h      = self.proj(h)                              # (1, _d)
        logits = self.classifier(h)                        # (1, n_classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = logits.argmax(dim=1)
        return logits, Y_prob, Y_hat, None, {}


# ---------------------------------------------------------------------------
# 2. Max Pooling MIL
# ---------------------------------------------------------------------------
class MaxPoolMIL(nn.Module):
    """Global max pool -> hidden projection -> linear classifier."""

    def __init__(self, encoding_size=1024, n_classes=2,
                 dropout=0.25, proj_dim=512, **kw):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        _d = _probe_dim(self.proj, encoding_size)
        self.classifier = nn.Linear(_d, n_classes)
        initialize_weights(self)

    def forward(self, h):
        h      = h.max(dim=0, keepdim=True).values        # (1, encoding_size)
        h      = self.proj(h)                             # (1, _d)
        logits = self.classifier(h)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = logits.argmax(dim=1)
        return logits, Y_prob, Y_hat, None, {}


# ---------------------------------------------------------------------------
# 3. ABMIL — Gated Attention-Based MIL (Ilse et al. 2018)
# ---------------------------------------------------------------------------
class _GatedAttn(nn.Module):
    def __init__(self, L, D, dropout):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(L, D), nn.Tanh(),    nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(D, 1)

    def forward(self, h):
        return self.w(self.V(h) * self.U(h))               # (N, 1)


class ABMIL(nn.Module):
    """Gated Attention-Based MIL."""

    def __init__(self, encoding_size=1024, n_classes=2,
                 hidden_dim=256, dropout=0.25, proj_dim=512, **kw):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        _d = _probe_dim(self.projection, encoding_size)    # actual projection dim
        self.attn       = _GatedAttn(_d, hidden_dim, dropout)
        self.classifier = nn.Linear(_d, n_classes)
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h      = self.projection(h)                        # (N, _d)
        A      = self.attn(h)                              # (N, 1)
        A_soft = F.softmax(A, dim=0)                       # (N, 1)
        if attention_only:
            return A_soft.squeeze()
        M      = (A_soft * h).sum(dim=0, keepdim=True)    # (1, _d)
        logits = self.classifier(M)                        # (1, n_classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = logits.argmax(dim=1)
        return logits, Y_prob, Y_hat, A_soft.squeeze(), {}


# ---------------------------------------------------------------------------
# 4. CLAM-SB (Single Branch) — Lu et al. 2021
# ---------------------------------------------------------------------------
class _AttnNetGated(nn.Module):
    def __init__(self, L, D, dropout, n_classes=1):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(L, D), nn.Tanh(),    nn.Dropout(dropout))
        self.b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.c = nn.Linear(D, n_classes)

    def forward(self, x):
        A = self.c(self.a(x) * self.b(x))                 # (N, n_classes)
        return A, x                                        # passes x through unchanged


class CLAM_SB(nn.Module):
    """CLAM Single-Branch.

    Dimension flow:
        input          : (N, encoding_size)
        self.proj      : (N, _d)        <- probed after build; _d == actual output
        self.attn      : attention (N, 1) + passthrough h (N, _d)
        self.clf       : (1, n_classes)
        self.inst_clfs : (2k, 2)  per class — ALL use _d
    """

    def __init__(self, encoding_size=1024, n_classes=2,
                 hidden_dim=256, dropout=0.25, k_sample=8,
                 instance_loss_fn=None, proj_dim=512, **kw):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample  = k_sample
        self.ifn       = instance_loss_fn or nn.CrossEntropyLoss()

        # Build projection first, then probe its actual output dimension
        self.proj = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        _d = _probe_dim(self.proj, encoding_size)          # <- auto-detected dim

        # All subsequent layers use _d — never proj_dim directly
        self.attn      = _AttnNetGated(_d, hidden_dim, dropout, n_classes=1)
        self.clf       = nn.Linear(_d, n_classes)
        self.inst_clfs = nn.ModuleList(
            [nn.Linear(_d, 2) for _ in range(n_classes)])
        initialize_weights(self)

    def _inst_eval(self, A, h, clf):
        k      = min(self.k_sample, len(A) // 2)
        top_p  = h[A.topk(k, dim=0).indices.squeeze()]
        top_n  = h[(-A).topk(k, dim=0).indices.squeeze()]
        pts    = torch.cat([torch.ones (k, dtype=torch.long, device=h.device),
                            torch.zeros(k, dtype=torch.long, device=h.device)])
        inst   = torch.cat([top_p, top_n])
        loss   = self.ifn(clf(inst), pts)
        preds  = clf(inst).argmax(dim=1)
        return loss, preds.cpu().numpy(), pts.cpu().numpy()

    def forward(self, h, label=None, instance_eval=False, attention_only=False):
        h      = self.proj(h)                              # (N, _d)
        A, h   = self.attn(h)                              # A:(N,1), h:(N, _d)
        A      = A.squeeze(1)                              # (N,)
        if attention_only:
            return F.softmax(A, dim=0)
        A_soft = F.softmax(A, dim=0)
        M      = (A_soft.unsqueeze(1) * h).sum(0, keepdim=True)  # (1, _d)
        logits = self.clf(M)                               # (1, n_classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = logits.argmax(dim=1)

        extras = {}
        if instance_eval and label is not None:
            iloss, ipreds, itargs = 0., [], []
            oh = F.one_hot(label.view(-1), self.n_classes).squeeze()
            for i, clf in enumerate(self.inst_clfs):
                if oh[i].item() == 1:
                    l, p, t = self._inst_eval(A.detach(), h.detach(), clf)
                    iloss += l
                    ipreds.extend(p)
                    itargs.extend(t)
            extras = {'instance_loss': iloss,
                      'inst_preds':  np.array(ipreds),
                      'inst_labels': np.array(itargs)}

        return logits, Y_prob, Y_hat, A_soft, extras


# ---------------------------------------------------------------------------
# 5. CLAM-MB (Multi Branch)
# ---------------------------------------------------------------------------
class CLAM_MB(nn.Module):
    """CLAM Multi-Branch — one attention branch per class.

    Same auto-detected _d invariant as CLAM_SB.
    """

    def __init__(self, encoding_size=1024, n_classes=2,
                 hidden_dim=256, dropout=0.25, k_sample=8,
                 instance_loss_fn=None, proj_dim=512, **kw):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample  = k_sample
        self.ifn       = instance_loss_fn or nn.CrossEntropyLoss()

        self.proj = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        _d = _probe_dim(self.proj, encoding_size)

        self.attn      = _AttnNetGated(_d, hidden_dim, dropout, n_classes=n_classes)
        self.clfs      = nn.ModuleList([nn.Linear(_d, 1) for _ in range(n_classes)])
        self.inst_clfs = nn.ModuleList([nn.Linear(_d, 2) for _ in range(n_classes)])
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, attention_only=False):
        h      = self.proj(h)                              # (N, _d)
        A, h   = self.attn(h)                              # (N, n_classes), (N, _d)
        A      = A.transpose(0, 1)                         # (n_classes, N)
        if attention_only:
            return F.softmax(A, dim=1)
        A_soft = F.softmax(A, dim=1)
        M      = torch.mm(A_soft, h)                       # (n_classes, _d)
        logits = torch.stack(
            [self.clfs[c](M[c]).squeeze()
             for c in range(self.n_classes)]).unsqueeze(0)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = logits.argmax(dim=1)
        return logits, Y_prob, Y_hat, A_soft[Y_hat.item()], {}


# ---------------------------------------------------------------------------
# 6. TransMIL — Transformer-based MIL (Shao et al. 2021)
# ---------------------------------------------------------------------------
class _NystromAttention(nn.Module):
    """Simplified 1-layer Nystrom-style self-attention."""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class TransMIL(nn.Module):
    """Transformer-based MIL.

    Uses hidden_dim as transformer width.  proj_dim is accepted but ignored
    so HPO can include TransMIL without affecting its architecture.
    """

    def __init__(self, encoding_size=1024, n_classes=2,
                 hidden_dim=512, dropout=0.25, **kw):
        super().__init__()
        self.proj    = nn.Sequential(nn.Linear(encoding_size, hidden_dim), nn.ReLU())
        _d = _probe_dim(self.proj, encoding_size)
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, _d))
        self.layer1  = _NystromAttention(_d, heads=8, dropout=dropout)
        self.layer2  = _NystromAttention(_d, heads=8, dropout=dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(_d), nn.Linear(_d, n_classes))
        initialize_weights(self)

    def forward(self, h):
        h      = self.proj(h).unsqueeze(0)                 # (1, N, _d)
        h      = h + self.pos_enc
        h      = self.layer1(h)
        h      = self.layer2(h)
        z      = h.squeeze(0).mean(dim=0, keepdim=True)    # (1, _d)
        logits = self.classifier(z)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat  = logits.argmax(dim=1)
        return logits, Y_prob, Y_hat, None, {}


# ---------------------------------------------------------------------------
# 7. DSMIL — Dual-Stream MIL (Li et al. 2021)
# ---------------------------------------------------------------------------
class DSMIL(nn.Module):
    """Dual-stream MIL with auto-detected projection dimension."""

    def __init__(self, encoding_size=1024, n_classes=2,
                 hidden_dim=256, dropout=0.25, proj_dim=512, **kw):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoding_size, proj_dim), nn.ReLU(), nn.Dropout(dropout))
        _d = _probe_dim(self.proj, encoding_size)

        self.inst_clf = nn.Linear(_d, n_classes)
        self.q        = nn.Linear(_d, hidden_dim)
        self.k        = nn.Linear(_d, hidden_dim)
        self.bag_clf  = nn.Linear(_d, n_classes)
        initialize_weights(self)

    def forward(self, h, attention_only=False):
        h           = self.proj(h)                         # (N, _d)
        inst_logits = self.inst_clf(h)                     # (N, n_classes)
        crit_idx    = inst_logits.argmax(dim=0)[1]
        crit        = h[crit_idx].unsqueeze(0)             # (1, _d)
        scores      = (self.q(crit) * self.k(h)).sum(-1)   # (N,)
        A           = F.softmax(scores, dim=0)
        if attention_only:
            return A
        M           = (A.unsqueeze(1) * h).sum(0, keepdim=True)  # (1, _d)
        bag_logits  = self.bag_clf(M)
        logits      = (inst_logits[crit_idx].unsqueeze(0) + bag_logits) / 2.0
        Y_prob      = F.softmax(logits, dim=1)
        Y_hat       = logits.argmax(dim=1)
        return logits, Y_prob, Y_hat, A, {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    'mean_pool': MeanPoolMIL,
    'max_pool':  MaxPoolMIL,
    'abmil':     ABMIL,
    'clam_sb':   CLAM_SB,
    'clam_mb':   CLAM_MB,
    'transmil':  TransMIL,
    'dsmil':     DSMIL,
}


def build_mil_model(config: dict) -> tuple:
    """Build and return a MIL model from config.

    Parameters read from config
    ---------------------------
    config.mil.model           : model key (e.g. 'abmil')
    config.mil.encoding_size   : input feature dimension — MUST match .pt files,
                                 NEVER changed by HPO
    config.mil.hidden_dim      : attention hidden size (default 256)
    config.mil.proj_dim        : requested projection size (default 512).
                                 The model probes self.proj to get the actual
                                 dimension and builds all downstream layers from
                                 that — so even stale hardcoded values cannot
                                 cause a shape mismatch.
    config.mil.dropout         : dropout rate (default 0.25)
    config.mil.k_sample        : CLAM k_sample (default 8)
    config.task.num_classes    : number of output classes

    Returns (model, n_params).
    """
    mil_cfg  = config.get('mil', {})
    task_cfg = config.get('task', {})

    model_key     = mil_cfg.get('model', 'abmil').lower().strip()
    encoding_size = mil_cfg.get('encoding_size', 1024)
    n_classes     = task_cfg.get('num_classes', 2)
    hidden_dim    = mil_cfg.get('hidden_dim', 256)
    dropout       = mil_cfg.get('dropout', 0.25)
    k_sample      = mil_cfg.get('k_sample', 8)
    proj_dim      = int(mil_cfg.get('proj_dim',
                        mil_cfg.get('feature_proj_dim', 512)))

    if model_key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MIL model '{model_key}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}")

    model = _MODEL_REGISTRY[model_key](
        encoding_size = encoding_size,
        n_classes     = n_classes,
        hidden_dim    = hidden_dim,
        proj_dim      = proj_dim,
        dropout       = dropout,
        k_sample      = k_sample,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_params


def has_attention(model: nn.Module) -> bool:
    """True if the model produces per-patch attention scores."""
    return isinstance(model, (ABMIL, CLAM_SB, CLAM_MB, DSMIL))
