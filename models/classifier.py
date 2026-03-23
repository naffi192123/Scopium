"""
models/classifier.py

Patch-level classifier factory and loader.

Supports the checkpoint format:
  ckpt = {
      "model_cfg"   : str | dict  (architecture name)
      "num_classes" : int
      "model_state" : OrderedDict
      "class_names" : list[str]   (optional)
      "in_features" : int         (optional; auto-inferred from state_dict)
  }

Supported architectures
-----------------------
  linear        : feat_dim → num_classes  (single Linear)
  mlp1 / mlp    : feat_dim → 256 → num_classes
  mlp2           : feat_dim → 512 → 256 → num_classes
  gated_mlp      : gate attention + 2 BN blocks + head
                   (state keys: gate.*, mlp.blocks.*, mlp.head.*)

The loader auto-detects the architecture from state_dict keys when
model_cfg is unknown or doesn't match any registered name.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TISSUE_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


# ---------------------------------------------------------------------------
# Architecture definitions
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    """Single linear layer: feat_dim → num_classes."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLP1Classifier(nn.Module):
    """feat_dim → 256 → num_classes (ReLU + Dropout)."""
    def __init__(self, in_features: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MLP2Classifier(nn.Module):
    """feat_dim → 512 → 256 → num_classes (ReLU + Dropout)."""
    def __init__(self, in_features: int, num_classes: int,
                 hidden1: int = 512, hidden2: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── GatedMLP: matches state dict with gate.*, mlp.blocks.*, mlp.head.* ──────

class _BNBlock(nn.Module):
    """Linear + BatchNorm1d block (state keys: block.0.* and block.1.*)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return F.relu(self.block(x))


class _MLPBody(nn.Module):
    """Two _BNBlock layers + a head Linear (state keys: blocks.*, head.*)."""
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            _BNBlock(in_features, hidden_dim),
            _BNBlock(hidden_dim,  hidden_dim),
        ])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class GatedMLPClassifier(nn.Module):
    """
    Gated-attention MLP classifier.

    State dict layout::
      gate.0.*             Linear(in_features, gate_dim)
      gate.1               ReLU  (no params → index 2 is next Linear)
      gate.2.*             Linear(gate_dim, in_features)
      mlp.blocks.0.block.0.*  Linear(in_features, hidden_dim)
      mlp.blocks.0.block.1.*  BatchNorm1d(hidden_dim)
      mlp.blocks.1.block.0.*  Linear(hidden_dim, hidden_dim)
      mlp.blocks.1.block.1.*  BatchNorm1d(hidden_dim)
      mlp.head.*           Linear(hidden_dim, num_classes)
    """
    def __init__(self, in_features: int, gate_dim: int,
                 hidden_dim: int, num_classes: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, in_features),
        )
        self.mlp = _MLPBody(in_features, hidden_dim, num_classes)

    def forward(self, x):
        attn = torch.sigmoid(self.gate(x))   # soft gate per feature dim
        return self.mlp(x * attn)


# ---------------------------------------------------------------------------
# Auto-detection from state_dict keys
# ---------------------------------------------------------------------------

def _detect_arch_from_state(state: dict) -> str:
    """
    Inspect the state_dict keys and return the most likely architecture name.
    Falls back to 'linear' when uncertain.
    """
    keys = set(state.keys())
    if any(k.startswith("gate.") for k in keys):
        return "gated_mlp"
    if "fc.weight" in keys:
        return "linear"
    if "net.0.weight" in keys and "net.4.weight" in keys:
        return "mlp2"
    if "net.0.weight" in keys:
        return "mlp1"
    return "linear"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_classifier(config: dict, num_classes: int,
                     in_features: int = None,
                     state: dict = None) -> nn.Module:
    """
    Build a classifier model.

    Parameters
    ----------
    config      : dict  Must contain key ``"model"`` with arch name.
    num_classes : int
    in_features : int   If None, reads config or uses 2048.
    state       : dict  Optional state_dict; used to auto-infer dimensions
                        for architectures like gated_mlp.
    """
    arch = (config.get("model") or "linear").lower().strip()

    if in_features is None:
        in_features = int(config.get("in_features", 2048))

    # ── Simple architectures ─────────────────────────────────────────────────
    if arch in ("linear", "fc"):
        model = LinearClassifier(in_features, num_classes)

    elif arch in ("mlp1", "mlp"):
        hidden = int(config.get("hidden_dim", 256))
        model  = MLP1Classifier(in_features, num_classes, hidden)

    elif arch == "mlp2":
        h1 = int(config.get("hidden_dim1", 512))
        h2 = int(config.get("hidden_dim2", 256))
        model = MLP2Classifier(in_features, num_classes, h1, h2)

    # ── GatedMLP: infer dims from state_dict if available ────────────────────
    elif arch == "gated_mlp":
        if state is not None:
            # gate.0.weight shape: (gate_dim, in_features)
            gate_w = state.get("gate.0.weight")
            block_w = state.get("mlp.blocks.0.block.0.weight")
            gate_dim   = gate_w.shape[0]  if gate_w   is not None else in_features
            hidden_dim = block_w.shape[0] if block_w  is not None else 256
        else:
            gate_dim   = int(config.get("gate_dim",    in_features))
            hidden_dim = int(config.get("hidden_dim",  256))
        model = GatedMLPClassifier(in_features, gate_dim, hidden_dim, num_classes)

    else:
        raise ValueError(
            f"Unknown classifier architecture '{arch}'. "
            "Supported: linear, mlp1, mlp2, gated_mlp"
        )

    logger.info(
        f"Classifier: arch={arch} | in_features={in_features} | "
        f"num_classes={num_classes}")
    return model


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_classifier(checkpoint_path: str, device=None):
    """
    Load a pretrained classifier from a .pth checkpoint.

    Auto-detects the architecture from state_dict keys so the correct model
    is always reconstructed even when model_cfg is unknown or incorrect.

    Returns
    -------
    model       : nn.Module  eval mode, on device
    class_names : list[str]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading classifier from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = int(ckpt["num_classes"])
    class_names = ckpt.get("class_names", TISSUE_CLASSES[:num_classes])
    state       = ckpt["model_state"]

    # ── Resolve architecture name ─────────────────────────────────────────────
    model_cfg = ckpt.get("model_cfg", "linear")
    if isinstance(model_cfg, str):
        cfg = {"model": model_cfg}
    else:
        cfg = dict(model_cfg)

    # Auto-detect if model_cfg is unknown / doesn't match keys
    detected = _detect_arch_from_state(state)
    declared = cfg.get("model", "linear").lower().strip()
    if detected != declared:
        logger.warning(
            f"model_cfg says '{declared}' but state_dict looks like '{detected}'. "
            f"Using auto-detected architecture '{detected}'.")
        cfg["model"] = detected

    # ── Infer in_features from the first 2-D weight tensor ───────────────────
    in_features = ckpt.get("in_features", None)
    if in_features is None:
        first_w = next((v for v in state.values() if v.ndim == 2), None)
        if first_w is not None:
            in_features = first_w.shape[1]
            logger.info(f"Inferred in_features={in_features} from state_dict.")

    # ── Build and load ────────────────────────────────────────────────────────
    model = build_classifier(
        cfg, num_classes=num_classes, in_features=in_features, state=state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    logger.info(f"Classifier loaded | classes={class_names} | device={device}")
    return model, class_names
