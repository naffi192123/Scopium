"""
models/classifier.py

Patch-level classifier factory and loader.

Supports the checkpoint format produced by common patch-level training pipelines:
  ckpt = {
      "model_cfg"   : str   (architecture name, e.g. "linear", "mlp1", "mlp2")
      "num_classes" : int
      "model_state" : OrderedDict   (nn.Module.state_dict())
  }

Usage
-----
from models.classifier import build_classifier, load_classifier

model, class_names = load_classifier("outputs/checkpoints/BEST_MODEL.pth")
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Tissue class names (order must match the checkpoint's output logits) ──────
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
    """One hidden layer MLP: feat_dim → 256 → num_classes."""
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
    """Two hidden layer MLP: feat_dim → 512 → 256 → num_classes."""
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_classifier(config: dict, num_classes: int, in_features: int = None) -> nn.Module:
    """
    Build a classifier model from a config dict.

    Parameters
    ----------
    config      : dict  Must contain key ``"model"`` with the architecture name.
                        Optional keys: ``"hidden_dim"``, ``"hidden_dim1"``,
                        ``"hidden_dim2"``, ``"in_features"``.
    num_classes : int   Number of output classes.
    in_features : int   Input feature dimension. If None, reads from config or
                        uses 2048 as a sensible default (ResNet-50 output).

    Returns
    -------
    nn.Module   (not yet on a device, in train mode)
    """
    arch = config.get("model", "linear").lower().strip()

    if in_features is None:
        in_features = int(config.get("in_features", 2048))

    if arch in ("linear", "fc"):
        model = LinearClassifier(in_features, num_classes)

    elif arch in ("mlp1", "mlp"):
        hidden = int(config.get("hidden_dim", 256))
        model  = MLP1Classifier(in_features, num_classes, hidden)

    elif arch == "mlp2":
        h1 = int(config.get("hidden_dim1", 512))
        h2 = int(config.get("hidden_dim2", 256))
        model = MLP2Classifier(in_features, num_classes, h1, h2)

    else:
        raise ValueError(
            f"Unknown classifier architecture '{arch}'. "
            "Supported: linear, mlp1, mlp2"
        )

    logger.info(
        f"Classifier: arch={arch} | in_features={in_features} | "
        f"num_classes={num_classes}"
    )
    return model


def load_classifier(checkpoint_path: str, device=None):
    """
    Load a pretrained classifier from a .pth checkpoint.

    Expected checkpoint format
    --------------------------
    ckpt = {
        "model_cfg"   : str | dict   — passed as config["model"] or full config dict
        "num_classes" : int
        "model_state" : OrderedDict
        "class_names" : list[str]    — optional; falls back to TISSUE_CLASSES
        "in_features" : int          — optional; inferred from state_dict if absent
    }

    Parameters
    ----------
    checkpoint_path : str
    device          : torch.device  (defaults to CUDA if available)

    Returns
    -------
    model       : nn.Module  eval mode, on device
    class_names : list[str]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading classifier from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes  = int(ckpt["num_classes"])
    class_names  = ckpt.get("class_names", TISSUE_CLASSES[:num_classes])

    # model_cfg can be a string (architecture name) or a full config dict
    model_cfg = ckpt.get("model_cfg", "linear")
    if isinstance(model_cfg, str):
        cfg = {"model": model_cfg}
    else:
        cfg = dict(model_cfg)

    # Try to infer in_features from the first weight tensor in state_dict
    state = ckpt["model_state"]
    in_features = ckpt.get("in_features", None)
    if in_features is None:
        first_weight = next(
            (v for v in state.values() if v.ndim == 2), None)
        if first_weight is not None:
            in_features = first_weight.shape[1]
            logger.info(
                f"Inferred in_features={in_features} from checkpoint state_dict.")

    model = build_classifier(cfg, num_classes=num_classes, in_features=in_features)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    logger.info(
        f"Classifier loaded | classes={class_names} | device={device}")
    return model, class_names
