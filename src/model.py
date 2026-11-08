"""One-logit full-image classifiers built with timm."""

import os
from pathlib import Path

# Keep downloaded weights inside the project rather than writing user-wide caches.
_cache = Path(__file__).resolve().parents[1] / "outputs" / ".cache"
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_cache / "torch"))

import timm
from torch import nn


class BinaryImageClassifier(nn.Module):
    """Backbone + global average pooling + a single unnormalized Fake logit."""

    def __init__(self, name="efficientnet_b0", pretrained=True, drop_rate=0.2):
        super().__init__()
        self.network = timm.create_model(name, pretrained=pretrained, num_classes=1, global_pool="avg", drop_rate=drop_rate)

    def forward(self, images):
        """Return [batch] logits; sigmoid belongs only in prediction/metrics."""
        return self.network(images).squeeze(-1)


def build_model(config, pretrained=None):
    """Construct a model; loading a checkpoint explicitly disables weight downloads."""
    return BinaryImageClassifier(config["name"], config["pretrained"] if pretrained is None else pretrained, config.get("drop_rate", 0.2))
