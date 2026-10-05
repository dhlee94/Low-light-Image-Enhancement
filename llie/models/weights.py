"""Weight initialization and checkpoint loading shared by all networks."""
from __future__ import annotations

import torch
import torch.nn as nn


def init_weights(m: nn.Module) -> None:
    """Xavier for Linear, Kaiming for Conv2d, zero bias. Use with ``module.apply``."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def load_weights(model: nn.Module, path: str) -> nn.Module:
    """Load a ``state_dict`` saved with ``torch.save(model.state_dict(), path)``.

    Loads onto CPU first so weights saved on another GPU id still load, then
    copies into the model's current device. Any key mismatch raises.
    """
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
