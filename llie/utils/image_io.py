import os

import numpy as np
from PIL import Image


def to_uint8_images(batch):
    """(B, 3, H, W) tensor in [0, 1] -> list of (H, W, 3) uint8 arrays (values clipped)."""
    arr = batch.detach().clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return list(arr)


def save_comparison(inputs, outputs, targets, out_dir, prefix):
    """Save [input | output | target] side by side, one PNG per sample."""
    os.makedirs(out_dir, exist_ok=True)
    for i, row in enumerate(zip(to_uint8_images(inputs), to_uint8_images(outputs), to_uint8_images(targets))):
        Image.fromarray(np.concatenate(row, axis=1)).save(os.path.join(out_dir, f"{prefix}_{i}.png"))


def save_images(outputs, paths, out_dir):
    """Save each output under the basename of its source image (as PNG)."""
    os.makedirs(out_dir, exist_ok=True)
    for img, path in zip(to_uint8_images(outputs), paths):
        name = os.path.splitext(os.path.basename(path))[0] + ".png"
        Image.fromarray(img).save(os.path.join(out_dir, name))
