"""Synthetic degradations.

Convention: the "high" variant of a pair is the MORE degraded one.
Numpy-level functions take (H, W, 3) uint8 arrays and a ``numpy.random.Generator``
so datasets can make them deterministic per sample.
"""
import io
import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

# (low, high) ranges of each degradation's parameter.
JPEG_QUALITY_HIGH = (40, 60)   # heavier compression
JPEG_QUALITY_LOW = (80, 90)
NOISE_VAR_HIGH = (5e-5, 5.1e-5)
NOISE_VAR_LOW = (1e-5, 1.1e-5)


def darken(img, factor):
    """Scale pixel values by ``factor`` (< 1 simulates low light)."""
    lut = np.clip(np.arange(256, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
    return lut[img]


def jpeg_compress(img, quality):
    """JPEG round trip in memory (no temporary files)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), optimize=True)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def add_gaussian_noise(img, var, rng):
    """Additive Gaussian noise with variance ``var`` on the [0, 1] scale."""
    noisy = img.astype(np.float32) / 255.0 + rng.normal(0.0, math.sqrt(var), size=img.shape)
    return (np.clip(noisy, 0.0, 1.0) * 255).astype(np.uint8)


def gaussian_kernel_size(sigma, max_size):
    """Odd kernel covering +-3 sigma, capped to fit inside the image."""
    k = 2 * math.ceil(3 * sigma) + 1
    cap = max_size if max_size % 2 else max_size - 1
    return max(3, min(k, cap))


def random_gaussian_blur(x, sigma_range, generator=None):
    """Blur a (B, C, H, W) tensor with one sigma drawn uniformly from ``sigma_range``.

    The kernel size follows sigma (previously a fixed 5x5 kernel made every
    sigma >= 5 look like the same box blur).
    """
    lo, hi = sigma_range
    sigma = lo + (hi - lo) * torch.rand(1, generator=generator).item()
    k = gaussian_kernel_size(sigma, min(x.shape[-2:]))
    return gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
