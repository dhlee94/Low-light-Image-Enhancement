"""Datasets. Every item is a dict of (3, H, W) float tensors in [0, 1].

ImageDataset (dual task):    {"input", "target_ycbcr", "target_rgb"}
QualityImageDataset:         {"input", "jpeg_high", "jpeg_low", "noise_high", "noise_low"}
Either with infer=True:      {"input", "path"}

``seed=None`` draws fresh random degradations each time (training); an integer
seed makes sample ``idx`` always get the same degradation (validation / test).
"""
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from llie.data import degradations as D


def load_image(path, img_mode="RGB"):
    """Open as 3-channel RGB; img_mode="L" converts to grayscale first."""
    img = Image.open(path)
    if img_mode == "L":
        img = img.convert("L")
    return img.convert("RGB")


def apply_mode(arr, img_mode):
    """Re-apply the grayscale conversion after a degradation that may add colour."""
    if img_mode != "L":
        return arr
    return np.array(Image.fromarray(arr).convert("L").convert("RGB"))


class ImageDataset(Dataset):
    """Low-light pairs: input is the image darkened by a random factor, target is the image."""

    def __init__(self, paths: Sequence[str], transform, img_mode="L", infer=False,
                 darken_range: Tuple[float, float] = (0.5, 0.9), seed: Optional[int] = None):
        self.paths = list(paths)
        self.transform = transform
        self.img_mode = img_mode
        self.infer = infer
        self.darken_range = darken_range
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def _rng(self, idx):
        if self.seed is not None:
            return np.random.default_rng((self.seed, idx))
        # Derived from the global numpy state, which seed_everything seeds per worker.
        return np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = load_image(path, self.img_mode)
        rgb = np.array(img)
        if self.infer:
            return {"input": self.transform(rgb), "path": path}

        dark = D.darken(rgb, self._rng(idx).uniform(*self.darken_range))
        return {
            "input": self.transform(dark),
            "target_ycbcr": self.transform(np.array(img.convert("YCbCr"))),
            "target_rgb": self.transform(rgb),
        }


class QualityImageDataset(ImageDataset):
    """Clean image plus a strong/weak pair for JPEG compression and Gaussian noise."""

    def __getitem__(self, idx):
        path = self.paths[idx]
        rgb = np.array(load_image(path, self.img_mode))
        if self.infer:
            return {"input": self.transform(rgb), "path": path}

        rng = self._rng(idx)
        # Degrade the mode-converted image, then convert again: in "L" mode every
        # output stays grayscale (previously the JPEG pair kept its colours).
        degraded = {
            "jpeg_high": D.jpeg_compress(rgb, rng.uniform(*D.JPEG_QUALITY_HIGH)),
            "jpeg_low": D.jpeg_compress(rgb, rng.uniform(*D.JPEG_QUALITY_LOW)),
            "noise_high": D.add_gaussian_noise(rgb, rng.uniform(*D.NOISE_VAR_HIGH), rng),
            "noise_low": D.add_gaussian_noise(rgb, rng.uniform(*D.NOISE_VAR_LOW), rng),
        }
        item = {"input": self.transform(rgb)}
        item.update({k: self.transform(apply_mode(v, self.img_mode)) for k, v in degraded.items()})
        return item
