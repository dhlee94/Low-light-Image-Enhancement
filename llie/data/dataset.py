"""Datasets. Every item is a dict of (3, H, W) float tensors in [0, 1].

ImageDataset (dual task):    {"input", "target_ycbcr", "target_rgb"}
    input is the image darkened synthetically, or the paired low-light image
    when ``input_paths`` is given (e.g. LOL).
QualityImageDataset:         {"input", "jpeg_high", "jpeg_low", "noise_high", "noise_low"}
Either with infer=True:      {"input", "path"}

``seed=None`` draws fresh random degradations each time (training); an integer
seed makes sample ``idx`` always get the same degradation (validation / test).
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from llie import config
from llie.data import degradations as D

_DARKEN_RANGE = tuple(config.TASKS["dual"]["darken_range"])


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
    """Low-light pairs: target is the image at ``paths[idx]``; input is ``input_paths[idx]``
    if given, otherwise the target darkened by a random factor."""

    def __init__(self, paths: Sequence[str], transform, img_mode="RGB", infer=False,
                 darken_range: Tuple[float, float] = _DARKEN_RANGE, seed: Optional[int] = None,
                 input_paths: Optional[Sequence[str]] = None):
        if input_paths is not None and len(input_paths) != len(paths):
            raise ValueError(f"{len(input_paths)} input paths for {len(paths)} target paths")
        self.paths = list(paths)
        self.input_paths = None if input_paths is None else list(input_paths)
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
        if self.infer:
            # With paired data the image to enhance is the low-light one.
            path = self.input_paths[idx] if self.input_paths is not None else self.paths[idx]
            return {"input": self.transform(np.array(load_image(path, self.img_mode))), "path": path}

        img = load_image(self.paths[idx], self.img_mode)
        rgb = np.array(img)
        if self.input_paths is not None:
            dark = np.array(load_image(self.input_paths[idx], self.img_mode))
            if dark.shape != rgb.shape:
                raise ValueError(f"size mismatch: {self.input_paths[idx]} {dark.shape} vs {self.paths[idx]} {rgb.shape}")
        else:
            dark = D.darken(rgb, self._rng(idx).uniform(*self.darken_range))
        return {
            "input": self.transform(dark),
            "target_ycbcr": self.transform(np.array(img.convert("YCbCr"))),
            "target_rgb": self.transform(rgb),
        }


class QualityImageDataset(ImageDataset):
    """Clean image plus a strong/weak pair for JPEG compression and Gaussian noise.

    ``degradations`` holds the (min, max) parameter ranges, keyed as in
    ``config.DEGRADATION_PRESETS`` (jpeg_quality_*, noise_var_*); other keys are
    ignored and missing ones come from the configured default preset.
    """

    def __init__(self, *args, degradations: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        preset = config.DEGRADATION_PRESETS[config.TASKS["quality"]["degradation_preset"]]
        self.degradations = dict(preset, **(degradations or {}))

    def __getitem__(self, idx):
        path = self.paths[idx]
        rgb = np.array(load_image(path, self.img_mode))
        if self.infer:
            return {"input": self.transform(rgb), "path": path}

        rng = self._rng(idx)
        d = self.degradations
        # Degrade the mode-converted image, then convert again: in "L" mode every
        # output stays grayscale (previously the JPEG pair kept its colours).
        degraded = {
            "jpeg_high": D.jpeg_compress(rgb, rng.uniform(*d["jpeg_quality_high"])),
            "jpeg_low": D.jpeg_compress(rgb, rng.uniform(*d["jpeg_quality_low"])),
            "noise_high": D.add_gaussian_noise(rgb, rng.uniform(*d["noise_var_high"]), rng),
            "noise_low": D.add_gaussian_noise(rgb, rng.uniform(*d["noise_var_low"]), rng),
        }
        item = {"input": self.transform(rgb)}
        item.update({k: self.transform(apply_mode(v, self.img_mode)) for k, v in degraded.items()})
        return item
