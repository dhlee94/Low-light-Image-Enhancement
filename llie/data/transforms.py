"""Resize / to-tensor transforms for (H, W, C) uint8 numpy images."""
import cv2
import numpy as np
import torch


def round_to_multiple(n, base=64):
    """Nearest multiple of ``base`` (never 0)."""
    return max(base, int(round(n / base)) * base)


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Resize:
    """Resize to ``size`` = (H, W).

    stretch=True: resize exactly to ``size`` (aspect ratio not preserved).
    stretch=False: keep aspect ratio within ``size``, rounded to multiples of 64;
        never upscale unless ``scaleup``.
    """

    def __init__(self, size=(1024, 1024), interpolation=cv2.INTER_LINEAR, scaleup=True, stretch=False):
        self.size = size
        self.interpolation = interpolation
        self.scaleup = scaleup
        self.stretch = stretch

    def __call__(self, data):
        h, w = data.shape[:2]
        if self.stretch:
            new_h, new_w = self.size
        else:
            r = min(self.size[0] / h, self.size[1] / w)
            if not self.scaleup:
                r = min(r, 1.0)
            new_h, new_w = round_to_multiple(h * r), round_to_multiple(w * r)
        data = cv2.resize(data, (new_w, new_h), interpolation=self.interpolation)  # cv2 takes (W, H)
        return np.ascontiguousarray(data)


class ToTensor:
    """(H, W, C) uint8 -> (C, H, W) float tensor in [0, 1]."""

    def __call__(self, data):
        return torch.from_numpy(np.ascontiguousarray(data.transpose(2, 0, 1))).float().div_(255.0)


def build_transform(size, interpolation=cv2.INTER_LINEAR, scaleup=False, stretch=True):
    """``size=None`` keeps the original resolution (only converts to a tensor)."""
    if size is None:
        return ToTensor()
    return Compose([Resize(size, interpolation=interpolation, scaleup=scaleup, stretch=stretch), ToTensor()])
