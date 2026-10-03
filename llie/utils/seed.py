import pytorch_lightning as pl
import torch


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (including DataLoader workers) and make cuDNN deterministic."""
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark=True picks non-deterministic kernels
