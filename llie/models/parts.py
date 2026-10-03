"""Building blocks of DualColorNetwork.

Module attribute names are part of the checkpoint format (state_dict keys);
do not rename them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Offset of the Cb/Cr channels (128 on the 0-255 scale) for inputs in [0, 1].
_CHROMA_OFFSET = 128.0 / 255.0


class RGB2YCbCr(nn.Module):
    """RGB -> YCbCr for tensors in [0, 1], shape (B, 3, H, W)."""

    def forward(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.169 * r - 0.331 * g + 0.5 * b + _CHROMA_OFFSET
        cr = 0.5 * r - 0.419 * g - 0.081 * b + _CHROMA_OFFSET
        return torch.stack((y, cb, cr), dim=1)


class YCbCr2RGB(nn.Module):
    """YCbCr -> RGB for tensors in [0, 1], shape (B, 3, H, W)."""

    def forward(self, x):
        y = x[:, 0]
        cb = x[:, 1] - _CHROMA_OFFSET
        cr = x[:, 2] - _CHROMA_OFFSET
        r = y + 1.403 * cr
        g = y - 0.344 * cb - 0.714 * cr
        b = y + 1.773 * cb
        return torch.stack((r, g, b), dim=1)


class Conv(nn.Module):
    """Conv2d [+ BatchNorm] + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, batchnorm=False, bias=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class GlobalPrior(nn.Module):
    """Image-level descriptor: downsample, conv, then [max, mean, std] pooling -> FC -> tanh."""

    def __init__(self, in_channels=3, hidden_channels=32, out_channels=64):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=7, stride=4, padding=3)
        self.conv1 = Conv(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = Conv(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_channels * 3, out_channels)

    def forward(self, x):
        x = self.conv2(self.conv1(self.downsample(x)))
        flat = x.flatten(2)
        max_pool = flat.amax(dim=2)
        avg_pool = flat.mean(dim=2)
        std_pool = flat.std(dim=2)
        return torch.tanh(self.fc(torch.cat([max_pool, avg_pool, std_pool], dim=1)))


class CPM(nn.Module):
    """Color processing module: per-pixel MLP conditioned on a global prior."""

    def __init__(self, in_channels=3, gp=32, hidden_channels=64, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.layer = Conv(in_channels=hidden_channels, out_channels=hidden_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding)
        self.global_layer = GlobalPrior(in_channels=in_channels, hidden_channels=gp, out_channels=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x):
        gp = self.global_layer(x)[:, :, None, None]  # (B, hidden, 1, 1), broadcast over pixels
        return self.fc2(self.layer(self.fc1(x) + gp))
