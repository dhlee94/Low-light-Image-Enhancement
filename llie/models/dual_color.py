import torch.nn as nn

from llie.models.parts import CPM, RGB2YCbCr, YCbCr2RGB
from llie.models.weights import init_weights


class Transitional(nn.Module):
    """RGB input -> enhanced YCbCr (three stacked CPMs)."""

    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super().__init__()
        self.convert_rgb2ycbcr = RGB2YCbCr()
        self.CPM1 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM2 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM3 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)

    def forward(self, x):
        return self.CPM3(self.CPM2(self.CPM1(self.convert_rgb2ycbcr(x))))


class Base(nn.Module):
    """Enhanced YCbCr -> enhanced RGB (three stacked CPMs)."""

    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super().__init__()
        self.convert_ycbcr2rgb = YCbCr2RGB()
        self.CPM1 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM2 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM3 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)

    def forward(self, x):
        return self.CPM3(self.CPM2(self.CPM1(self.convert_ycbcr2rgb(x))))


class DualColorNetwork(nn.Module):
    """Two-stage enhancement network working in YCbCr then RGB.

    forward(x)              -> (ycbcr, rgb)   training
    forward(x, infer=True)  -> rgb            inference
    forward(x, only=True)   -> ycbcr          used as a feature by QualityNetwork
    """

    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super().__init__()
        # Constructor arguments were previously ignored (hard-coded 3/32/64).
        self.transitional_block = Transitional(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.Base_block = Base(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.apply(init_weights)

    def forward(self, x, infer=False, only=False):
        ycbcr = self.transitional_block(x)
        if only:
            return ycbcr
        rgb = self.Base_block(ycbcr)
        if infer:
            return rgb
        return ycbcr, rgb
