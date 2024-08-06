import torch
import torch.nn as nn
from models.parts import *

class Transitional(nn.Module):
    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super(Transitional, self).__init__()
        self.convert_rgb2ycbcr = RGB2YCbCr()
        self.CPM1 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM2 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM3 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
    def forward(self, x):
        B = x.shape[0]
        ycbcr = self.convert_rgb2ycbcr(x)
        ycbcr = self.CPM1(ycbcr)
        ycbcr = self.CPM2(ycbcr)
        ycbcr = self.CPM3(ycbcr)
        return ycbcr
    
class Base(nn.Module):
    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super(Base, self).__init__()
        self.convert_ycbcr2rgb = YCbCr2RGB()
        self.CPM1 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM2 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
        self.CPM3 = CPM(in_channels=in_channels, gp=gp, hidden_channels=hidden_channels)
    def forward(self, x):
        B = x.shape[0]
        rgb = self.convert_ycbcr2rgb(x)
        rgb = self.CPM1(rgb)
        rgb = self.CPM2(rgb)
        rgb = self.CPM3(rgb)
        return rgb
    
class DualColorNetwork(nn.Module):
    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super(DualColorNetwork, self).__init__()
        self.transitional_block = Transitional(in_channels=3, gp=32, hidden_channels=64)
        self.Base_block = Base(in_channels=3, gp=32, hidden_channels=64)
        self._initialize_weights()

    def forward(self, x, infer=False, only=False):
        ycbcr = self.transitional_block(x)
        if only:
            return ycbcr
        rgb = self.Base_block(ycbcr)
        if infer:
            return rgb
        return ycbcr, rgb
    
    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(init_weights)
                