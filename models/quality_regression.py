import torch.nn as nn
import torch

class QualityRegression(nn.Module):
    def __init__(self, in_channels, brightness=None, focus=None):
        self.quality = nn.Linear(in_channels, 1)
        if brightness:
            self.brightness = nn.Linear(in_channels, 1)
        if focus:
            self.focus = nn.Linear(in_channels, 1)
        self._initialize_weights()

    def forward(self, x):
        if self.brightness and self.focus:
            return self.quality(x), self.brightness(x), self.focus(x)
        else:
            return self.quality(x)

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