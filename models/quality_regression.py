import torch.nn as nn
import torch

class QualityRegression(nn.Module):
    def __init__(self, in_channels):
        self.brightness = nn.Linear(in_channels, 1)
        self.focus = nn.Linear(in_channels, 1)
        self._initialize_weights()

    def forward(self, x):
        return self.brightness(x), self.focus(x)

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