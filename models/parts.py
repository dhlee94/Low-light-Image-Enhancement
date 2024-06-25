import torch
import torch.nn as nn
import torch.nn.functional as F

class RGB2YCbCr(nn.Module):
    def __init__(self):
        super(RGB2YCbCr, self).__init__()

    def forward(self, x):
        R = x[:, 0, :, :]
        G = x[:, 1, :, :]
        B = x[:, 2, :, :]
        
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128 / 255.0
        Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128 / 255.0
        
        out = torch.stack((Y, Cb, Cr), dim=1)
        return out

class YCbCr2RGB(nn.Module):
    def __init__(self):
        super(YCbCr2RGB, self).__init__()

    def forward(self, x):
        Y = x[:, 0, :, :]
        Cb = x[:, 1, :, :] - 128 / 255.0
        Cr = x[:, 2, :, :] - 128 / 255.0
        
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
        
        out = torch.stack((R, G, B), dim=1)
        return out
    
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, batchnorm=False, bias=True):
        super(Conv, self).__init__()
        conv = nn.ModuleList()
        conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if batchnorm:
            conv.append(nn.BatchNorm2d(out_channels))
        conv.append(nn.ReLU())        
        self.conv = nn.Sequential(*conv)
    def forward(self, x):
        return self.conv(x)
    
class GlobalPrior(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=64):
        super(GlobalPrior, self).__init__()
        self.downsample = Conv(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=4, padding=3)
        self.conv1 = Conv(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = Conv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.fc = nn.Linear(hidden_channels*3, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        std_pool = torch.std(x.view(x.size(0), x.size(1), -1), dim=2)
        
        global_prior = torch.cat([max_pool, avg_pool, std_pool], dim=1)
        global_prior = self.fc(global_prior)
        return torch.tanh(global_prior)
    
class CPM(nn.Module):
    def __init__(self, in_channels=3, gp=32, hidden_channels=64):
        super(CPM, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.layer = Conv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.global_layer = GlobalPrior(in_channels=in_channels, hidden_channels=gp, out_channels=hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.in_channels= in_channels

    def forward(self, x):
        B, C, H, W  = x.shape
        gp = self.global_layer(x).view(B, -1, 1, 1)
        x = self.fc1(x.reshape(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, -1, H, W)        
        x = self.layer((x+gp))
        return self.fc2(x.reshape(B, -1, H*W).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, self.in_channels, H, W)