import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
def get_network(name, pretrained=True):
    network = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
	    "resnet101": torchvision.models.resnet101(pretrained=pretrained),
	    "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]

class Block(nn.Module):
    def __init__(self, num_channels, in_channels, hidden_channels):
        super(Block, self).__init__()
        layer = nn.ModuleList()
        for idx in range(num_channels):
            layer.append(
                nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 
                          kernel_size=1, stride=1, padding=0) if idx==0 else nn.Conv2d(hidden_channels, hidden_channels, 
                                                                                       kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU())
            )
        self.layer = nn.Sequential(*layer)
        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        B = x.shape[0]
        x = self.adaptive(self.layer(x))
        return x.reshape(B, -1)
    
class QualityNetwork(nn.Module):
    def __init__(self, encoder="resnet18", num_channels=1, hidden_div=2):
        super(QualityNetwork, self).__init__()
        encoder_block = get_network(name=encoder, pretrained=True)
        self.encoder = torch.nn.Sequential(*list(encoder_block.children())[:8])
        in_channels = self.encoder[-1][-1].bn2.bias.shape[0]
        self.convert_layer = Block(num_channels=num_channels, in_channels=2*in_channels, hidden_channels=int(2*in_channels/hidden_div))
        self.linear = nn.Linear(in_features=int(2*in_channels/hidden_div), out_features=4)
        self.sigmoid = nn.Sigmoid()
        self.encoder.eval()
        self._initialize_weights()

    def forward(self, x, ycbcr=None, rotation=False):
        with torch.no_grad():
            x = self.encoder(x)
            assert ycbcr is not None, "have to input YCBCR image into the model"
            ycbcr = self.encoder(ycbcr)            
            x_concat = torch.cat([x.detach(), ycbcr.detach()], dim=1)
        x_concat = self.convert_layer(x_concat)
        if rotation:
            return self.linear(x_concat)
        else:
            return self.sigmoid(x_concat)

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