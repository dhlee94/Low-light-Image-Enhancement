import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from timm.models.layers import to_2tuple
from einops import rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class DualAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, ycbcr):
        x = self.norm(x)
        ycbcr = self.norm(ycbcr)

        k, v = self.to_kv(x).chunk(2, dim = -1)
        q = self.to_q(ycbcr)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(nn.ModuleList([
                DualAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout) if idx==0 else \
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, dim, dropout = dropout)
            ]))

    def forward(self, x, ycbcr):
        for idx, (attn, ff) in enumerate(self.layers):
            x = attn(x, ycbcr) + ycbcr if idx==0 else attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.Conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, W, H = x.shape

        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.Conv(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        self.to_patch_embedding = PatchEmbedding(image_size, patch_size, in_channels=channels, embed_dim=dim, norm_layer=nn.LayerNorm)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dropout)

    def forward(self, img, ycbcr):
        x = self.to_patch_embedding(img)
        x_ycbcr = self.to_patch_embedding(ycbcr)

        x = self.dropout(x+self.pos_embedding)
        ycbcr = self.dropout(x_ycbcr+self.pos_embedding)
        x = self.transformer(x, ycbcr).mean(dim=-1)
        return x
    
class QualityNetwork(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, channels, dim_head, drop_out, emb_dropout):
        super(QualityNetwork, self).__init__()
        self.encoder = ViT(image_size, patch_size, dim, depth, heads, channels, dim_head, drop_out, emb_dropout)
        self.linear = nn.Linear(in_features=int(self.encoder.num_patches), out_features=4)
        self.quality_linear = nn.Linear(in_features=int(self.encoder.num_patches), out_features=1)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x, ycbcr, batch_size=1, infer=False):    
        assert ycbcr is not None, "have to input YCBCR image into the model"
        if infer:
            x = self.encoder(x, ycbcr)
            return self.quality_linear(self.sigmoid(x)), self.sigmoid(x)
        else:
            x = self.encoder(x, ycbcr)
            quality = x[:3*batch_size, :]
            rotate = x[3*batch_size:, :]
            return self.quality_linear(quality), self.sigmoid(quality), self.linear(rotate)

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