import torch
from torch import nn, Tensor
from typing import Optional
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.Conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, W, H = x.shape

        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.Conv(x)
        return x
    
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

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, ycbcr, pos, query_pos):
        ycbcr = ycbcr + self.dropout(self.attn(query=self.with_pos_embed(ycbcr, query_pos), 
                                       key=self.with_pos_embed(x, pos), 
                                       value=x)[0])
        return self.norm(ycbcr)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, query_pos):
        q = k = self.with_pos_embed(x, query_pos)
        x = x + self.dropout(self.attn(query=q, key=k, value=x)[0])
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, dim, encoder_depth, decoder_depth, heads, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.pre_layers = nn.ModuleList([])
        for _ in range(encoder_depth):
            self.pre_layers.append(nn.ModuleList([
                SelfAttention(dim, heads, dropout),
                FeedForward(dim, dim, dropout=dropout)
            ]))
        for _ in range(decoder_depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads, dropout),
                CrossAttention(dim, heads, dropout),                
                FeedForward(dim, dim, dropout=dropout)
            ]))
        
    def forward(self, x, ycbcr, pos_embed, query_embed):
        for attn, ff in self.pre_layers:
            x = attn(x, query_pos=query_embed) + x
            x = ff(x) + x
        for attn1, attn2, ff in self.layers:
            ycbcr = attn1(ycbcr, query_pos=query_embed) + ycbcr
            ycbcr = attn2(x, ycbcr, pos=pos_embed, query_pos=query_embed) + ycbcr
            ycbcr = ff(ycbcr) + ycbcr
        return ycbcr
    
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, encoder_depth, decoder_depth, heads, 
                 channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.to_patch_embedding = PatchEmbedding(pair(image_size), pair(patch_size), in_channels=channels, embed_dim=dim)
        self.num_patches = self.to_patch_embedding.num_patches
        self.pos_embed = PositionEmbeddingSine(dim//2, normalize=True)
        self.query_embed = nn.Embedding(self.num_patches, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, encoder_depth, decoder_depth, heads, dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, img, ycbcr):
        batch_size = img.shape[0]
        x = self.to_patch_embedding(img)
        x_ycbcr = self.to_patch_embedding(ycbcr)
        pos_embed = self.pos_embed(x, mask=None).flatten(2).permute(0, 2, 1)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.transformer(self.norm1(x.flatten(2).permute(0, 2, 1)),
                             self.norm2(x_ycbcr.flatten(2).permute(0, 2, 1)),
                             pos_embed, query_embed).mean(dim=-1)
        return x
    
class QualityNetwork(nn.Module):
    def __init__(self, image_size, patch_size, dim, encoder_depth, decoder_depth, heads, 
                 channels, drop_out, emb_dropout):
        super(QualityNetwork, self).__init__()
        self.encoder = ViT(image_size, patch_size, dim, encoder_depth, decoder_depth, 
                           heads, channels, drop_out, emb_dropout)
        self.linear = nn.Linear(in_features=int(self.encoder.num_patches), out_features=4)
        self.quality_linear = nn.Linear(in_features=int(self.encoder.num_patches), out_features=1)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def forward(self, x, ycbcr, batch_size=1, infer=False):    
        assert ycbcr is not None, "have to input YCBCR image into the model"
        if infer:
            x = self.encoder(x, ycbcr)
            return self.quality_linear(x), self.sigmoid(x)
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