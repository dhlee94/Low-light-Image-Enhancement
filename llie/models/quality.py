"""Transformer-based quality network.

Tokens are image patches of the input (self-attention encoder) and of its
YCbCr enhancement (cross-attention decoder). All attention layers use
``batch_first=True``: tensors are (batch, tokens, dim) throughout. The image
feature is the layer-normalized mean of the output tokens, (batch, dim).
"""
import math
from typing import Optional

import torch
from torch import Tensor, nn

from llie.models.weights import init_weights


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
    return tensor if pos is None else tensor + pos


class PositionEmbeddingSine(nn.Module):
    """2D sine position embedding (DETR style); outputs 2 * num_pos_feats channels."""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

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
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            "Image dimensions must be divisible by the patch size."
        self.img_size = img_size  # (H, W)
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.Conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, h, w = x.shape
        assert (h, w) == tuple(self.img_size), f"expected input size {tuple(self.img_size)}, got {(h, w)}"
        return self.Conv(x)


class FeedForward(nn.Module):
    """Pre-norm MLP; the caller adds the residual."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Post-norm cross-attention block; the residual is included: norm(q + attn(q, kv))."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ycbcr, pos, query_pos):
        attended = self.attn(query=with_pos_embed(ycbcr, query_pos), key=with_pos_embed(x, pos), value=x)[0]
        return self.norm(ycbcr + self.dropout(attended))


class SelfAttention(nn.Module):
    """Post-norm self-attention block; the residual is included: norm(x + attn(x))."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, query_pos):
        q = k = with_pos_embed(x, query_pos)
        return self.norm(x + self.dropout(self.attn(query=q, key=k, value=x)[0]))


class Transformer(nn.Module):
    def __init__(self, dim, encoder_depth, decoder_depth, heads, dropout=0.0):
        super().__init__()
        self.pre_layers = nn.ModuleList([
            nn.ModuleList([SelfAttention(dim, heads, dropout), FeedForward(dim, dim, dropout=dropout)])
            for _ in range(encoder_depth)
        ])
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(dim, heads, dropout),
                CrossAttention(dim, heads, dropout),
                FeedForward(dim, dim, dropout=dropout),
            ])
            for _ in range(decoder_depth)
        ])

    def forward(self, x, ycbcr, pos_embed, query_embed):
        # Attention blocks already contain their residual connection, so only the
        # feed-forward blocks add one here (previously the residual was added twice).
        for attn, ff in self.pre_layers:
            x = attn(x, query_pos=query_embed)
            x = ff(x) + x
        for self_attn, cross_attn, ff in self.layers:
            ycbcr = self_attn(ycbcr, query_pos=query_embed)
            ycbcr = cross_attn(x, ycbcr, pos=pos_embed, query_pos=query_embed)
            ycbcr = ff(ycbcr) + ycbcr
        return ycbcr


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, encoder_depth, decoder_depth, heads,
                 channels=3, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.to_patch_embedding = PatchEmbedding(pair(image_size), pair(patch_size), in_channels=channels, embed_dim=dim)
        self.num_patches = self.to_patch_embedding.num_patches
        self.pos_embed = PositionEmbeddingSine(dim // 2, normalize=True)
        self.query_embed = nn.Embedding(self.num_patches, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, encoder_depth, decoder_depth, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, img, ycbcr):
        """(B, C, H, W) x 2 -> image feature (B, dim)."""
        batch_size = img.shape[0]
        x = self.to_patch_embedding(img)
        x_ycbcr = self.to_patch_embedding(ycbcr)
        pos_embed = self.pos_embed(x).flatten(2).transpose(1, 2)               # (B, N, dim)
        query_embed = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = self.dropout(self.norm1(x.flatten(2).transpose(1, 2)))         # (B, N, dim)
        tokens_ycbcr = self.dropout(self.norm2(x_ycbcr.flatten(2).transpose(1, 2)))
        out = self.transformer(tokens, tokens_ycbcr, pos_embed, query_embed)
        # Average over patches, not channels: averaging each token over its channels
        # (the previous version) left one scalar per patch, tied to its position.
        return self.norm_out(out.mean(dim=1))


class QualityNetwork(nn.Module):
    """Image feature (dim) + two heads: quality score (1) and rotation class (4).

    forward(x, ycbcr, infer=True)               -> (score, feature)
    forward(x, ycbcr, batch_size, infer=False)  -> (score, feature, rotation_logits)
        where the first 3 * batch_size rows are quality samples and the rest are
        rotated samples (kept for backward compatibility; QualityModule uses
        encode/score/classify_rotation directly).
    """

    def __init__(self, image_size, patch_size, dim, encoder_depth, decoder_depth, heads,
                 channels, drop_out, emb_dropout):
        super().__init__()
        self.encoder = ViT(image_size, patch_size, dim, encoder_depth, decoder_depth,
                           heads, channels, drop_out, emb_dropout)
        self.linear = nn.Linear(in_features=dim, out_features=4)
        self.quality_linear = nn.Linear(in_features=dim, out_features=1)
        self.apply(init_weights)

    def encode(self, x, ycbcr):
        return self.encoder(x, ycbcr)

    def score(self, feature):
        return self.quality_linear(feature)

    def classify_rotation(self, feature):
        return self.linear(feature)

    def forward(self, x, ycbcr, batch_size=1, infer=False):
        feature = self.encode(x, ycbcr)
        if infer:
            return self.score(feature), feature
        quality, rotate = feature[:3 * batch_size], feature[3 * batch_size:]
        return self.score(quality), quality, self.classify_rotation(rotate)
