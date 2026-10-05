"""Loss functions."""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(x):
    """Mean squared difference between neighbouring pixels."""
    h_diff = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
    w_diff = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
    return (h_diff + w_diff) / x.numel()


def color_loss(output, target):
    """1 - mean per-pixel cosine similarity of RGB vectors."""
    return 1 - torch.mean(F.cosine_similarity(output, target, dim=1))


class GroupContrastiveLoss(nn.Module):
    """Contrastive loss between two groups of embeddings (low / high degradation).

    For every sample the "positive" score is its mean cosine similarity to the
    other members of its own group; all cross-group pairs are negatives:

        loss = -log( sum_i  exp(p_i / T) / (exp(p_i / T) + sum_j exp(s_ij / T)) / 2B )

    Masks are built per call, so the module can be created once and reused with
    any batch size. A batch of one has no within-group pair; the loss is then 0.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = float(temperature)
        self._warned_small_batch = False

    def forward(self, emb_low, emb_high):
        b = emb_low.shape[0]
        if b < 2:
            if not self._warned_small_batch:
                warnings.warn("GroupContrastiveLoss needs batch size >= 2; returning 0 for this batch.")
                self._warned_small_batch = True
            return emb_low.sum() * 0.0  # keeps the graph so backward() still works

        z = F.normalize(torch.cat([emb_low, emb_high], dim=0), dim=1)
        sim = z @ z.t()  # cosine similarity, (2B, 2B)

        off_diag = 1.0 - torch.eye(b, device=z.device, dtype=z.dtype)
        sim_low_low = (sim[:b, :b] * off_diag).sum(dim=1) / (b - 1)
        sim_high_high = (sim[b:, b:] * off_diag).sum(dim=1) / (b - 1)
        positives = torch.cat([sim_low_low, sim_high_high])

        cross_group = torch.ones_like(sim)
        cross_group[:b, :b] = 0
        cross_group[b:, b:] = 0

        numerator = torch.exp(positives / self.temperature)
        denominator = (cross_group * torch.exp(sim / self.temperature)).sum(dim=1)
        return -torch.log((numerator / (numerator + denominator)).sum() / (2 * b))


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020) over L2-normalized embeddings.

    Every embedding with the same label is a positive of the anchor, all others are
    negatives:  loss_i = -mean_{p in P(i)} log( exp(s_ip / T) / sum_{a != i} exp(s_ia / T) ),
    averaged over anchors that have at least one positive (0 if none do).
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, embeddings, labels):
        z = F.normalize(embeddings, dim=1)
        logits = z @ z.t() / self.temperature
        n = z.shape[0]
        not_self = ~torch.eye(n, dtype=torch.bool, device=z.device)
        positives = (labels[:, None] == labels[None, :]) & not_self

        logits = logits.masked_fill(~not_self, float("-inf"))
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        n_pos = positives.sum(dim=1)
        has_pos = n_pos > 0
        if not has_pos.any():
            return embeddings.sum() * 0.0  # keeps the graph so backward() still works
        mean_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1)[has_pos] / n_pos[has_pos]
        return -mean_log_prob.mean()
