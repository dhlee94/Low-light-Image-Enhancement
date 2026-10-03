import torch


@torch.no_grad()
def select_hardest_pair(color_model, scorer, highs, lows):
    """Pick, per sample, the degradation pair the scorer separates the most.

    Args:
        color_model: DualColorNetwork, used to compute the YCbCr branch input.
        scorer: QualityNetwork used for selection (the EMA copy).
        highs, lows: lists of K tensors (B, C, H, W); ``highs[k]`` is the strongly
            and ``lows[k]`` the weakly degraded version for degradation type k.

    Returns:
        (high, low, high_ycbcr, low_ycbcr), each (B, C, H, W).
    """
    high = torch.stack(highs, dim=1)  # (B, K, C, H, W)
    low = torch.stack(lows, dim=1)
    b, k = high.shape[:2]

    high_ycbcr = color_model(high.flatten(0, 1), only=True)
    low_ycbcr = color_model(low.flatten(0, 1), only=True)
    score_high, _ = scorer(high.flatten(0, 1), high_ycbcr, infer=True)
    score_low, _ = scorer(low.flatten(0, 1), low_ycbcr, infer=True)

    diff = (score_high - score_low).abs().view(b, k)
    idx = diff.argmax(dim=1)
    rows = torch.arange(b, device=idx.device)
    return (
        high[rows, idx],
        low[rows, idx],
        high_ycbcr.view_as(high)[rows, idx],
        low_ycbcr.view_as(low)[rows, idx],
    )
