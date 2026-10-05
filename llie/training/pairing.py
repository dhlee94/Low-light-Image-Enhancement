import torch

PAIR_SELECTIONS = ("max_gap", "min_margin", "random")


@torch.no_grad()
def select_pair(color_model, scorer, highs, lows, mode="max_gap", generator=None):
    """Pick, per sample, which degradation type's (strong, weak) pair to train on.

    Args:
        color_model: DualColorNetwork, used to compute the YCbCr branch input.
        scorer: QualityNetwork used for selection (the EMA copy).
        highs, lows: lists of K tensors (B, C, H, W); ``highs[k]`` is the strongly
            and ``lows[k]`` the weakly degraded version for degradation type k.
        mode: "max_gap"    - largest |score_high - score_low|: the pair the scorer
                             already separates most (the original behaviour);
              "min_margin" - smallest score_high - score_low: the hardest pair,
                             including pairs the scorer ranks the wrong way;
              "random"     - a uniformly random type (the scorer is not used).
        generator: CPU ``torch.Generator`` for "random" (seeded evaluation).

    Returns:
        (high, low, high_ycbcr, low_ycbcr, index): four (B, C, H, W) tensors and the
        chosen degradation type per sample, (B,).
    """
    if mode not in PAIR_SELECTIONS:
        raise ValueError(f"unknown pair selection {mode!r}; expected one of {PAIR_SELECTIONS}")
    high = torch.stack(highs, dim=1)  # (B, K, C, H, W)
    low = torch.stack(lows, dim=1)
    b, k = high.shape[:2]
    rows = torch.arange(b, device=high.device)

    if mode == "random":
        idx = torch.randint(k, (b,), generator=generator).to(high.device)
        high, low = high[rows, idx], low[rows, idx]
        return high, low, color_model(high, only=True), color_model(low, only=True), idx

    high_ycbcr = color_model(high.flatten(0, 1), only=True)
    low_ycbcr = color_model(low.flatten(0, 1), only=True)
    score_high, _ = scorer(high.flatten(0, 1), high_ycbcr, infer=True)
    score_low, _ = scorer(low.flatten(0, 1), low_ycbcr, infer=True)

    margin = (score_high - score_low).view(b, k)  # the rank loss wants this > 0
    idx = margin.abs().argmax(dim=1) if mode == "max_gap" else margin.argmin(dim=1)
    return (
        high[rows, idx],
        low[rows, idx],
        high_ycbcr.view_as(high)[rows, idx],
        low_ycbcr.view_as(low)[rows, idx],
        idx,
    )
