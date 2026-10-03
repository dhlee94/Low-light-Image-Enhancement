import torch


def rotate_batch(x):
    """Return all four 90-degree rotations of a square batch and their labels.

    (B, C, H, W) -> (4B, C, H, W) ordered [0, 90, 180, 270] degrees, so the first
    B rows are the unrotated input; labels are (4B,) with values 0..3.
    """
    assert x.shape[-1] == x.shape[-2], "rotation prediction needs square images"
    rotated = torch.cat([torch.rot90(x, k, dims=(2, 3)) for k in range(4)], dim=0)
    labels = torch.arange(4, device=x.device).repeat_interleave(x.shape[0])
    return rotated, labels
