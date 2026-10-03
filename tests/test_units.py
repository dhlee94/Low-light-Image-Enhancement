"""Regression tests for the bugs fixed in the refactor (one test per bug where possible)."""
import os

import numpy as np
import pytest
import torch
from PIL import Image

from llie.training.losses import GroupContrastiveLoss
from llie.data.dataset import ImageDataset, QualityImageDataset
from llie.data.transforms import build_transform
from llie.models.dual_color import DualColorNetwork
from llie.models.parts import RGB2YCbCr
from llie.models.quality import QualityNetwork
from llie.utils.metrics import calculate_delta_e, calculate_psnr, calculate_ssim, rgb_to_lab
from llie.training.pairing import select_hardest_pair
from llie.training.rotation import rotate_batch


def _random_images(b=2, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, size, size, generator=g)


# ---- DualColorNetwork init --------------------------------------------------------
def test_dual_color_network_starts_as_identity():
    # Without the CPM residual + zero-init fc2 the initial output ranged about -24..10.
    x = _random_images()
    ycbcr, rgb = DualColorNetwork()(x)
    assert torch.allclose(ycbcr, RGB2YCbCr()(x), atol=1e-6)
    assert torch.allclose(rgb, x, atol=1e-2)  # CSC matrices are rounded, not exact inverses


def test_dual_color_network_matches_paper_size():
    # Paper reports 92,882 parameters; the extra 64->64 hidden conv per CPM made it 121,170.
    assert sum(p.numel() for p in DualColorNetwork().parameters()) == 96210


# ---- paired (real low-light) data ---------------------------------------------------
def test_paired_dataset_uses_real_low_light_input(tmp_path):
    rng = np.random.default_rng(0)
    low, high = tmp_path / "low.png", tmp_path / "high.png"
    Image.fromarray(rng.integers(0, 60, (40, 48, 3), dtype=np.uint8)).save(low)
    Image.fromarray(rng.integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(high)
    to_tensor = build_transform(None)  # original resolution

    item = ImageDataset([str(high)], to_tensor, input_paths=[str(low)])[0]
    torch.testing.assert_close(item["input"], to_tensor(np.array(Image.open(low))))
    torch.testing.assert_close(item["target_rgb"], to_tensor(np.array(Image.open(high))))
    assert item["input"].shape == (3, 40, 48)

    pred = ImageDataset([str(high)], to_tensor, input_paths=[str(low)], infer=True)[0]
    assert pred["path"] == str(low)


# ---- metrics (B3) -------------------------------------------------------------
def test_ssim_matches_skimage():
    skmetrics = pytest.importorskip("skimage.metrics")
    x = _random_images(b=1, size=96)
    assert calculate_ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)
    noisy = (x + 0.2 * torch.randn(x.shape, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    expected = skmetrics.structural_similarity(
        x[0].permute(1, 2, 0).numpy(), noisy[0].permute(1, 2, 0).numpy(), channel_axis=2, data_range=1,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    # Small gap = border handling (zero padding here, cropping in skimage).
    # The old implementation (window / 121, channels summed) was ~0.13 too high.
    assert calculate_ssim(x, noisy).item() == pytest.approx(expected, abs=0.02)


def test_psnr_identical_is_finite():
    x = _random_images()
    assert calculate_psnr(x, x).item() == pytest.approx(100.0)


def test_lab_matches_skimage():
    skcolor = pytest.importorskip("skimage.color")
    x = _random_images(b=1)
    expected = skcolor.rgb2lab(x[0].permute(1, 2, 0).numpy())
    got = rgb_to_lab(x)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-3)
    assert calculate_delta_e(x, x).item() == pytest.approx(0.0, abs=1e-4)


# ---- attention axis (B1) ------------------------------------------------------
def test_quality_network_samples_are_independent():
    torch.manual_seed(0)
    net = QualityNetwork(image_size=32, patch_size=8, dim=32, encoder_depth=1, decoder_depth=1, heads=4,
                         channels=3, drop_out=0.0, emb_dropout=0.0).eval()
    x, y = _random_images(b=4), _random_images(b=4, seed=1)
    with torch.no_grad():
        batched, _ = net(x, y, infer=True)
        single, _ = net(x[:1], y[:1], infer=True)
    # With attention over the batch axis, sample 0's score depended on samples 1..3.
    torch.testing.assert_close(batched[:1], single, rtol=1e-4, atol=1e-5)


# ---- helpers ------------------------------------------------------------------
def test_rotate_batch_matches_legacy_rotation():
    x = _random_images(b=3)
    rotated, labels = rotate_batch(x)
    legacy = [lambda t: t, lambda t: t.flip(2).transpose(1, 2), lambda t: t.flip(1).flip(2),
              lambda t: t.transpose(1, 2).flip(2)]  # former tensor_rot_* on (C, H, W)
    for k in range(4):
        for i in range(3):
            torch.testing.assert_close(rotated[k * 3 + i], legacy[k](x[i]))
    assert labels.tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3


def test_select_hardest_pair_picks_largest_score_gap():
    b = 2
    highs = [torch.full((b, 3, 4, 4), v) for v in (0.1, 0.9, 0.5)]
    lows = [torch.zeros(b, 3, 4, 4)] * 3

    def color_model(x, only=True):
        return x

    def scorer(x, ycbcr, infer=True):
        return x.mean(dim=(1, 2, 3)).unsqueeze(1), None

    high, low, high_y, _ = select_hardest_pair(color_model, scorer, highs, lows)
    assert torch.allclose(high, highs[1]) and torch.allclose(high_y, highs[1])


def test_contrastive_loss_batch_of_one_is_finite():
    loss_fn = GroupContrastiveLoss(temperature=0.5)
    with pytest.warns(UserWarning):
        loss = loss_fn(torch.randn(1, 8, requires_grad=True), torch.randn(1, 8))
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_fn(torch.randn(4, 8), torch.randn(4, 8)))


# ---- quality dataset (B2, B5, B6, P1) -------------------------------------------
@pytest.fixture
def dotted_image_dir(tmp_path):
    # A directory name containing "." broke the old path.split('.') logic.
    d = tmp_path / "data.v1"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(d / f"img.{i}.png")
    return d


def test_quality_dataset_items(dotted_image_dir):
    before = sorted(os.listdir(dotted_image_dir))
    paths = [str(dotted_image_dir / name) for name in before]
    ds = QualityImageDataset(paths, build_transform((32, 32)), img_mode="L", seed=0)
    item = ds[0]

    assert set(item) == {"input", "jpeg_high", "jpeg_low", "noise_high", "noise_low"}
    for value in item.values():
        assert value.shape == (3, 32, 32)
        # img_mode="L": every channel identical, including the JPEG pair.
        assert torch.equal(value[0], value[1]) and torch.equal(value[1], value[2])
    # "high" means more degraded.
    err = {k: (item[k] - item["input"]).abs().mean() for k in ("jpeg_high", "jpeg_low", "noise_high", "noise_low")}
    assert err["jpeg_high"] > err["jpeg_low"]
    assert err["noise_high"] > err["noise_low"]
    # Degradations are seeded per index, and nothing is written to disk.
    torch.testing.assert_close(ds[0]["jpeg_high"], item["jpeg_high"])
    assert sorted(os.listdir(dotted_image_dir)) == before
