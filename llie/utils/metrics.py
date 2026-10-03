"""Image quality metrics for (B, 3, H, W) tensors in [0, 1]. All return a scalar batch mean."""
import torch
import torch.nn.functional as F

_PSNR_CAP_DB = 100.0  # reported when the images are identical (MSE == 0)


def gaussian_kernel(size=11, sigma=1.5, device=None, dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)  # (size, size), sums to 1


def calculate_psnr(img1, img2, max_val=1.0):
    """PSNR computed per image, then averaged over the batch."""
    mse = (img1 - img2).pow(2).flatten(1).mean(dim=1)
    psnr = 10 * torch.log10(max_val ** 2 / mse.clamp_min(1e-12))
    return psnr.clamp_max(_PSNR_CAP_DB).mean()


def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    """SSIM with an 11x11 Gaussian window applied to each channel separately."""
    channels = img1.shape[1]
    kernel = gaussian_kernel(window_size, sigma, img1.device, img1.dtype)
    window = kernel.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(x):
        return F.conv2d(x, window, padding=pad, groups=channels)

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = filt(img1), filt(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = filt(img1 * img1) - mu1_sq
    sigma2_sq = filt(img2 * img2) - mu2_sq
    sigma12 = filt(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


# sRGB (D65) -> XYZ, same constants as skimage.color.rgb2lab.
_RGB_TO_XYZ = torch.tensor([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
])
_D65_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    """(B, 3, H, W) sRGB in [0, 1] -> CIE Lab, computed on the input's device."""
    rgb = rgb.clamp(0, 1)
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = _RGB_TO_XYZ.to(rgb.device, rgb.dtype)
    white = _D65_WHITE.to(rgb.device, rgb.dtype)
    xyz = torch.einsum("ij,bjhw->bihw", m, linear) / white[None, :, None, None]
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


def calculate_delta_e(img1, img2):
    """CIE76 colour difference (mean over pixels and batch)."""
    return torch.sqrt((rgb_to_lab(img1) - rgb_to_lab(img2)).pow(2).sum(dim=1)).mean()
