import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from PIL import Image

def seed_everything(seed: int = 304):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def gaussian_filter(size, sigma=1.5):
    x, y = torch.meshgrid([torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32)])
    x = x - size // 2
    y = y - size // 2
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 1.0 
    psnr = 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    device = img1.get_device()
    window = gaussian_filter(size=11, sigma=1.5).reshape(1, 1, 11, 11).expand(1, 3, 11, 11).to(torch.device(f"cuda:{device}"))
    mu1 = F.conv2d(img1, window / 121, padding=5)
    mu2 = F.conv2d(img2, window / 121, padding=5)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window / 121, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window / 121, padding=5) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window / 121, padding=5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def rgb_to_lab(image):
    image = image.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (N, H, W, C)
    lab_image = color.rgb2lab(image)
    return torch.from_numpy(lab_image).permute(0, 3, 1, 2)  # Convert back to (N, C, H, W)

def calculate_delta_e(img1, img2):
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    delta_e = torch.sqrt(torch.sum((lab1 - lab2) ** 2, dim=1))
    return delta_e.mean()

def closest_number(num):
    closest_int = round(num / 64)
    closest_power = closest_int * 64
    return closest_power

def image_saving(images, retouchings, ground_truths, path, index):
    for idx, (image, retouching, ground_truth) in enumerate(zip(images, retouchings, ground_truths)):
        total_width = image.shape[1] + retouching.shape[1] + ground_truth.shape[1]
        combined_image = Image.new("RGB", (total_width, image.shape[0]))
        images = [image, retouching, ground_truth]
        x_offset = 0
        for image in images:
            img = Image.fromarray(np.uint8(image))
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width
        combined_image.save(os.path.join(path, f"{index}_{idx}.png"), "PNG")

