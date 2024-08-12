import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color
from PIL import Image
import torchvision.transforms as T

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
    device = img1.device
    window = gaussian_filter(size=11, sigma=1.5).reshape(1, 1, 11, 11).expand(1, 3, 11, 11).to(device)
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

def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
    return x.flip(1).flip(2)

def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		if label == 1:
			img = tensor_rot_90(img)
		elif label == 2:
			img = tensor_rot_180(img)
		elif label == 3:
			img = tensor_rot_270(img)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='expand'):
    device = batch.device
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3])
        batch = batch.repeat((4,1,1,1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels.to(device)

def calculate_quality_diff(model, quality_model, input,
                           input_low, input_high,  noise_high, noise_low, 
                           blur, device):
    high = []
    low = []
    B, C, H, W = input.shape
    sigma1 = 40 + np.random.random() * 20
    sigma2 = 5 + np.random.random() * 15
    with torch.no_grad():
        quality_model.eval()
        if blur:
            blur_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(input).to(device)
            blur_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(input).to(device)
            high.append(blur_high)
            low.append(blur_low)
        if input_low is not None and input_high is not None:
            high.append(input_high)
            low.append(input_low)
        if noise_high is not None and noise_low is not None:
            high.append(noise_high)
            low.append(noise_low)
        length = len(high)
        high = torch.cat(high, dim=0)
        low  = torch.cat(low, dim=0)
        high_ycbcr = model(high, only=True)
        low_ycbcr = model(low, only=True)
        quality_high, _ = quality_model(high, high_ycbcr, infer=True)
        quality_low, _ = quality_model(low, low_ycbcr, infer=True)
        high = high.reshape(-1, B, C, H, W).permute(1, 0, 2, 3, 4)
        low = low.reshape(-1, B, C, H, W).permute(1, 0, 2, 3, 4)
        high_ycbcr = high_ycbcr.reshape(-1, B, C, H, W).permute(1, 0, 2, 3, 4)
        low_ycbcr = low_ycbcr.reshape(-1, B, C, H, W).permute(1, 0, 2, 3, 4)
        diff = torch.abs(quality_high - quality_low).reshape(length, B, -1).sum(dim=-1).permute(1, 0)
        index = [torch.argmax(data).item() for data in diff]
        high_input = torch.stack([high[idx, index[idx], ...] for idx in range(B)] , dim=0)
        low_input = torch.stack([low[idx, index[idx], ...] for idx in range(B)], dim=0)
        high_ycbcr = torch.stack([high_ycbcr[idx, index[idx], ...] for idx in range(B)] , dim=0)
        low_ycbcr = torch.stack([low_ycbcr[idx, index[idx], ...] for idx in range(B)], dim=0)
    return high_input.detach(), low_input.detach(), high_ycbcr.detach(), low_ycbcr.detach()