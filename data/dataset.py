import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from skimage.util import random_noise
import cv2
import io

class ImageDataset(Dataset):
    def __init__(self, Image_path, transform=None, transform_ori=None, infer=False, types="L"):
        self.input_image_data = Image_path['image']
        self.transform = transform
        self.transform_ori = transform_ori
        self.infer = infer
        self.types = types
    def __len__(self):
        return len(self.input_image_data)
    
    def __getitem__(self, idx):
        img = Image.open(self.input_image_data[idx])
        if self.types=="L":
            gray = img.convert("L")
            rgb = gray.convert("RGB")
        else:
            rgb = img.convert("RGB")
        ycbcr = np.array(rgb.convert("YCbCr"))
        rgb = np.array(rgb)
        trans_img = self.transform(rgb)
        if self.infer:
            return trans_img
        return trans_img, self.transform_ori(ycbcr), self.transform_ori(rgb)

class QualityImageDataset(ImageDataset):    
    def _image_compression(self, img, path, types):
        sigma1 = 40 + np.random.random() * 20 # 40-60
        sigma2 = 80 + np.random.random() * 10 # 80-90
        if types=="L":
            img.save(os.path.join(path + "_high_gray.jpg"), optimize=True, quality=int(sigma1))
            img.save(os.path.join(path + "_low_gray.jpg"), optimize=True, quality=int(sigma2))
            high = Image.open(os.path.join(path + "_high_gray.jpg"))
            low = Image.open(os.path.join(path + "_low_gray.jpg"))
        else:
            img.save(os.path.join(path + "_high.jpg"), optimize=True, quality=int(sigma1))
            img.save(os.path.join(path + "_low.jpg"), optimize=True, quality=int(sigma2))
            high = Image.open(os.path.join(path + "_high.jpg"))
            low = Image.open(os.path.join(path + "_low.jpg"))
        return np.array(high), np.array(low)
    
    def _add_noise(self, path):
        sigma1 = 0.00005+ np.random.random() * 0.000001
        sigma2 = 0.00001+ np.random.random() * 0.000001   #low noise

        origin = cv2.imread(path)
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)

        noise = random_noise(origin, mode='gaussian',var=sigma1)
        high = Image.fromarray((noise * 255).astype('uint8'))

        noise = random_noise(origin, mode='gaussian',var=sigma2)
        low = Image.fromarray((noise * 255).astype('uint8'))
        return high, low
    
    def __getitem__(self, idx):
        img = Image.open(self.input_image_data[idx])
        base_name = self.input_image_data[idx].split('.')[0]
        if self.types=="L":
            input = img.convert("L").convert("RGB")
        else:
            input = img.convert("RGB")

        if self.infer:
            input = np.array(input)
            return self.transform(input)
        high_noise, low_noise = self._add_noise(path=self.input_image_data[idx])
        if self.types=="L":
            high_noise = high_noise.convert("L").convert("RGB")
            low_noise = low_noise.convert("L").convert("RGB")
        else:
            high_noise = high_noise.convert("RGB")
            low_noise = low_noise.convert("RGB")

        high_compression, low_compression = self._image_compression(img=img, path=base_name, types=self.types)

        return self.transform(np.array(input)), self.transform(np.array(high_noise)), self.transform(np.array(low_noise)), self.transform(high_compression), self.transform(low_compression)
