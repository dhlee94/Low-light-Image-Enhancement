import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class ImageDataset(Dataset):
    def __init__(self, Image_path, transform=None, transform_ori=None, infer=False):
        self.input_image_data = Image_path['image']
        self.transform = transform
        self.transform_ori = transform_ori
        self.infer = infer
        
    def __len__(self):
        return len(self.input_image_data)

    def _convert(self, img):
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128 / 255.0
        Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128 / 255.0
        
        ycbcr = np.stack((Y, Cb, Cr), axis=-1)
        return ycbcr
    
    def __getitem__(self, idx):
        input = Image.open(self.input_image_data[idx]).convert("RGB")
        input = np.array(input)
        trans_img = self.transform(input)
        if self.infer:
            return trans_img
        return trans_img, self.transform_ori(self._convert(input)), self.transform_ori(input)
