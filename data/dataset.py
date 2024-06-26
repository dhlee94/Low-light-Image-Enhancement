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
    
    def __getitem__(self, idx):
        img = Image.open(self.input_image_data[idx])
        input = np.array(img.convert("RGB"))
        ycbcr = np.array(img.convert("YCbCr"))
        trans_img = self.transform(input)
        if self.infer:
            return trans_img
        return trans_img, self.transform_ori(ycbcr), self.transform_ori(input)
