import torch
import random
import cv2
import torchvision.transforms as T
import albumentations as A
import numpy as np
from utils.utils import closest_number

class Compose:
    def __init__(self, transform):
        self.transform = transform if isinstance(transform, list) else [transform]
    def __call__(self, data):
        for transform in self.transform:
            if random.random()<transform.p:
                data = transform(data)
        return data
    
    def add(self, new_transform):
        self.transform.append(new_transform)

class Resize:
    def __init__(self, size=(1024, 1024), interpolation=cv2.INTER_LINEAR, scaleup=True, stratch=False, p=1.0):
        self.size = size
        self.interpolation = interpolation
        self.p = p
        self.scaleup = scaleup
        self.stratch = stratch

    def __call__(self, data):
        shape = data.shape[:2]
        if self.stratch:
            new_size = self.size
        else:
            r = min(self.size[1]/shape[1], self.size[0]/shape[0])
            if not self.scaleup:
                r = min(r, 1.0)
            new_size = closest_number(int(round(shape[1]*r))), closest_number(int(round(shape[0]*r)))
        data = cv2.resize(data, new_size, interpolation=self.interpolation)
        return np.ascontiguousarray(data)


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)            
            ])
        self.p = p
    def __call__(self, data):
        return self.transform(data)

class Albumentations:
    def __init__(self, p=1.0):
        """
        transform
            MedianBlur : MedianBlur applies a median filter to an image, reducing noise by replacing each pixel's value with the median value of its local neighborhood.
            CLAHE : CLAHE enhances image contrast by dividing the image into small blocks and applying histogram equalization to each.
            RandomGamma : RandomGamma randomly adjusts the gamma values of an image to alter its brightness and contrast, enhancing visual perception.
            ImageCompression : ImageCompression reduces the quality of an image by compressing it, simulating the effects of saving in a lossy format like JPEG.
        """
        self.p = p
        self.transform = A.Compose([
            A.OneOf([
            A.Blur(p=1.0),
            A.MedianBlur(p=1.0),
            A.CLAHE(p=1.0),
            A.RandomGamma(p=1.0),
            A.ImageCompression(quality_lower=75, p=1.0)
            ], p=0.5),
            A.RandomBrightnessContrast(p=1.0),            
        ])
    def __call__(self, data):
        data = self.transform(image=data)
        return data["image"]