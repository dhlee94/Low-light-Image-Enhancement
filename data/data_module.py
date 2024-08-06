import pytorch_lightning as pl
from torch.utils.data import DataLoader
import cv2
import utils.augments as A
from data.dataset import ImageDataset, QualityImageDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, Image_path, img_size=(512, 512), batch_size=4, num_workers=0, interpolation=cv2.INTER_LINEAR, stratch=True, scaleup=False, img_types="L", types="retouching"):
        super().__init__()
        self.Image_path = Image_path
        self.img_size = img_size 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.interpolation = interpolation
        self.stratch = stratch
        self.scaleup = scaleup
        self.types = types
        self.img_types = img_types
    def _image_transform(self):
        return A.Compose([
            A.Resize(size=self.img_size, interpolation=self.interpolation, scaleup=self.scaleup, stratch=self.stratch),
            A.Normalize()])

    def _train_transform(self):
        return A.Compose([
                        A.Albumentations(p=1.0),
                        A.Resize(size=self.img_size, interpolation=self.interpolation, scaleup=self.scaleup, stratch=self.stratch),
                        A.Normalize()]) if self.types=="retouching" else A.Compose([
                                                                         A.Resize(size=self.img_size, interpolation=self.interpolation, scaleup=self.scaleup, stratch=self.stratch),
                                                                         A.Normalize()])

    def train_dataloader(self):
        train_dataset = ImageDataset(Image_path=self.Image_path['train'],
                                    transform=self._train_transform(), transform_ori=self._image_transform(), types=self.img_types) if self.types=="retouching" else QualityImageDataset(Image_path=self.Image_path['train'],
                                    transform=self._train_transform(), types=self.img_types)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        valid_dataset = ImageDataset(Image_path=self.Image_path['valid'], 
                                    transform=self._train_transform(), transform_ori=self._image_transform(), types=self.img_types) if self.types=="retouching" else QualityImageDataset(Image_path=self.Image_path['train'],
                                    transform=self._train_transform(), types=self.img_types)
        return DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test_dataset = ImageDataset(Image_path=self.Image_path['valid'],
                                    transform=self._train_transform(), transform_ori=self._image_transform(), types=self.img_types) if self.types=="retouching" else QualityImageDataset(Image_path=self.Image_path['train'],
                                    transform=self._train_transform(), types=self.img_types)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def predict_dataloader(self):
        test_dataset = ImageDataset(Image_path=self.Image_path['test'], transform=self._image_transform(), infer=True, types=self.img_types) if self.types=="retouching" else QualityImageDataset(Image_path=self.Image_path['train'],
                                    transform=self._train_transform(), types=self.img_types)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
