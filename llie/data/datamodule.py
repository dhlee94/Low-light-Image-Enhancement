import cv2
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from llie import config
from llie.data.dataset import ImageDataset, QualityImageDataset
from llie.data.transforms import build_transform

_DATASETS = {"dual": ImageDataset, "quality": QualityImageDataset}


class DataModule(pl.LightningDataModule):
    """Builds loaders from per-split DataFrames with an ``image`` path column.

    An optional ``input`` column (dual task) gives the paired low-light image of
    each ``image``; without it the input is synthesized by darkening.

    csvs: {"train", "valid", "test"} -> DataFrame. Predict mode reads "test".
    Validation/test degradations are seeded so every evaluation sees the same inputs.
    """

    def __init__(self, csvs, task, img_size=(512, 512), img_mode="RGB", batch_size=4, num_workers=0,
                 seed=0, darken_range=tuple(config.TASKS["dual"]["darken_range"]), degradations=None,
                 interpolation=cv2.INTER_LINEAR):
        super().__init__()
        if task not in _DATASETS:
            raise ValueError(f"unknown task {task!r}")
        self.csvs = csvs
        self.task = task
        self.img_mode = img_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.darken_range = tuple(darken_range)
        self.degradations = degradations
        self.transform = build_transform(img_size, interpolation=interpolation, scaleup=False, stretch=True)

    def build_dataset(self, split):
        csv_key = {"train": "train", "val": "valid", "test": "test", "predict": "test"}[split]
        df = self.csvs[csv_key]
        paths = df["image"].tolist()
        dataset_cls = ImageDataset if split == "predict" else _DATASETS[self.task]
        kwargs = {}
        if dataset_cls is ImageDataset and "input" in df.columns:
            kwargs["input_paths"] = df["input"].tolist()
        if dataset_cls is QualityImageDataset:
            kwargs["degradations"] = self.degradations
        return dataset_cls(
            paths, self.transform, img_mode=self.img_mode, infer=(split == "predict"),
            darken_range=self.darken_range, seed=None if split == "train" else self.seed, **kwargs)

    def _loader(self, split):
        # The contrastive loss needs >= 2 samples, so drop a trailing batch of 1 when training.
        drop_last = split == "train" and self.task == "quality"
        return DataLoader(self.build_dataset(split), batch_size=self.batch_size, shuffle=(split == "train"),
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")

    def predict_dataloader(self):
        return self._loader("predict")
