import os
import time

import torch
import torch.nn as nn

from llie.training.base_module import BaseEnhancementModule
from llie.training.losses import color_loss, total_variation_loss
from llie.models.dual_color import DualColorNetwork
from llie.models.weights import load_weights
from llie.utils.image_io import save_comparison, save_images
from llie.utils.metrics import calculate_delta_e, calculate_psnr, calculate_ssim


class DualColorModule(BaseEnhancementModule):
    """Trains DualColorNetwork to restore darkened images."""

    WEIGHT_FILES = {"model": "model.pth"}
    MONITOR = ("psnr", "max")
    LOSS_NAMES = ("ycbcr", "rgb", "tv", "color")

    def __init__(self, args):
        super().__init__(args)
        self.model = DualColorNetwork(in_channels=args.in_channels, gp=args.gp, hidden_channels=args.hidden_channels)
        if args.pretrain or args.mode != "train":
            load_weights(self.model, args.model_path)
        self.l1 = nn.L1Loss()
        self.loss_weights = dict(zip(self.LOSS_NAMES, args.loss_weights))

    def forward(self, x):
        return self.model(x, infer=True)

    def _compute_losses(self, ycbcr, rgb, target_ycbcr, target_rgb):
        return {
            "ycbcr": self.l1(ycbcr, target_ycbcr),
            "rgb": self.l1(rgb, target_rgb),
            "tv": total_variation_loss(rgb),
            "color": color_loss(rgb, target_rgb),
        }

    def training_step(self, batch, batch_idx):
        ycbcr, rgb = self.model(batch["input"])
        losses = self._compute_losses(ycbcr, rgb, batch["target_ycbcr"], batch["target_rgb"])
        total = sum(self.loss_weights[name] * value for name, value in losses.items())
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, batch_size=self._batch_size(batch))
        self.log("train/loss", total, prog_bar=True, batch_size=self._batch_size(batch))
        return total

    def shared_eval_step(self, batch, batch_idx, stage):
        x, target = batch["input"], batch["target_rgb"]
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        rgb = self.model(x, infer=True)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        rgb = rgb.clamp(0, 1)  # metrics and saved images use the displayable range
        if stage == "val" and batch_idx < self.args.save_n_images and not self.trainer.sanity_checking:
            save_comparison(x, rgb, target, self.args.img_save_path, prefix=str(batch_idx))
        return {
            "delta_e": calculate_delta_e(rgb, target),
            "psnr": calculate_psnr(rgb, target),
            "ssim": calculate_ssim(rgb, target),
            "time": elapsed,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        rgb = self.model(batch["input"], infer=True).clamp(0, 1)
        save_images(rgb, batch["path"], os.path.join(self.args.pred_save_path, "enhanced"))
        return {"path": batch["path"]}
