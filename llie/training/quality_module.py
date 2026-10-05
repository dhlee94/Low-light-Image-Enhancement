import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from llie import config
from llie.data.dataset import ImageDataset
from llie.data.transforms import build_transform

from llie.training.base_module import BaseEnhancementModule
from llie.training.losses import GroupContrastiveLoss, SupervisedContrastiveLoss
from llie.data.degradations import random_gaussian_blur
from llie.models.dual_color import DualColorNetwork
from llie.models.quality import QualityNetwork
from llie.models.weights import load_weights
from llie.training.pairing import select_pair
from llie.training.rotation import rotate_batch
from llie.utils.iqa import evaluate_features, load_iqa_table, validation_references


def build_quality_network(args):
    return QualityNetwork(image_size=args.img_size, patch_size=args.patch_size, dim=args.dim,
                          encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth, heads=args.heads,
                          channels=args.in_channels, drop_out=args.drop_out, emb_dropout=args.emb_dropout,
                          separate_ycbcr_embedding=args.separate_ycbcr_embedding)


class QualityModule(BaseEnhancementModule):
    """Self-supervised quality network.

    Per batch: build strong/weak degradation pairs (blur, JPEG, noise), keep one
    type per sample (``--pair_selection``; originally the pair the EMA model
    separates most), and train on
      - contrastive: weak vs strong feature groups (``--contrastive``)
      - rank:        strong pair should score higher than weak pair
      - rotation:    predict 0/90/180/270 rotation of the clean input
      - distance:    strong pair features farther from the clean input than weak ones

    The best weights are picked by the validation loss, or with ``--monitor srcc``
    by the KADID SRCC of a Ridge probe on the held-out validation references
    (never used by ``scripts/eval_iqa.py``).
    """

    WEIGHT_FILES = {"quality_model": "quality_model.pth", "ema_model": "select_quality_model.pth"}
    MONITOR = ("loss", "min")
    LOSS_NAMES = ("contrastive", "rank", "rotate", "distance")

    def __init__(self, args):
        super().__init__(args)
        # Frozen enhancement network: provides the YCbCr input of the quality network.
        self.dual_model = DualColorNetwork(in_channels=args.in_channels, gp=args.gp, hidden_channels=args.hidden_channels)
        if args.dual_pretrain:
            load_weights(self.dual_model, args.dual_model_path)
        else:
            warnings.warn("DualColorNetwork is randomly initialized (--no_dual_pretrain); use only for debugging.")
        self.dual_model.requires_grad_(False)

        # Trained network and its EMA copy (used only to pick the hardest pair).
        self.quality_model = build_quality_network(args)
        self.ema_model = build_quality_network(args)
        if args.pretrain or args.mode != "train":
            load_weights(self.quality_model, args.model_path)
        self.ema_model.load_state_dict(self.quality_model.state_dict())
        self.ema_model.requires_grad_(False)

        if args.contrastive == "group":
            self.contrastive = GroupContrastiveLoss(temperature=args.temperature)
        elif args.contrastive == "type_severity":
            self.contrastive = SupervisedContrastiveLoss(temperature=args.temperature)
        else:
            self.contrastive = None
        self.cross_entropy = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()  # same as BCELoss(sigmoid(x)) but numerically stable
        self.loss_weights = dict(zip(self.LOSS_NAMES, args.loss_weights))
        if args.monitor == "srcc":
            self.MONITOR = ("srcc", "max")
        self._iqa_val = None  # (loader, labels, groups), built on first use

    def train(self, mode=True):
        # Lightning toggles train/eval on the whole module; frozen parts stay in eval.
        super().train(mode)
        self.dual_model.eval()
        self.ema_model.eval()
        return self

    def trainable_parameters(self):
        return self.quality_model.parameters()

    def forward(self, x):
        """Inference: (score (B, 1), feature (B, dim))."""
        with torch.no_grad():
            ycbcr = self.dual_model(x, only=True)
        return self.quality_model(x, ycbcr, infer=True)

    # ---- training pipeline ----------------------------------------------------
    def _make_pairs(self, batch, generator=None):
        x = batch["input"]
        highs = [random_gaussian_blur(x, self.args.blur_sigma_high, generator), batch["jpeg_high"], batch["noise_high"]]
        lows = [random_gaussian_blur(x, self.args.blur_sigma_low, generator), batch["jpeg_low"], batch["noise_low"]]
        return select_pair(self.dual_model, self.ema_model, highs, lows,
                           mode=self.args.pair_selection, generator=generator)

    def _compute_losses(self, batch, generator=None):
        x = batch["input"]
        b = x.shape[0]
        high, low, high_ycbcr, low_ycbcr, pair_type = self._make_pairs(batch, generator)
        rotated, rot_labels = rotate_batch(x)  # rotated[:b] is x itself
        with torch.no_grad():
            rotated_ycbcr = self.dual_model(rotated, only=True)

        # One encoder pass over [rotations (incl. clean input), strong, weak].
        features = self.quality_model.encode(torch.cat([rotated, high, low]),
                                             torch.cat([rotated_ycbcr, high_ycbcr, low_ycbcr]))
        feat_rot, feat_high, feat_low = features.split([4 * b, b, b])
        feat = feat_rot[:b]
        score, score_high, score_low = self.quality_model.score(torch.cat([feat, feat_high, feat_low])).split(b)
        rot_logits = self.quality_model.classify_rotation(feat_rot)

        dist_high = F.pairwise_distance(feat_high, feat)
        dist_low = F.pairwise_distance(feat_low, feat)
        # (score_high - score) - (score_low - score) == score_high - score_low
        return {
            "contrastive": self._contrastive_loss(feat_low, feat_high, pair_type),
            "rank": self.bce(score_high - score_low, torch.ones_like(score)),
            "rotate": self.cross_entropy(rot_logits, rot_labels),
            "distance": self.bce(dist_high - dist_low, torch.ones_like(dist_high)),
        }

    def _contrastive_loss(self, feat_low, feat_high, pair_type):
        if self.args.contrastive == "group":
            return self.contrastive(feat_low, feat_high)
        if self.args.contrastive == "type_severity":
            # label = 2 * degradation type + severity (0 weak, 1 strong)
            labels = torch.cat([2 * pair_type, 2 * pair_type + 1])
            return self.contrastive(torch.cat([feat_low, feat_high]), labels)
        return feat_low.sum() * 0.0  # disabled; keeps the logged value and the graph

    def _total(self, losses):
        return sum(self.loss_weights[name] * value for name, value in losses.items())

    def training_step(self, batch, batch_idx):
        losses = self._compute_losses(batch)
        total = self._total(losses)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, batch_size=self._batch_size(batch))
        self.log("train/loss", total, prog_bar=True, batch_size=self._batch_size(batch))
        return total

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA after every optimizer step: ema = gamma * ema + (1 - gamma) * current
        gamma = self.args.ema_gamma
        for ema, current in zip(self.ema_model.parameters(), self.quality_model.parameters()):
            ema.mul_(gamma).add_(current.detach(), alpha=1 - gamma)

    def shared_eval_step(self, batch, batch_idx, stage):
        # Seeded blur so every evaluation uses the same degradations.
        generator = torch.Generator().manual_seed(self.args.seed + batch_idx)
        losses = self._compute_losses(batch, generator)
        losses["loss"] = self._total(losses)
        return losses

    # ---- SRCC validation (--monitor srcc) ----------------------------------------
    def _iqa_val_data(self):
        if self._iqa_val is None:
            iqa = config.IQA_DATA
            paths, labels, groups = load_iqa_table(self.args.iqa_label_csv, self.args.iqa_image_dir,
                                                   iqa["image_col"], iqa["group_col"], iqa["label_col"])
            keep = np.isin(groups, sorted(validation_references(groups)))
            # Same preprocessing as scripts/eval_iqa.py.
            transform = build_transform((self.args.img_size, self.args.img_size), scaleup=True, stretch=True)
            dataset = ImageDataset([p for p, k in zip(paths, keep) if k], transform, img_mode="RGB", infer=True)
            loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers)
            self._iqa_val = (loader, labels[keep], groups[keep])
        return self._iqa_val

    @torch.no_grad()
    def iqa_validation(self):
        """SRCC / PLCC of a Ridge probe on the quality features of the validation references."""
        loader, labels, groups = self._iqa_val_data()
        feats = []
        for batch in loader:
            x = batch["input"].to(self.device)
            feats.append(self.quality_model.encode(x, self.dual_model(x, only=True)).float().cpu())
        result = evaluate_features(torch.cat(feats).numpy(), labels, groups,
                                   n_splits=self.args.iqa_val_n_splits, seed=self.args.seed)
        return {"srcc": result["srcc"], "plcc": result["plcc"]}

    def on_validation_epoch_end(self):
        if self.args.monitor == "srcc" and not self.trainer.sanity_checking:
            self.val_metrics.update(**self.iqa_validation())
        super().on_validation_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        score, _ = self(batch["input"])
        return {"path": list(batch["path"]), "score": score.squeeze(1).cpu().tolist()}

    def write_predictions(self, outputs):
        rows = [(p, s) for out in outputs for p, s in zip(out["path"], out["score"])]
        os.makedirs(self.args.pred_save_path, exist_ok=True)
        out_path = os.path.join(self.args.pred_save_path, "quality_scores.csv")
        pd.DataFrame(rows, columns=["image", "score"]).to_csv(out_path, index=False)
        return out_path
