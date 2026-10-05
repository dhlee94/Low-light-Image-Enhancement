"""Evaluate quality features against human scores (KADID-10k by default).

Features are extracted once, then a Ridge regressor is fit and tested on
``--n_splits`` random reference-disjoint 80/20 splits (median SRCC / PLCC); the
Ridge strength is chosen inside each train split by reference-grouped 5-fold CV.

    # trained QualityNetwork (needs the DualColorNetwork it was trained on)
    python scripts/eval_iqa.py --backbone quality --dual_model_path weights/model.pth \\
        --quality_model_path weights/quality_model.pth
    # baselines that need no training
    python scripts/eval_iqa.py --backbone random     # randomly initialized Dual + Quality networks
    python scripts/eval_iqa.py --backbone resnet50   # ImageNet ResNet-50, global-average-pooled

KADID-10k: https://database.mmsp-kn.de/kadid-10k-database.html
    curl -L -o data/kadid10k.zip https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip
    unzip -q data/kadid10k.zip -d data/   # -> data/kadid10k/{dmos.csv,images/}
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from llie.cli import add_dual_model_args, add_quality_model_args
from llie.data.dataset import ImageDataset
from llie.data.transforms import build_transform
from llie.models.dual_color import DualColorNetwork
from llie.models.weights import load_weights
from llie.training.quality_module import build_quality_network
from llie.utils.iqa import DEFAULT_ALPHAS, evaluate_features, fit_regressor, srcc
from llie.utils.seed import seed_everything

BACKBONES = ("quality", "random", "resnet50")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=BACKBONES, default="quality")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=default_device())
    p.add_argument("--label_csv", default="data/kadid10k/dmos.csv", help="CSV with image names, groups and labels")
    p.add_argument("--image_dir", default="data/kadid10k/images", help="directory the image names are relative to")
    p.add_argument("--image_col", default="dist_img")
    p.add_argument("--group_col", default="ref_img",
                   help="reference image of each sample; splits never share a reference")
    p.add_argument("--label_col", default="dmos")
    p.add_argument("--img_size", type=int, default=256,
                   help="square input size for quality/random (must match training); resnet50 uses the original size")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
                   help="Ridge strengths searched per split (one value = fixed)")
    p.add_argument("--dual_model_path", default="./weights/model.pth")
    p.add_argument("--quality_model_path", default="./weights/quality_model.pth")
    p.add_argument("--out_dir", default="./iqa_results", help="features (.npz) and metrics (.json) per backbone")
    p.add_argument("--save_regressor", action="store_true",
                   help="also fit Ridge on ALL samples and pickle it to <out_dir>/<backbone>_ridge.pkl")
    add_dual_model_args(p)
    add_quality_model_args(p)
    return p.parse_args(argv)


def _loader(paths, transform, args):
    return DataLoader(ImageDataset(paths, transform, img_mode="RGB", infer=True),
                      batch_size=args.batch_size, num_workers=args.workers)


@torch.no_grad()
def extract_quality(args, paths, device, pretrained):
    """QualityNetwork features (B, dim) and its raw score head output."""
    dual_model = DualColorNetwork(in_channels=args.in_channels, gp=args.gp, hidden_channels=args.hidden_channels)
    quality_model = build_quality_network(args)
    if pretrained:
        load_weights(dual_model, args.dual_model_path)
        load_weights(quality_model, args.quality_model_path)
    dual_model.to(device).eval()
    quality_model.to(device).eval()

    # Same preprocessing as training: stretched to img_size.
    transform = build_transform((args.img_size, args.img_size), scaleup=True, stretch=True)
    scores, feats = [], []
    for batch in tqdm(_loader(paths, transform, args), desc=args.backbone):
        x = batch["input"].to(device)
        score, feat = quality_model(x, dual_model(x, only=True), infer=True)
        scores.append(score.squeeze(1).cpu())
        feats.append(feat.cpu())
    return torch.cat(feats).numpy(), torch.cat(scores).numpy()


@torch.no_grad()
def extract_resnet50(args, paths, device):
    """2048-d global-average-pooled ImageNet ResNet-50 features at the original resolution."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    feats = []
    for batch in tqdm(_loader(paths, build_transform(None), args), desc="resnet50"):
        x = (batch["input"].to(device) - mean) / std
        feats.append(model(x).cpu())
    return torch.cat(feats).numpy(), None


def extract(args, paths, device):
    if args.backbone == "resnet50":
        return extract_resnet50(args, paths, device)
    return extract_quality(args, paths, device, pretrained=args.backbone == "quality")


def main(args):
    df = pd.read_csv(args.label_csv)
    paths = [os.path.join(args.image_dir, name) for name in df[args.image_col]]
    labels = df[args.label_col].to_numpy(dtype=np.float64)
    if args.group_col in df.columns:
        groups = df[args.group_col].to_numpy()
    else:
        warnings.warn(f"no '{args.group_col}' column: splitting per image, so content can leak across splits")
        groups = np.arange(len(df))

    feats, scores = extract(args, paths, torch.device(args.device))
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f"{args.backbone}_features.npz"),
             features=feats, labels=labels, groups=groups, **({} if scores is None else {"scores": scores}))

    result = evaluate_features(feats, labels, groups, n_splits=args.n_splits, test_size=args.test_size,
                               alphas=args.alphas, seed=args.seed)
    result.update(backbone=args.backbone, n_images=len(df), feature_dim=int(feats.shape[1]))
    if scores is not None:
        # Score head without any regression (self-supervised: sign is arbitrary).
        result["score_head_abs_srcc"] = abs(srcc(scores, labels))

    print(f"[{args.backbone}] {len(df)} images, {feats.shape[1]}-d features, "
          f"{args.n_splits} reference-disjoint splits (median)")
    print(f"  SRCC {result['srcc']:.4f} (std {result['srcc_std']:.4f})   "
          f"PLCC {result['plcc']:.4f} (std {result['plcc_std']:.4f})   "
          f"alpha per split {sorted(set(result['per_split']['alpha']))}")
    if "score_head_abs_srcc" in result:
        print(f"  |SRCC| of the raw score head (all images, no regression): {result['score_head_abs_srcc']:.4f}")
    with open(os.path.join(args.out_dir, f"{args.backbone}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    if args.save_regressor:
        path = os.path.join(args.out_dir, f"{args.backbone}_ridge.pkl")
        with open(path, "wb") as f, np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pickle.dump(fit_regressor(feats, labels, groups, args.alphas), f)
        print(f"  Ridge fit on all {len(df)} samples -> {path}")
    return result


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
