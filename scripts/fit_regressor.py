"""Fit a Ridge regressor from quality-network features to human scores (e.g. KADID DMOS).

    python scripts/fit_regressor.py --label_csv kadid/labels.csv --image_dir kadid/images
"""
import argparse
import os
import pickle

import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

from llie.cli import add_dual_model_args, add_quality_model_args
from llie.training.quality_module import build_quality_network
from llie.data.dataset import ImageDataset
from llie.data.transforms import build_transform
from llie.models.dual_color import DualColorNetwork
from llie.models.weights import load_weights
from llie.utils.seed import seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--label_csv", required=True, help="CSV with image file names and labels")
    p.add_argument("--image_dir", required=True, help="directory the image names are relative to")
    p.add_argument("--image_col", default="image")
    p.add_argument("--label_col", default="dmos")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength")
    p.add_argument("--dual_model_path", default="./weights/model.pth")
    p.add_argument("--quality_model_path", default="./weights/quality_model.pth")
    p.add_argument("--model_save_path", default="./weights/regression.save", help="pickled Ridge model")
    p.add_argument("--out_csv", default="./regression_result.csv")
    add_dual_model_args(p)
    add_quality_model_args(p)
    return p.parse_args(argv)


@torch.no_grad()
def extract_features(args, paths, device):
    dual_model = load_weights(
        DualColorNetwork(in_channels=args.in_channels, gp=args.gp, hidden_channels=args.hidden_channels),
        args.dual_model_path).to(device).eval()
    quality_model = load_weights(build_quality_network(args), args.quality_model_path).to(device).eval()

    # Same preprocessing as before: RGB, stretched to img_size (upscaling allowed).
    transform = build_transform((args.img_size, args.img_size), scaleup=True, stretch=True)
    loader = DataLoader(ImageDataset(paths, transform, img_mode="RGB", infer=True),
                        batch_size=args.batch_size, num_workers=args.workers)
    scores, feats = [], []
    for batch in tqdm(loader):
        x = batch["input"].to(device)
        score, feat = quality_model(x, dual_model(x, only=True), infer=True)
        scores.append(score.squeeze(1).cpu())
        feats.append(feat.cpu())
    return torch.cat(scores).numpy(), torch.cat(feats).numpy()


def main(args):
    labels_df = pd.read_csv(args.label_csv)
    paths = [os.path.join(args.image_dir, name) for name in labels_df[args.image_col]]
    labels = labels_df[args.label_col].to_numpy()

    scores, feats = extract_features(args, paths, torch.device(args.device))
    reg = Ridge(alpha=args.alpha).fit(feats, labels)

    os.makedirs(os.path.dirname(args.model_save_path) or ".", exist_ok=True)
    with open(args.model_save_path, "wb") as f:
        pickle.dump(reg, f)

    pd.DataFrame({
        "image": labels_df[args.image_col],
        "regression": scores.astype(float),           # raw quality-head output
        "predict": reg.predict(feats).astype(float),  # Ridge prediction (fit on the same data)
        "label": labels,
    }).to_csv(args.out_csv, index=False)
    print(f"Ridge model -> {args.model_save_path}, results -> {args.out_csv}")


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
