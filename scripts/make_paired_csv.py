"""Write a split CSV for a paired dataset whose low/high images share file names.

    # LOL (https://huggingface.co/datasets/geekyrakshit/LoL-Dataset), unzipped into data/
    python scripts/make_paired_csv.py data/lol_dataset/our485 csv/lol/train.csv
    python scripts/make_paired_csv.py data/lol_dataset/eval15 csv/lol/valid.csv

Columns: ``image`` (normal-light target), ``input`` (low-light input).
"""
import argparse
import os

import pandas as pd

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="directory containing the low/ and high/ sub-directories")
    p.add_argument("out_csv")
    p.add_argument("--low", default="low", help="sub-directory of low-light inputs")
    p.add_argument("--high", default="high", help="sub-directory of normal-light targets")
    args = p.parse_args()

    low_dir, high_dir = os.path.join(args.root, args.low), os.path.join(args.root, args.high)
    names = sorted(n for n in os.listdir(high_dir) if n.lower().endswith(IMAGE_EXTS))
    missing = [n for n in names if not os.path.exists(os.path.join(low_dir, n))]
    if missing:
        raise SystemExit(f"{len(missing)} targets have no low-light pair, e.g. {missing[:3]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame({
        "image": [os.path.abspath(os.path.join(high_dir, n)) for n in names],
        "input": [os.path.abspath(os.path.join(low_dir, n)) for n in names],
    }).to_csv(args.out_csv, index=False)
    print(f"{len(names)} pairs -> {args.out_csv}")


if __name__ == "__main__":
    main()
