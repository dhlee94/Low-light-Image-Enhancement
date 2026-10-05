# image-enhancement for color

Low-light enhancement network (`DualColorNetwork`) and a self-supervised
quality network (`QualityNetwork`) trained on top of it.

## Setup

```bash
pip install -e ".[dev]"     # pip >= 21.3; or: pip install -r requirements.txt
```

`--csv_path` is a directory with `train.csv`, `valid.csv` (and optionally
`test.csv`, otherwise `valid.csv` is reused); each has an `image` column of paths.

## Usage

```bash
# 1) enhancement network -> weights/model.pth (best val PSNR)
python scripts/train.py --task dual --csv_path ./csv --devices 0

# 2) quality network on the frozen enhancement network
#    -> weights/quality_model.pth, weights/select_quality_model.pth (best val loss)
python scripts/train.py --task quality --csv_path ./csv --dual_model_path ./weights/model.pth --devices 0

# evaluate / predict (loads --model_path)
python scripts/train.py --task dual --mode test    --csv_path ./csv --model_path ./weights/model.pth
python scripts/train.py --task dual --mode predict --csv_path ./csv --model_path ./weights/model.pth   # -> predictions/enhanced/

# 3) evaluate quality features on KADID-10k (see "Quality evaluation" below)
python scripts/eval_iqa.py --backbone quality --dual_model_path ./weights/model.pth --quality_model_path ./weights/quality_model.pth
```

### Paired low-light data (LOL)

By default the dual task synthesizes inputs by darkening. With an `input` column in the
CSVs it trains on real low-light / normal-light pairs instead:

```bash
# 1) download LOL (~330MB; 485 train / 15 eval pairs, 600x400) -> data/lol_dataset/{our485,eval15}/{low,high}
#    mirror of https://daooshee.github.io/BMVC2018website/
mkdir -p data
curl -L -o data/lol_dataset.zip https://huggingface.co/datasets/geekyrakshit/LoL-Dataset/resolve/main/lol_dataset.zip
unzip -q data/lol_dataset.zip -d data/

# 2) split CSVs (image = normal-light target, input = low-light), then train
python scripts/make_paired_csv.py data/lol_dataset/our485 data/csv/lol/train.csv
python scripts/make_paired_csv.py data/lol_dataset/eval15 data/csv/lol/valid.csv
python scripts/train.py --task dual --csv_path data/csv/lol --img_size 0 --model_save_path weights/lol   # 0 = original resolution
```

### Quality evaluation (KADID-10k)

Features are scored with a Ridge regressor over 10 random 80/20 splits that never share a
reference image between train and test; the median test SRCC / PLCC is reported.

```bash
# download KADID-10k (~3GB, 10,125 images) -> data/kadid10k/{dmos.csv,images/}
curl -L -o data/kadid10k.zip https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip
unzip -q data/kadid10k.zip -d data/

python scripts/eval_iqa.py --backbone random     # baseline: untrained Dual + Quality networks
python scripts/eval_iqa.py --backbone resnet50   # baseline: ImageNet ResNet-50 features
python scripts/eval_iqa.py --backbone quality --dual_model_path ... --quality_model_path ...
# -> iqa_results/<backbone>_{metrics.json,features.npz}; --save_regressor also pickles a Ridge fit on all images
```

Baselines (KADID-10k, median of 10 splits; Ridge alpha chosen by grouped CV, 1e5 in every split):

| backbone | features | SRCC | PLCC |
|---|---|---|---|
| `random` (untrained Dual + Quality) | 1024 | 0.077 | 0.107 |
| `resnet50` (ImageNet) | 2048 | 0.495 | 0.525 |

`python scripts/train.py --task <dual|quality> --help` lists every option with the task's defaults.

## Layout

```
llie/
  cli.py          command-line arguments and per-task defaults
  models/         DualColorNetwork (dual_color.py, parts.py), QualityNetwork (quality.py), weight I/O
  data/           datasets, DataModule, synthetic degradations, transforms
  training/       LightningModules, losses, schedulers, quality-task helpers (pairing, rotation)
  utils/          metrics (PSNR / SSIM / ΔE), IQA evaluation (SRCC / PLCC), image saving, seeding
scripts/          train.py, eval_iqa.py, make_paired_csv.py
tests/
```

## Tests

```bash
pytest              # unit tests + a tiny CPU end-to-end run (~5 s)
```
