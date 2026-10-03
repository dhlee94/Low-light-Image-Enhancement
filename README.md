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

# 3) Ridge regressor from quality features to human scores
python scripts/fit_regressor.py --label_csv kadid/labels.csv --image_dir kadid/images
```

`python scripts/train.py --task <dual|quality> --help` lists every option with the task's defaults.

## Layout

```
llie/
  cli.py          command-line arguments and per-task defaults
  models/         DualColorNetwork (dual_color.py, parts.py), QualityNetwork (quality.py), weight I/O
  data/           datasets, DataModule, synthetic degradations, transforms
  training/       LightningModules, losses, schedulers, quality-task helpers (pairing, rotation)
  utils/          metrics (PSNR / SSIM / ΔE), image saving, seeding
scripts/          train.py, fit_regressor.py
tests/
```

## Tests

```bash
pytest              # unit tests + a tiny CPU end-to-end run (~5 s)
```
