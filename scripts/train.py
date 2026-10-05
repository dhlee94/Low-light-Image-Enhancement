"""Single entry point for both tasks.

    python scripts/train.py --task dual    --csv_path ./csv --mode train
    python scripts/train.py --task quality --csv_path ./csv --dual_model_path ./weights/model.pth
"""
import os
import sys
import time

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from llie import config
from llie.cli import parse_args
from llie.training.dual_module import DualColorModule
from llie.training.quality_module import QualityModule
from llie.data.datamodule import DataModule
from llie.utils.seed import seed_everything

MODULES = {"dual": DualColorModule, "quality": QualityModule}


def load_split_csvs(csv_path):
    """train.csv and valid.csv are required; test.csv falls back to valid.csv."""
    def read(name):
        return pd.read_csv(os.path.join(csv_path, name))

    valid = read("valid.csv")
    test_file = os.path.join(csv_path, "test.csv")
    return {"train": read("train.csv"), "valid": valid, "test": pd.read_csv(test_file) if os.path.exists(test_file) else valid}


def build_trainer(args):
    return pl.Trainer(
        max_epochs=args.epoch,
        accelerator=args.accelerator,
        devices=args.devices if args.devices else "auto",
        check_val_every_n_epoch=args.check_val,
        log_every_n_steps=args.check_loss,
        fast_dev_run=args.fast_dev_run,
        logger=CSVLogger(args.log_path, name=args.task),
        enable_checkpointing=False,  # best weights are saved by the module itself
    )


def timed(label, fn, *a, **kw):
    start = time.time()
    result = fn(*a, **kw)
    print(f"{label} time : {time.time() - start:.1f}s")
    return result


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    seed_everything(args.seed)

    datamodule = DataModule(load_split_csvs(args.csv_path), task=args.task, img_size=args.img_shape,
                            img_mode=args.img_mode, batch_size=args.batch_size, num_workers=args.workers,
                            seed=args.seed, darken_range=getattr(args, "darken_range", config.TASKS["dual"]["darken_range"]),
                            degradations={k: getattr(args, k) for k in config.DEGRADATION_PRESETS["original"]
                                          if hasattr(args, k)})
    module = MODULES[args.task](args)
    trainer = build_trainer(args)

    if args.mode == "train":
        timed("training", trainer.fit, module, datamodule)
        module.restore_best_weights()  # evaluate the best checkpoint, not the last epoch
        timed("test", trainer.test, module, datamodule)
    elif args.mode == "test":
        timed("test", trainer.test, module, datamodule)
    else:
        outputs = timed("predict", trainer.predict, module, datamodule)
        if hasattr(module, "write_predictions"):
            print(f"predictions written to {module.write_predictions(outputs)}")
        else:
            print(f"enhanced images written to {os.path.join(args.pred_save_path, 'enhanced')}")


if __name__ == "__main__":
    main()
