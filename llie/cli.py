"""Command-line arguments shared by every entry point.

All arguments are defined once here. ``scripts/train.py`` and ``scripts/fit_regressor.py`` build
their parsers from these groups, and the per-task differences live in
``TASK_DEFAULTS`` instead of being copy-pasted into separate scripts.
"""
from __future__ import annotations

import argparse
from typing import Sequence

TASKS = ("dual", "quality")
MODES = ("train", "test", "predict")

# Defaults that differ between the two training tasks (same values as the
# pre-refactor main.py / quality_main.py, so existing experiments are unchanged).
TASK_DEFAULTS = {
    "dual": dict(
        batch_size=1,
        img_size=512,
        loss_weights=[0.01, 1.0, 0.01, 0.1],
        optim="AdamW",
        lr=5e-5,
        scheduler="CosineWarmUp",
        eta_scheduler=5e-5,
        model_path="./weights/model.pth",
    ),
    "quality": dict(
        batch_size=8,
        img_size=256,
        loss_weights=[1.0, 1.0, 1.0, 1.0],
        optim="SGD",
        lr=1.25e-3,
        scheduler="LambdaLR",
        eta_scheduler=1.25e-4,
        model_path="./weights/quality_model.pth",
    ),
}


def add_dual_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("DualColorNetwork")
    g.add_argument("--in_channels", type=int, default=3, help="model input channels")
    g.add_argument("--gp", type=int, default=32, help="hidden channels of the global prior branch")
    g.add_argument("--hidden_channels", type=int, default=64, help="CPM hidden channels")


def add_quality_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("QualityNetwork (ViT)")
    g.add_argument("--patch_size", type=int, default=8, help="ViT patch size")
    g.add_argument("--dim", type=int, default=512, help="ViT embedding dim (must be even)")
    g.add_argument("--encoder_depth", type=int, default=2, help="number of self-attention blocks")
    g.add_argument("--decoder_depth", type=int, default=1, help="number of cross-attention blocks")
    g.add_argument("--heads", type=int, default=8, help="number of attention heads")
    g.add_argument("--drop_out", type=float, default=0.0, help="attention/FFN dropout")
    g.add_argument("--emb_dropout", type=float, default=0.0, help="patch-embedding dropout")


def _add_optim_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("optimization")
    g.add_argument("--loss_weights", type=float, nargs=4,
                   help="dual: [ycbcr, rgb, tv, color] / quality: [contrastive, rank, rotation, distance]")
    g.add_argument("--optim", choices=("SGD", "AdamW"))
    g.add_argument("--lr", type=float, help="learning rate")
    g.add_argument("--momentum", type=float, default=0.95, help="SGD momentum")
    g.add_argument("--eps", type=float, default=1e-8, help="AdamW eps")
    g.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], help="AdamW betas")
    g.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay (the paper uses plain Adam)")

    g = parser.add_argument_group("scheduler")
    g.add_argument("--scheduler", choices=("LambdaLR", "CosineWarmUp", "none"))
    g.add_argument("--lambda_weight", type=float, default=0.975, help="LambdaLR: lr *= lambda_weight ** epoch")
    g.add_argument("--t_scheduler", type=int, default=100, help="CosineWarmUp: first cycle length (T_0)")
    g.add_argument("--trigger_scheduler", type=int, default=1, help="CosineWarmUp: cycle length multiplier (T_mult)")
    g.add_argument("--eta_scheduler", type=float, help="CosineWarmUp: peak learning rate (eta_max)")
    g.add_argument("--up_scheduler", type=int, default=10, help="CosineWarmUp: warm-up epochs (T_up)")
    g.add_argument("--gamma_scheduler", type=float, default=0.5, help="CosineWarmUp: eta_max decay per cycle")


def build_parser(task: str) -> argparse.ArgumentParser:
    """Full parser for ``train.py`` with the defaults of ``task`` applied."""
    p = argparse.ArgumentParser(description="Low-light image enhancement training")
    p.add_argument("--task", choices=TASKS, required=True, help="dual: enhancement network / quality: quality network")
    p.add_argument("--mode", choices=MODES, default="train")
    p.add_argument("--csv_path", required=True,
                   help="directory with train.csv / valid.csv [/ test.csv]; each has an 'image' column "
                        "and, for paired data, an 'input' column of low-light images")
    p.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("runtime")
    g.add_argument("--accelerator", default="auto", help="cpu / gpu / mps / auto")
    g.add_argument("--devices", type=int, nargs="+", default=None,
                   help="GPU ids to use (e.g. --devices 0 1); omit to let Lightning choose")
    g.add_argument("--epoch", type=int, default=300, help="max epochs")
    g.add_argument("--batch_size", type=int)
    g.add_argument("--workers", type=int, default=1, help="DataLoader workers")
    g.add_argument("--check_val", type=int, default=10, help="validate every N epochs")
    g.add_argument("--check_loss", type=int, default=100, help="log training losses every N steps")
    g.add_argument("--fast_dev_run", type=int, default=0, help="run N batches of each stage (smoke test)")
    g.add_argument("--log_path", "--log-path", default="./log", help="CSV log directory")

    g = p.add_argument_group("data")
    g.add_argument("--img_size", type=int,
                   help="square input size; 0 keeps the original resolution (dual task, batch_size 1 "
                        "unless all images share one size)")
    g.add_argument("--img_mode", choices=("L", "RGB"), default="RGB",
                   help="L: images are converted to grayscale (replicated to 3 channels)")

    g = p.add_argument_group("weights / outputs")
    g.add_argument("--pretrain", action="store_true", help="initialize from --model_path")
    g.add_argument("--model_path", help="weights to load when --pretrain (or for test/predict)")
    g.add_argument("--model_save_path", default="./weights", help="directory for the best weights")
    g.add_argument("--img_save_path", default="./imgs", help="directory for validation comparison images")
    g.add_argument("--save_n_images", type=int, default=8, help="validation batches to save as images")
    g.add_argument("--pred_save_path", default="./predictions", help="directory for predict-mode outputs")

    _add_optim_args(p)
    add_dual_model_args(p)

    if task == "dual":
        g = p.add_argument_group("dual task")
        g.add_argument("--darken_range", type=float, nargs=2, default=[0.5, 0.9],
                       help="input is synthesized by scaling pixel values by a factor in this range")
    else:
        add_quality_model_args(p)
        g = p.add_argument_group("quality task")
        g.add_argument("--temperature", type=float, default=0.5, help="group contrastive loss temperature")
        g.add_argument("--ema_gamma", type=float, default=0.9, help="EMA decay of the pair-selection model")
        g.add_argument("--dual_model_path", default="./weights/model.pth", help="pretrained DualColorNetwork")
        g.add_argument("--no_dual_pretrain", dest="dual_pretrain", action="store_false",
                       help="do not load --dual_model_path (debugging only)")
        g.add_argument("--blur_sigma_high", type=float, nargs=2, default=[2.0, 4.0],
                       help="sigma range of the strongly blurred image")
        g.add_argument("--blur_sigma_low", type=float, nargs=2, default=[0.5, 1.5],
                       help="sigma range of the weakly blurred image")

    p.set_defaults(**TASK_DEFAULTS[task])
    return p


def validate_args(args: argparse.Namespace) -> None:
    """Fail fast on combinations that would otherwise crash deep inside training."""
    if args.img_size < 0:
        raise ValueError(f"--img_size must be >= 0, got {args.img_size}")
    if args.task == "quality":
        if args.img_size == 0:
            raise ValueError("--img_size 0 (original resolution) is only supported for --task dual")
        if args.img_size % args.patch_size:
            raise ValueError(f"--img_size ({args.img_size}) must be divisible by --patch_size ({args.patch_size})")
        if args.dim % 2:
            raise ValueError(f"--dim ({args.dim}) must be even (sine position embedding)")
        if args.dim % args.heads:
            raise ValueError(f"--dim ({args.dim}) must be divisible by --heads ({args.heads})")
    else:
        lo, hi = args.darken_range
        if not 0 < lo <= hi:
            raise ValueError(f"--darken_range must satisfy 0 < low <= high, got {args.darken_range}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    # Peek at --task first so the full parser can apply that task's defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", choices=TASKS)
    known, _ = pre.parse_known_args(argv)
    args = build_parser(known.task or "dual").parse_args(argv)
    validate_args(args)
    args.img_shape = (args.img_size, args.img_size) if args.img_size else None
    return args
