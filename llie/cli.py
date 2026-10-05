"""Command-line arguments shared by every entry point.

Arguments declare only their type and help text; every default comes from
``llie/config.py`` (applied with ``set_defaults``), so the command line overrides
the config for one run and the config is the only place defaults live.
"""
from __future__ import annotations

import argparse
from typing import Sequence

from llie import config

TASKS = tuple(config.TASKS)
MODES = ("train", "test", "predict")


def add_dual_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("DualColorNetwork")
    g.add_argument("--in_channels", type=int, help="model input channels")
    g.add_argument("--gp", type=int, help="hidden channels of the global prior branch")
    g.add_argument("--hidden_channels", type=int, help="CPM hidden channels")
    parser.set_defaults(**config.DUAL_MODEL)


def add_quality_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("QualityNetwork (ViT)")
    g.add_argument("--patch_size", type=int, help="ViT patch size")
    g.add_argument("--dim", type=int, help="ViT embedding dim (must be even)")
    g.add_argument("--encoder_depth", type=int, help="number of self-attention blocks")
    g.add_argument("--decoder_depth", type=int, help="number of cross-attention blocks")
    g.add_argument("--heads", type=int, help="number of attention heads")
    g.add_argument("--drop_out", type=float, help="attention/FFN dropout")
    g.add_argument("--emb_dropout", type=float, help="patch-embedding dropout")
    g.add_argument("--separate_ycbcr_embedding", action=argparse.BooleanOptionalAction,
                   help="give the YCbCr tokens their own patch embedding (--no-separate_ycbcr_embedding: shared)")
    parser.set_defaults(**config.QUALITY_MODEL)


DEGRADATION_KEYS = tuple(config.DEGRADATION_PRESETS["original"])


def add_degradation_args(parser: argparse.ArgumentParser) -> None:
    """Ranges default to None and are filled from --degradation_preset in ``resolve_degradations``."""
    g = parser.add_argument_group("quality-task degradations: (min, max) ranges; the 'high' pair is more degraded; "
                                  "unset ranges come from --degradation_preset")
    g.add_argument("--degradation_preset", choices=tuple(config.DEGRADATION_PRESETS))
    g.add_argument("--blur_sigma_high", type=float, nargs=2, help="Gaussian blur sigma, strong")
    g.add_argument("--blur_sigma_low", type=float, nargs=2, help="Gaussian blur sigma, weak")
    g.add_argument("--jpeg_quality_high", type=float, nargs=2, help="JPEG quality, strong (lower = heavier)")
    g.add_argument("--jpeg_quality_low", type=float, nargs=2, help="JPEG quality, weak")
    g.add_argument("--noise_var_high", type=float, nargs=2, help="Gaussian noise variance on [0, 1], strong")
    g.add_argument("--noise_var_low", type=float, nargs=2, help="Gaussian noise variance on [0, 1], weak")


def resolve_degradations(args: argparse.Namespace) -> None:
    """Fill every range not given on the command line from the chosen preset."""
    preset = config.DEGRADATION_PRESETS[args.degradation_preset]
    for key in DEGRADATION_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, list(preset[key]))


def _add_optim_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("optimization")
    g.add_argument("--loss_weights", type=float, nargs=4,
                   help="dual: [ycbcr, rgb, tv, color] / quality: [contrastive, rank, rotation, distance]")
    g.add_argument("--optim", choices=("SGD", "AdamW"))
    g.add_argument("--lr", type=float, help="learning rate")
    g.add_argument("--momentum", type=float, help="SGD momentum")
    g.add_argument("--eps", type=float, help="AdamW eps")
    g.add_argument("--betas", type=float, nargs=2, help="AdamW betas")
    g.add_argument("--weight_decay", type=float, help="AdamW weight decay")

    g = parser.add_argument_group("scheduler")
    g.add_argument("--scheduler", choices=("LambdaLR", "CosineWarmUp", "none"))
    g.add_argument("--lambda_weight", type=float, help="LambdaLR: lr *= lambda_weight ** epoch")
    g.add_argument("--t_scheduler", type=int, help="CosineWarmUp: first cycle length (T_0)")
    g.add_argument("--trigger_scheduler", type=int, help="CosineWarmUp: cycle length multiplier (T_mult)")
    g.add_argument("--eta_scheduler", type=float, help="CosineWarmUp: peak learning rate (eta_max)")
    g.add_argument("--up_scheduler", type=int, help="CosineWarmUp: warm-up epochs (T_up)")
    g.add_argument("--gamma_scheduler", type=float, help="CosineWarmUp: eta_max decay per cycle")


def build_parser(task: str) -> argparse.ArgumentParser:
    """Full parser for ``train.py`` with the defaults of ``task`` applied."""
    p = argparse.ArgumentParser(description="Low-light image enhancement training (defaults: llie/config.py)")
    p.add_argument("--task", choices=TASKS, required=True, help="dual: enhancement network / quality: quality network")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--csv_path", required=True,
                   help="directory with train.csv / valid.csv [/ test.csv]; each has an 'image' column "
                        "and, for paired data, an 'input' column of low-light images")
    p.add_argument("--seed", type=int)

    g = p.add_argument_group("runtime")
    g.add_argument("--accelerator", help="cpu / gpu / mps / auto")
    g.add_argument("--devices", type=int, nargs="+",
                   help="GPU ids to use (e.g. --devices 0 1); omit to let Lightning choose")
    g.add_argument("--epoch", type=int, help="max epochs")
    g.add_argument("--batch_size", type=int)
    g.add_argument("--workers", type=int, help="DataLoader workers")
    g.add_argument("--check_val", type=int, help="validate every N epochs")
    g.add_argument("--check_loss", type=int, help="log training losses every N steps")
    g.add_argument("--fast_dev_run", type=int, help="run N batches of each stage (smoke test)")
    g.add_argument("--log_path", "--log-path", help="CSV log directory")

    g = p.add_argument_group("data")
    g.add_argument("--img_size", type=int,
                   help="square input size; 0 keeps the original resolution (dual task, batch_size 1 "
                        "unless all images share one size)")
    g.add_argument("--img_mode", choices=("L", "RGB"),
                   help="L: images are converted to grayscale (replicated to 3 channels)")

    g = p.add_argument_group("weights / outputs")
    g.add_argument("--pretrain", action="store_true", help="initialize from --model_path")
    g.add_argument("--model_path", help="weights to load when --pretrain (or for test/predict)")
    g.add_argument("--model_save_path", help="directory for the best weights")
    g.add_argument("--img_save_path", help="directory for validation comparison images")
    g.add_argument("--save_n_images", type=int, help="validation batches to save as images")
    g.add_argument("--pred_save_path", help="directory for predict-mode outputs")

    _add_optim_args(p)
    add_dual_model_args(p)

    if task == "dual":
        g = p.add_argument_group("dual task")
        g.add_argument("--darken_range", type=float, nargs=2,
                       help="input is synthesized by scaling pixel values by a factor in this range")
    else:
        add_quality_model_args(p)
        add_degradation_args(p)
        g = p.add_argument_group("quality task")
        g.add_argument("--temperature", type=float, help="group contrastive loss temperature")
        g.add_argument("--ema_gamma", type=float, help="EMA decay of the pair-selection model")
        g.add_argument("--dual_model_path", help="pretrained DualColorNetwork")
        g.add_argument("--no_dual_pretrain", dest="dual_pretrain", action="store_false",
                       help="do not load --dual_model_path (debugging only)")
        g.add_argument("--pair_selection", choices=("max_gap", "min_margin", "random"),
                       help="max_gap: pair the EMA model separates most (original) / min_margin: hardest pair / "
                            "random: random degradation type")
        g.add_argument("--contrastive", choices=("group", "type_severity", "none"),
                       help="group: weak vs strong (original) / type_severity: positives share degradation type "
                            "and severity / none: disabled")

    p.set_defaults(**config.TRAIN)
    p.set_defaults(**config.TASKS[task])
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
    if args.task == "quality":
        resolve_degradations(args)
    validate_args(args)
    args.img_shape = (args.img_size, args.img_size) if args.img_size else None
    return args
