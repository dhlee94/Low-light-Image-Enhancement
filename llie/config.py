"""Every default hyperparameter, in one place.

``llie/cli.py`` builds the command-line parsers from these values, so any of them
can still be overridden per run (e.g. ``--lr 1e-4``); edit this file to change a
default. Library code that needs a default (degradation ranges, Ridge grid) also
reads it from here. Fixed constants (colour-space matrices, ImageNet mean/std)
are not hyperparameters and stay next to the code that uses them.
"""

# ---- shared by both training tasks --------------------------------------------
TRAIN = dict(
    mode="train",
    seed=0,
    # runtime
    accelerator="auto",
    devices=None,            # GPU ids; None lets Lightning choose
    epoch=300,
    workers=1,
    check_val=10,            # validate every N epochs
    check_loss=100,          # log training losses every N steps
    fast_dev_run=0,
    log_path="./log",
    # data
    img_mode="RGB",          # "L" converts images to grayscale
    # weights / outputs
    pretrain=False,
    model_save_path="./weights",
    img_save_path="./imgs",
    save_n_images=8,
    pred_save_path="./predictions",
    # optimizer (fields unused by the chosen optimizer are ignored)
    momentum=0.95,           # SGD
    eps=1e-8,                # AdamW
    betas=[0.9, 0.999],      # AdamW
    weight_decay=0.0,        # AdamW; the DualCSNet paper uses plain Adam
    # scheduler
    lambda_weight=0.975,     # LambdaLR: lr *= lambda_weight ** epoch
    t_scheduler=100,         # CosineWarmUp: first cycle length (T_0)
    trigger_scheduler=1,     # CosineWarmUp: cycle length multiplier (T_mult)
    up_scheduler=10,         # CosineWarmUp: warm-up epochs (T_up)
    gamma_scheduler=0.5,     # CosineWarmUp: eta_max decay per cycle
)

# ---- per task (override TRAIN) --------------------------------------------------
TASKS = {
    # Follows the DualCSNet paper (Sci. Rep. 2023): loss weights 0.01 / 1 / 0.1 / 0.1,
    # Adam betas (0.9, 0.99) and a constant learning rate of 5e-5.
    "dual": dict(
        batch_size=1,
        img_size=512,                       # 0 keeps the original resolution
        loss_weights=[0.01, 1.0, 0.1, 0.1],  # ycbcr, rgb, tv, color
        optim="AdamW",
        lr=5e-5,
        betas=[0.9, 0.99],
        scheduler="none",
        eta_scheduler=5e-5,
        model_path="./weights/model.pth",
        darken_range=[0.5, 0.9],            # synthetic input = image * U(range) (no 'input' column)
    ),
    "quality": dict(
        batch_size=8,
        img_size=256,
        loss_weights=[1.0, 1.0, 1.0, 1.0],  # contrastive, rank, rotation, distance
        optim="SGD",
        lr=1.25e-3,
        scheduler="LambdaLR",
        eta_scheduler=1.25e-4,
        model_path="./weights/quality_model.pth",
        temperature=0.5,                    # group contrastive loss
        ema_gamma=0.9,                      # EMA decay of the pair-selection model
        dual_model_path="./weights/model.pth",
        dual_pretrain=True,
    ),
}

# ---- networks --------------------------------------------------------------------
DUAL_MODEL = dict(
    in_channels=3,
    gp=32,                   # hidden channels of the global prior branch
    hidden_channels=64,      # CPM hidden channels
)

QUALITY_MODEL = dict(
    patch_size=8,
    dim=512,                 # must be even (sine position embedding) and divisible by heads
    encoder_depth=2,
    decoder_depth=1,
    heads=8,
    drop_out=0.0,
    emb_dropout=0.0,
    # True: RGB and YCbCr tokens get their own patch embedding; False: one shared embedding.
    separate_ycbcr_embedding=False,
)

# ---- synthetic degradations for the quality task ("high" = more degraded) ----------
DEGRADATIONS = dict(
    blur_sigma_high=[2.0, 4.0],
    blur_sigma_low=[0.5, 1.5],
    jpeg_quality_high=[40, 60],
    jpeg_quality_low=[80, 90],
    noise_var_high=[5e-5, 5.1e-5],   # Gaussian noise variance on the [0, 1] scale
    noise_var_low=[1e-5, 1.1e-5],
)

# ---- quality evaluation (scripts/eval_iqa.py) ----------------------------------------
EVAL_IQA = dict(
    backbone="quality",
    seed=0,
    label_csv="data/kadid10k/dmos.csv",
    image_dir="data/kadid10k/images",
    image_col="dist_img",
    group_col="ref_img",     # splits never share a reference image
    label_col="dmos",
    img_size=256,            # quality/random input size; must match training
    batch_size=16,
    workers=2,
    n_splits=10,
    test_size=0.2,
    alphas=[10.0 ** k for k in range(-2, 9)],  # Ridge strengths searched per split
    inner_folds=5,           # grouped CV folds for choosing alpha
    dual_model_path="./weights/model.pth",
    quality_model_path="./weights/quality_model.pth",
    out_dir="./iqa_results",
    save_regressor=False,
)
