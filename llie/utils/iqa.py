"""Evaluation of quality features against human scores (e.g. KADID-10k DMOS).

Protocol (as in CONTRIQUE / Re-IQA): repeat ``n_splits`` random train/test splits
that never put two images of the same reference (pristine) image on both sides,
fit a Ridge regressor on standardized train features, and report the median
SRCC / PLCC on the test side. The Ridge strength is chosen per split by a
reference-grouped cross-validation on the train side only.
"""
import os
import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llie import config


def srcc(pred, label) -> float:
    return float(stats.spearmanr(pred, label)[0])


def plcc(pred, label) -> float:
    return float(stats.pearsonr(pred, label)[0])


DEFAULT_ALPHAS = tuple(config.EVAL_IQA["alphas"])
DEFAULT_INNER_FOLDS = config.EVAL_IQA["inner_folds"]


def make_regressor(alpha=1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_regressor(features, labels, groups, alphas=DEFAULT_ALPHAS, inner_folds=DEFAULT_INNER_FOLDS):
    """Ridge with alpha picked by reference-grouped K-fold CV; one alpha -> no search."""
    if len(alphas) == 1:
        return make_regressor(alphas[0]).fit(features, labels)
    folds = min(inner_folds, len(np.unique(groups)))
    if folds < 2:
        raise ValueError(f"choosing alpha needs train samples from >= 2 reference images, got {folds}; "
                         "use more references or pass a single alpha")
    search = GridSearchCV(make_regressor(), {"ridge__alpha": list(alphas)}, cv=GroupKFold(folds))
    return search.fit(features, labels, groups=groups).best_estimator_


def _predict(reg, features):
    # numpy 2 + macOS Accelerate raises spurious divide/overflow flags in matmul
    # (checked against a manual solve); guard the result itself instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pred = reg.predict(features)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Ridge produced non-finite predictions")
    return pred


def load_iqa_table(label_csv, image_dir, image_col, group_col, label_col):
    """(paths, labels, groups) of a human-score CSV; groups = reference image per sample."""
    df = pd.read_csv(label_csv)
    paths = [os.path.join(image_dir, name) for name in df[image_col]]
    labels = df[label_col].to_numpy(dtype=np.float64)
    if group_col in df.columns:
        groups = df[group_col].to_numpy()
    else:
        warnings.warn(f"no '{group_col}' column: splitting per image, so content can leak across splits")
        groups = np.arange(len(df))
    return paths, labels, groups


def validation_references(groups: Sequence, fraction=config.IQA_DATA["val_ref_fraction"],
                          seed=config.IQA_DATA["val_ref_seed"]) -> set:
    """Fixed set of references held out for model selection (at least one, never all)."""
    refs = np.array(sorted(set(np.asarray(groups).tolist())))
    n_val = min(len(refs) - 1, max(1, int(round(fraction * len(refs)))))
    return set(np.random.default_rng(seed).permutation(refs)[:n_val].tolist())


def grouped_splits(groups: Sequence, n_splits=10, test_size=0.2, seed=0):
    """Yield (train_idx, test_idx); all images of one reference stay on one side."""
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return splitter.split(np.zeros(len(groups)), groups=groups)


def evaluate_features(features, labels, groups, n_splits=10, test_size=0.2, alphas=DEFAULT_ALPHAS,
                      seed=0, inner_folds=DEFAULT_INNER_FOLDS) -> Dict:
    """Median (and per-split) test SRCC / PLCC of a Ridge regressor on ``features``."""
    features, labels = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups)
    per_split = {"srcc": [], "plcc": [], "alpha": []}
    for train_idx, test_idx in grouped_splits(groups, n_splits, test_size, seed):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            reg = fit_regressor(features[train_idx], labels[train_idx], groups[train_idx], alphas, inner_folds)
        pred = _predict(reg, features[test_idx])
        per_split["srcc"].append(srcc(pred, labels[test_idx]))
        per_split["plcc"].append(plcc(pred, labels[test_idx]))
        per_split["alpha"].append(float(reg.named_steps["ridge"].alpha))
    return {
        "srcc": float(np.median(per_split["srcc"])),
        "plcc": float(np.median(per_split["plcc"])),
        "srcc_std": float(np.std(per_split["srcc"])),
        "plcc_std": float(np.std(per_split["plcc"])),
        "per_split": per_split,
    }
