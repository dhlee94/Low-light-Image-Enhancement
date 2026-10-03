"""End-to-end: train dual -> train quality on its weights -> predict, on tiny CPU settings."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _load_script(name):
    """scripts/ is not a package; load the entry point by file path."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load_script("train")


@pytest.fixture
def csv_dir(tmp_path):
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    paths = []
    for i in range(6):
        path = img_dir / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    for split in ("train", "valid"):
        pd.DataFrame({"image": paths}).to_csv(tmp_path / f"{split}.csv", index=False)
    return tmp_path


def _common(csv_dir, tmp_path):
    return ["--csv_path", str(csv_dir), "--accelerator", "cpu", "--workers", "0", "--epoch", "2",
            "--check_val", "1", "--img_size", "32", "--batch_size", "2",
            "--model_save_path", str(tmp_path / "weights"), "--img_save_path", str(tmp_path / "imgs"),
            "--pred_save_path", str(tmp_path / "pred"), "--log_path", str(tmp_path / "log")]


def test_dual_then_quality_pipeline(csv_dir, tmp_path):
    common = _common(csv_dir, tmp_path)
    weights = tmp_path / "weights"

    train.main(["--task", "dual", *common])
    assert (weights / "model.pth").exists()
    assert os.listdir(tmp_path / "imgs"), "validation comparison images were not saved"

    train.main(["--task", "dual", "--mode", "predict", "--model_path", str(weights / "model.pth"), *common])
    assert len(os.listdir(tmp_path / "pred" / "enhanced")) == 6

    quality = ["--task", "quality", "--dual_model_path", str(weights / "model.pth"), "--dim", "32",
               "--heads", "4", "--patch_size", "8", "--blur_sigma_high", "1", "2", "--blur_sigma_low", "0.3", "0.6"]
    train.main([*quality, *common])
    assert (weights / "quality_model.pth").exists() and (weights / "select_quality_model.pth").exists()

    train.main([*quality, "--mode", "predict", "--model_path", str(weights / "quality_model.pth"), *common])
    scores = pd.read_csv(tmp_path / "pred" / "quality_scores.csv")
    assert len(scores) == 6 and scores["score"].notna().all()


def test_dual_paired_original_resolution(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(3):
        low, high = tmp_path / f"low{i}.png", tmp_path / f"high{i}.png"
        Image.fromarray(rng.integers(0, 60, (24, 36, 3), dtype=np.uint8)).save(low)
        Image.fromarray(rng.integers(0, 255, (24, 36, 3), dtype=np.uint8)).save(high)
        rows.append({"image": str(high), "input": str(low)})
    for split in ("train", "valid"):
        pd.DataFrame(rows).to_csv(tmp_path / f"{split}.csv", index=False)

    common = [arg for arg in _common(tmp_path, tmp_path)]
    common[common.index("--img_size") + 1] = "0"
    train.main(["--task", "dual", *common])
    train.main(["--task", "dual", "--mode", "predict", "--model_path", str(tmp_path / "weights" / "model.pth"), *common])
    assert sorted(os.listdir(tmp_path / "pred" / "enhanced")) == ["low0.png", "low1.png", "low2.png"]
