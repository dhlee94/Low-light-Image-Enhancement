"""Logic shared by the dual and quality LightningModules.

Subclasses implement ``training_step``, ``shared_eval_step`` and ``predict_step``
and declare which sub-modules are saved (``WEIGHT_FILES``) and which validation
metric selects the best weights (``MONITOR``).
"""
import os
from collections import defaultdict
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch

from llie.training.schedulers import build_optimizer, build_scheduler
from llie.models.weights import load_weights


class MetricAccumulator:
    """Batch-size weighted running mean of scalar metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, n=1, **metrics):
        for name, value in metrics.items():
            self._sums[name] += float(value) * n
            self._counts[name] += n

    def compute(self) -> Dict[str, float]:
        return {name: self._sums[name] / self._counts[name] for name in self._sums}


def format_table(title, metrics):
    width = max(len(k) for k in metrics) if metrics else 0
    lines = [f"[{title}]"] + [f"  {name:<{width}} : {value:.6f}" for name, value in metrics.items()]
    return "\n".join(lines)


class BaseEnhancementModule(pl.LightningModule):
    # attribute name -> file name inside --model_save_path
    WEIGHT_FILES: Dict[str, str] = {}
    # (validation metric, "min" | "max")
    MONITOR: Tuple[str, str] = ("loss", "min")

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.val_metrics = MetricAccumulator()
        self.test_metrics = MetricAccumulator()
        self.best_score = None

    # ---- hooks for subclasses -------------------------------------------------
    def trainable_parameters(self):
        return self.parameters()

    def shared_eval_step(self, batch, batch_idx, stage) -> Dict[str, torch.Tensor]:
        """Return scalar metrics for one validation/test batch."""
        raise NotImplementedError

    # ---- evaluation loop ------------------------------------------------------
    @staticmethod
    def _batch_size(batch):
        return len(batch["input"])

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def validation_step(self, batch, batch_idx):
        self.val_metrics.update(n=self._batch_size(batch), **self.shared_eval_step(batch, batch_idx, "val"))

    def on_validation_epoch_end(self):
        self._finish_eval("val", self.val_metrics)
        if not self.trainer.sanity_checking:
            self._maybe_save_best(self.val_metrics.compute())

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx):
        self.test_metrics.update(n=self._batch_size(batch), **self.shared_eval_step(batch, batch_idx, "test"))

    def on_test_epoch_end(self):
        self._finish_eval("test", self.test_metrics)

    def _finish_eval(self, stage, accumulator):
        results = accumulator.compute()
        if not results:
            return
        self.log_dict({f"{stage}/{k}": v for k, v in results.items()})
        if self.trainer.is_global_zero:
            print(format_table(f"{stage} epoch {self.current_epoch}", results))

    # ---- checkpointing --------------------------------------------------------
    def _is_better(self, score):
        if self.best_score is None:
            return True
        return score < self.best_score if self.MONITOR[1] == "min" else score > self.best_score

    def _maybe_save_best(self, results):
        name = self.MONITOR[0]
        score = results[name]
        if self._is_better(score):
            self.best_score = score
            self.save_weights()
            print(f"New best val/{name} = {score:.6f} at epoch {self.current_epoch}; weights saved.")

    def on_fit_end(self):
        # No validation ran (e.g. --epoch < --check_val): keep the final weights.
        if self.best_score is None:
            self.save_weights()

    def save_weights(self):
        if not self.trainer.is_global_zero:
            return
        os.makedirs(self.args.model_save_path, exist_ok=True)
        for attr, filename in self.WEIGHT_FILES.items():
            torch.save(getattr(self, attr).state_dict(), os.path.join(self.args.model_save_path, filename))

    def restore_best_weights(self):
        """Reload the files written by ``save_weights`` (used before testing after fit)."""
        for attr, filename in self.WEIGHT_FILES.items():
            path = os.path.join(self.args.model_save_path, filename)
            if os.path.exists(path):
                load_weights(getattr(self, attr), path)

    # ---- optimization ---------------------------------------------------------
    def configure_optimizers(self):
        optimizer = build_optimizer(self.trainable_parameters(), self.args)
        scheduler = build_scheduler(optimizer, self.args)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
