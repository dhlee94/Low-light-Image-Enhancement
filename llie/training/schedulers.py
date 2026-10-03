"""Optimizer / learning-rate scheduler factories."""
import math

import torch.optim as optim

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # torch < 2.0
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class CosineAnnealingWarmUpRestarts(LRScheduler):
    """Linear warm-up to ``eta_max`` then cosine decay, restarting every cycle.

    The optimizer's lr is the floor of each cycle; ``eta_max`` is multiplied by
    ``gamma`` after every restart.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected non-negative integer T_up, but got {T_up}")
        if T_up >= T_0:
            raise ValueError(f"T_up ({T_up}) must be smaller than T_0 ({T_0})")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        if self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        progress = (self.T_cur - self.T_up) / (self.T_i - self.T_up)
        return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        elif epoch >= self.T_0:
            if self.T_mult == 1:
                self.T_cur = epoch % self.T_0
                self.cycle = epoch // self.T_0
            else:
                n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.cycle = n
                self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** n
        else:
            self.T_i = self.T_0
            self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def build_optimizer(params, args):
    if args.optim == "SGD":
        return optim.SGD(params, lr=args.lr, momentum=args.momentum)
    return optim.AdamW(params, lr=args.lr, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    """Returns None when ``--scheduler none``. Schedulers step once per epoch."""
    if args.scheduler == "none":
        return None
    if args.scheduler == "LambdaLR":
        decay = args.lambda_weight
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: decay ** epoch)
    return CosineAnnealingWarmUpRestarts(
        optimizer, T_0=args.t_scheduler, T_mult=args.trigger_scheduler, eta_max=args.eta_scheduler,
        T_up=args.up_scheduler, gamma=args.gamma_scheduler)
