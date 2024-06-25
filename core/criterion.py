import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import logging

def total_variation_loss(x):
    h_diff = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
    w_diff = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
    loss = (h_diff + w_diff) / (x.size(0) * x.size(1) * x.size(2) * x.size(3))
    return loss

def color_loss(output, target):
    return 1 - torch.mean(torch.cosine_similarity(output, target, dim=1))