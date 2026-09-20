"""Pure-PyTorch Chamfer distances (upstream Point-MAE/BERT/GPT, no CUDA extension).
With d1/d2 the squared NN distances: L1 = (mean(sqrt(d1)) + mean(sqrt(d2))) / 2,
L2 = mean(d1) + mean(d2). The default pretraining loss cdl12 = L1 + L2.
"""

import torch
import torch.nn as nn


def _nn_squared_distances(pred, target):
    """pred (B, N, 3), target (B, M, 3) -> (d1 [B, N], d2 [B, M]): squared distance from
    each pred point to its nearest target point, and vice versa."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    diff = pred.unsqueeze(2) - target.unsqueeze(1)   # (B, N, M, 3)
    distances_sq = (diff * diff).sum(dim=-1)         # (B, N, M)

    d1 = distances_sq.min(dim=2)[0]                  # (B, N)
    d2 = distances_sq.min(dim=1)[0]                  # (B, M)
    return d1, d2


class ChamferDistanceL1(nn.Module):
    """Chamfer-L1: halved Euclidean distances, as in the original repos."""

    def __init__(self, eps=1e-8):
        super().__init__()
        # small epsilon inside the sqrt keeps the gradient finite at d == 0
        self.eps = eps

    def forward(self, pred, target):
        d1, d2 = _nn_squared_distances(pred, target)
        d1 = torch.sqrt(d1 + self.eps)
        d2 = torch.sqrt(d2 + self.eps)
        return (d1.mean() + d2.mean()) / 2


class ChamferDistanceL2(nn.Module):
    """Chamfer-L2: sum of mean *squared* distances, as in the original repos."""

    def forward(self, pred, target):
        d1, d2 = _nn_squared_distances(pred, target)
        return d1.mean() + d2.mean()
