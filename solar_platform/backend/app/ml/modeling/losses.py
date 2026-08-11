"""
app.ml.modeling.losses
----------------------
Loss functions for scientific curve surrogate learning.

Includes Huber Loss (Smooth L1), MAE, MSE, and Relative Error Loss.
"""
from __future__ import annotations

import math
from typing import List, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def calculate_huber_loss_numeric(predictions: List[float], targets: List[float], delta: float = 1.0) -> float:
    """Calculate Smooth L1 Huber Loss on numeric lists."""
    if not predictions or not targets:
        return 0.0
    total_loss = 0.0
    for pred, tgt in zip(predictions, targets):
        diff = abs(pred - tgt)
        if diff < delta:
            loss = 0.5 * (diff ** 2)
        else:
            loss = delta * (diff - 0.5 * delta)
        total_loss += loss
    return total_loss / len(predictions)


def calculate_mae_numeric(predictions: List[float], targets: List[float]) -> float:
    """Calculate Mean Absolute Error on numeric lists."""
    if not predictions or not targets:
        return 0.0
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)


def calculate_rmse_numeric(predictions: List[float], targets: List[float]) -> float:
    """Calculate Root Mean Squared Error on numeric lists."""
    if not predictions or not targets:
        return 0.0
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    return math.sqrt(mse)


if TORCH_AVAILABLE:
    class ScientificHuberLoss(nn.Module):
        """PyTorch Smooth L1 Huber Loss for Curve Surrogate Learning."""

        def __init__(self, delta: float = 1.0):
            super().__init__()
            self.huber = nn.SmoothL1Loss(beta=delta)

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.huber(pred, target)
else:
    class ScientificHuberLoss:
        """Fallback numeric Huber Loss container when PyTorch is not available."""

        def __init__(self, delta: float = 1.0):
            self.delta = delta

        def forward(self, pred: List[float], target: List[float]) -> float:
            return calculate_huber_loss_numeric(pred, target, delta=self.delta)

