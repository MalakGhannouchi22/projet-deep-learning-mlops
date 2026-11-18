"""Loss helpers for imbalanced segmentation datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def pixel_class_weights(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], number_of_classes: int
) -> torch.Tensor:
    """Return bounded inverse-frequency weights calculated from training masks."""
    counts = torch.zeros(number_of_classes, dtype=torch.float64)
    for _, mask in dataset:
        counts += torch.bincount(mask.reshape(-1), minlength=number_of_classes)
    present = counts > 0
    weights = torch.ones(number_of_classes, dtype=torch.float64)
    frequencies = counts[present] / counts[present].sum()
    weights[present] = frequencies.median() / frequencies
    weights.clamp_(min=0.25, max=8.0)
    weights[0] = min(float(weights[0]), 0.5)
    return weights.float()
