"""Loss functions shared by segmentation and multi-task training."""

from __future__ import annotations

import torch
from torch import nn


def soft_dice_score(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    probability = probability.flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def combined_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target.float())
    return 0.5 * bce + 0.5 * (1.0 - soft_dice_score(logits, target))


def combined_busi_loss(
    segmentation_logits: torch.Tensor,
    segmentation_target: torch.Tensor,
    classification_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    classification_weight: float = 1.0,
) -> torch.Tensor:
    segmentation_loss = combined_segmentation_loss(segmentation_logits, segmentation_target)
    classification_loss = nn.functional.cross_entropy(classification_logits, labels)
    return segmentation_loss + classification_weight * classification_loss
