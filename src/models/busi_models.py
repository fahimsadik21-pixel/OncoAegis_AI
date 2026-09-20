"""BUSI segmentation plus benign/malignant/normal classification model."""

from __future__ import annotations

import torch
from torch import nn

from src.models.unet import UNet


class BUSIMultiTaskUNet(nn.Module):
    """Shared U-Net encoder with segmentation and image-level class heads."""

    def __init__(self, num_classes: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.segmenter = UNet(in_channels=1, out_channels=1, base_channels=base_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.segmenter.bottleneck_channels, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, num_classes),
        )
        self.num_classes = num_classes
        self.base_channels = base_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        segmentation_logits, bottleneck = self.segmenter(x, return_features=True)
        classification_logits = self.classifier(bottleneck)
        return {
            "segmentation_logits": segmentation_logits,
            "classification_logits": classification_logits,
        }
