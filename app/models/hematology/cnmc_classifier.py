"""
C-NMC 2019 ALL vs HEM image classifier.

Class 0 = HEM
Class 1 = ALL

Research use only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.MaxPool2d(
                kernel_size=2
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


class CNMCClassifier(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(
                3,
                32,
            ),
            ConvBlock(
                32,
                64,
            ),
            ConvBlock(
                64,
                128,
            ),
            ConvBlock(
                128,
                256,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                p=0.30
            ),
            nn.Linear(
                256,
                128,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Dropout(
                p=0.25
            ),
            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        x = self.features(
            x
        )

        x = self.pool(
            x
        )

        x = self.classifier(
            x
        )

        return x


if __name__ == "__main__":

    model = CNMCClassifier()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    y = model(
        x
    )

    print(
        y.shape
    )