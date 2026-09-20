"""
Lightweight 3D U-Net for coarse liver localization.

Stage 1 of the IRCADb01 liver analysis pipeline.

Classes:
    0 = background
    1 = liver region

Tumor voxels are treated as part of the liver region
for the localization stage.

Research use only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.block(
            x
        )


class LiverLocator3DUNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
    ):

        super().__init__()

        # Smaller than the tumor model
        # to keep RTX 3050 6GB usage manageable.

        self.encoder1 = DoubleConv3D(
            in_channels,
            16,
        )

        self.pool1 = nn.MaxPool3d(
            2
        )

        self.encoder2 = DoubleConv3D(
            16,
            32,
        )

        self.pool2 = nn.MaxPool3d(
            2
        )

        self.encoder3 = DoubleConv3D(
            32,
            64,
        )

        self.pool3 = nn.MaxPool3d(
            2
        )

        self.bottleneck = DoubleConv3D(
            64,
            128,
        )

        self.up3 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.decoder3 = DoubleConv3D(
            128,
            64,
        )

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.decoder2 = DoubleConv3D(
            64,
            32,
        )

        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2,
        )

        self.decoder1 = DoubleConv3D(
            32,
            16,
        )

        self.output = nn.Conv3d(
            16,
            out_channels,
            kernel_size=1,
        )

    @staticmethod
    def _match_size(
        source: torch.Tensor,
        target: torch.Tensor,
    ):

        if source.shape[2:] != target.shape[2:]:

            source = F.interpolate(
                source,
                size=target.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return source

    def forward(
        self,
        x,
    ):

        e1 = self.encoder1(
            x
        )

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(
            b
        )

        d3 = self._match_size(
            d3,
            e3,
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )

        d3 = self.decoder3(
            d3
        )

        d2 = self.up2(
            d3
        )

        d2 = self._match_size(
            d2,
            e2,
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )

        d2 = self.decoder2(
            d2
        )

        d1 = self.up1(
            d2
        )

        d1 = self._match_size(
            d1,
            e1,
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )

        d1 = self.decoder1(
            d1
        )

        return self.output(
            d1
        )