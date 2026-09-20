"""
3D U-Net for MSD Task07 Pancreas CT segmentation.

Classes:
    0 = background
    1 = pancreas
    2 = cancer

Input:
    [B, 1, D, H, W]

Output:
    [B, 3, D, H, W]

Research baseline only.
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


class PancreasTumor3DUNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
    ):

        super().__init__()

        # Encoder

        self.enc1 = DoubleConv3D(
            in_channels,
            16,
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        )

        self.enc2 = DoubleConv3D(
            16,
            32,
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        )

        self.enc3 = DoubleConv3D(
            32,
            64,
        )

        self.pool3 = nn.MaxPool3d(
            kernel_size=2
        )

        # Bottleneck

        self.bottleneck = DoubleConv3D(
            64,
            128,
        )

        # Decoder

        self.up3 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv3D(
            128,
            64,
        )

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv3D(
            64,
            32,
        )

        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv3D(
            32,
            16,
        )

        # Output

        self.output = nn.Conv3d(
            16,
            num_classes,
            kernel_size=1,
        )

    @staticmethod
    def _match_size(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if source.shape[2:] == target.shape[2:]:
            return source

        return F.interpolate(
            source,
            size=target.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        x,
    ):

        # Encoder

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool1(
                e1
            )
        )

        e3 = self.enc3(
            self.pool2(
                e2
            )
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool3(
                e3
            )
        )

        # Decoder 3

        d3 = self.up3(
            b
        )

        d3 = self._match_size(
            d3,
            e3,
        )

        d3 = torch.cat(
            [
                d3,
                e3,
            ],
            dim=1,
        )

        d3 = self.dec3(
            d3
        )

        # Decoder 2

        d2 = self.up2(
            d3
        )

        d2 = self._match_size(
            d2,
            e2,
        )

        d2 = torch.cat(
            [
                d2,
                e2,
            ],
            dim=1,
        )

        d2 = self.dec2(
            d2
        )

        # Decoder 1

        d1 = self.up1(
            d2
        )

        d1 = self._match_size(
            d1,
            e1,
        )

        d1 = torch.cat(
            [
                d1,
                e1,
            ],
            dim=1,
        )

        d1 = self.dec1(
            d1
        )

        return self.output(
            d1
        )


if __name__ == "__main__":

    model = (
        PancreasTumor3DUNet()
    )

    x = torch.randn(
        1,
        1,
        128,
        128,
        128,
    )

    with torch.no_grad():

        y = model(
            x
        )

    print(
        "input:",
        x.shape
    )

    print(
        "output:",
        y.shape
    )