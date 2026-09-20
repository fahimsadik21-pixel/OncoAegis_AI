"""
2D U-Net for ISIC2016 skin lesion segmentation.

Classes:
    0 = background
    1 = lesion

Research use only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

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
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


def _match_size(
    source,
    target,
):

    if source.shape[2:] == target.shape[2:]:
        return source

    return F.interpolate(
        source,
        size=target.shape[2:],
        mode="bilinear",
        align_corners=False,
    )


class SkinLesionUNet2D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
    ):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        self.enc1 = DoubleConv(
            in_channels,
            c1,
        )

        self.pool1 = nn.MaxPool2d(
            2
        )

        self.enc2 = DoubleConv(
            c1,
            c2,
        )

        self.pool2 = nn.MaxPool2d(
            2
        )

        self.enc3 = DoubleConv(
            c2,
            c3,
        )

        self.pool3 = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            c3,
            c4,
        )

        self.up3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            c3 + c3,
            c3,
        )

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            c2 + c2,
            c2,
        )

        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            c1 + c1,
            c1,
        )

        self.out = nn.Conv2d(
            c1,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):

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

        b = self.bottleneck(
            self.pool3(
                e3
            )
        )

        d3 = self.up3(
            b
        )

        d3 = _match_size(
            d3,
            e3
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

        d2 = self.up2(
            d3
        )

        d2 = _match_size(
            d2,
            e2
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

        d1 = self.up1(
            d2
        )

        d1 = _match_size(
            d1,
            e1
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

        return self.out(
            d1
        )


if __name__ == "__main__":

    model = SkinLesionUNet2D()

    x = torch.randn(
        1,
        3,
        256,
        256,
    )

    y = model(
        x
    )

    print(
        y.shape
    )