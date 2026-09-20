"""
3D U-Net architecture for MSD Brain Tumor MRI segmentation.

Research model only.
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
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )


    def forward(self, x):

        return self.block(x)




class BrainTumor3DUNet(nn.Module):

    """
    Checkpoint-compatible 3D U-Net.

    Input:
        B x 4 x H x W x D

    Output:
        B x 4 x H x W x D
    """


    def __init__(
        self,
        in_channels=4,
        out_channels=4,
    ):

        super().__init__()


        self.encoder1 = DoubleConv3D(
            in_channels,
            32
        )


        self.pool1 = nn.MaxPool3d(
            2
        )


        self.encoder2 = DoubleConv3D(
            32,
            64
        )


        self.pool2 = nn.MaxPool3d(
            2
        )


        self.bottleneck = DoubleConv3D(
            64,
            128
        )


        self.up2 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2
        )


        self.decoder2 = DoubleConv3D(
            128,
            64
        )


        self.up1 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )


        self.decoder1 = DoubleConv3D(
            64,
            32
        )


        self.output = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )



    def _match_size(
        self,
        x,
        target
    ):

        if x.shape[2:] != target.shape[2:]:

            x = F.interpolate(
                x,
                size=target.shape[2:],
                mode="trilinear",
                align_corners=False
            )

        return x



    def forward(
        self,
        x
    ):


        e1 = self.encoder1(
            x
        )


        e2 = self.encoder2(
            self.pool1(e1)
        )


        b = self.bottleneck(
            self.pool2(e2)
        )


        d2 = self.up2(
            b
        )


        d2 = self._match_size(
            d2,
            e2
        )


        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )


        d2 = self.decoder2(
            d2
        )



        d1 = self.up1(
            d2
        )


        d1 = self._match_size(
            d1,
            e1
        )


        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )


        d1 = self.decoder1(
            d1
        )


        return self.output(
            d1
        )