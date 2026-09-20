"""
Skin lesion image preprocessing.

Target:
    256 x 256 RGB

Research use only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


TARGET_SIZE = (
    256,
    256,
)


def preprocess_skin_image(
    image: torch.Tensor,
):

    if image.ndim != 3:

        raise ValueError(
            f"Expected [3,H,W], got {image.shape}"
        )

    image = (
        image.float()
        /
        255.0
    )

    image = image.unsqueeze(
        0
    )

    image = F.interpolate(
        image,
        size=TARGET_SIZE,
        mode="bilinear",
        align_corners=False,
    )

    return image.squeeze(
        0
    )


def preprocess_skin_mask(
    mask: torch.Tensor,
):

    if mask.ndim != 2:

        raise ValueError(
            f"Expected [H,W], got {mask.shape}"
        )

    mask = (
        mask.float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    mask = F.interpolate(
        mask,
        size=TARGET_SIZE,
        mode="nearest",
    )

    return (
        mask
        .squeeze(0)
        .squeeze(0)
        .long()
    )