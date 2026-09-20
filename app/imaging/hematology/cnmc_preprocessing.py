from __future__ import annotations

import torch
import torch.nn.functional as F


TARGET_SIZE = (224, 224)


def preprocess_cnmc_image(
    image: torch.Tensor,
) -> torch.Tensor:

    if image.ndim != 3:
        raise ValueError(
            f"Expected [3,H,W], got {image.shape}"
        )

    if image.shape[0] != 3:
        raise ValueError(
            f"Expected RGB image, got {image.shape}"
        )

    image = image.float() / 255.0

    image = image.unsqueeze(0)

    image = F.interpolate(
        image,
        size=TARGET_SIZE,
        mode="bilinear",
        align_corners=False,
    )

    image = image.squeeze(0)

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=image.dtype,
        device=image.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=image.dtype,
        device=image.device,
    ).view(3, 1, 1)

    image = (
        image - mean
    ) / std

    return image