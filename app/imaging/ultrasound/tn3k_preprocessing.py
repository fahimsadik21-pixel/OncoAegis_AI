from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = 256


def load_grayscale_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def load_binary_mask(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def preprocess_image(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:

    image = image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    array = array / 255.0

    # [H, W] -> [1, H, W]
    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    return tensor


def preprocess_mask(
    mask: Image.Image,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:

    mask = mask.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST,
    )

    array = np.asarray(
        mask,
        dtype=np.float32,
    )

    # Robust binary conversion for JPG masks
    array = (array >= 127.5).astype(
        np.float32
    )

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    return tensor


def preprocess_pair(
    image_path: str | Path,
    mask_path: str | Path,
    image_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:

    image = load_grayscale_image(
        image_path
    )

    mask = load_binary_mask(
        mask_path
    )

    return (
        preprocess_image(
            image,
            image_size=image_size,
        ),
        preprocess_mask(
            mask,
            image_size=image_size,
        ),
    )