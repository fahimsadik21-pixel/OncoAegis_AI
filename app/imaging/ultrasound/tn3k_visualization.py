from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def create_tn3k_overlay(
    image_path: str | Path,
    mask: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.40,
) -> Path:

    image_path = Path(image_path)
    output_path = Path(output_path)

    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    mask_image = Image.fromarray(
        ((mask > 0) * 255).astype(
            np.uint8
        )
    )

    mask_image = mask_image.resize(
        (width, height),
        Image.Resampling.NEAREST,
    )

    base = np.asarray(
        image,
        dtype=np.float32,
    )

    binary_mask = (
        np.asarray(mask_image) > 0
    )

    overlay = base.copy()

    # Red overlay for predicted nodule region.
    overlay[
        binary_mask,
        0
    ] = (
        (1.0 - alpha)
        * overlay[binary_mask, 0]
        + alpha * 255.0
    )

    overlay[
        binary_mask,
        1
    ] = (
        (1.0 - alpha)
        * overlay[binary_mask, 1]
    )

    overlay[
        binary_mask,
        2
    ] = (
        (1.0 - alpha)
        * overlay[binary_mask, 2]
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        overlay
    ).save(output_path)

    return output_path


def save_binary_mask(
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:

    output_path = Path(
        output_path
    )

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    rendered = (
        (mask > 0) * 255
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        rendered
    ).save(output_path)

    return output_path