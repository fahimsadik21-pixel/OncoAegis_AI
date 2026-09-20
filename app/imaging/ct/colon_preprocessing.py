"""
MSD Task10 Colon CT preprocessing.

Labels:
    0 = background
    1 = colon cancer primary lesion

Internal shape:
    [D, H, W]

Processed shape:
    [96, 96, 96]

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


TARGET_SIZE = (
    96,
    96,
    96,
)

HU_MIN = -150.0
HU_MAX = 250.0


class ColonPreprocessingError(Exception):
    pass


@dataclass
class ProcessedColonVolume:

    image: torch.Tensor

    label: torch.Tensor | None

    original_shape: tuple[int, int, int]

    processed_shape: tuple[int, int, int, int]

    hu_window: tuple[float, float]

    def to_dict(self):

        return {
            "original_shape":
                self.original_shape,

            "processed_shape":
                self.processed_shape,

            "hu_window":
                self.hu_window,
        }


def normalize_colon_ct(
    volume: np.ndarray,
) -> np.ndarray:

    volume = volume.astype(
        np.float32
    )

    volume = np.clip(
        volume,
        HU_MIN,
        HU_MAX,
    )

    volume = (
        volume
        -
        HU_MIN
    ) / (
        HU_MAX
        -
        HU_MIN
    )

    return volume.astype(
        np.float32
    )


def resize_image(
    volume: np.ndarray,
    size=TARGET_SIZE,
) -> torch.Tensor:

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            volume
        )
    ).float()

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=size,
        mode="trilinear",
        align_corners=False,
    )

    return (
        tensor
        .squeeze(0)
        .squeeze(0)
    )


def resize_label(
    label: np.ndarray,
    size=TARGET_SIZE,
) -> torch.Tensor:

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            label
        )
    ).float()

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=size,
        mode="nearest",
    )

    return (
        tensor
        .squeeze(0)
        .squeeze(0)
        .long()
    )


def preprocess_colon_volume(
    image: np.ndarray,
    label: np.ndarray | None = None,
) -> ProcessedColonVolume:

    if image.ndim != 3:

        raise ColonPreprocessingError(
            f"Expected 3D CT volume, got {image.shape}"
        )

    original_shape = tuple(
        int(x)
        for x
        in image.shape
    )

    image = normalize_colon_ct(
        image
    )

    image = resize_image(
        image
    ).unsqueeze(0)

    processed_label = None

    if label is not None:

        if label.ndim != 3:

            raise ColonPreprocessingError(
                f"Expected 3D label, got {label.shape}"
            )

        if tuple(label.shape) != original_shape:

            raise ColonPreprocessingError(
                "Image and label shapes do not match."
            )

        unique_labels = set(
            int(x)
            for x
            in np.unique(
                label
            )
        )

        unexpected = (
            unique_labels
            -
            {0, 1}
        )

        if unexpected:

            raise ColonPreprocessingError(
                f"Unexpected labels: {sorted(unexpected)}"
            )

        processed_label = resize_label(
            label
        )

    return ProcessedColonVolume(

        image=image,

        label=processed_label,

        original_shape=original_shape,

        processed_shape=tuple(
            int(x)
            for x
            in image.shape
        ),

        hu_window=(
            HU_MIN,
            HU_MAX,
        ),
    )