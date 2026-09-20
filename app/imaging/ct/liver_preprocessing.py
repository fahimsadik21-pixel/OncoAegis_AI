"""
Liver CT preprocessing utilities.

IRCADb01 Liver Tumor Segmentation

Key idea:
- Use liver mask to create a liver-centered ROI crop
- Pad the crop
- Resize image and label together to a fixed 3D size

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


TARGET_SIZE = (
    128,
    128,
    128,
)

ROI_MARGIN = (
    8,
    32,
    32,
)


class LiverPreprocessingError(Exception):
    pass


@dataclass
class ProcessedLiverVolume:

    image: torch.Tensor
    label: torch.Tensor

    original_shape: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    processed_shape: tuple[int, int, int, int]

    crop_bounds: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]

    def to_dict(self):

        return {
            "original_shape":
                self.original_shape,

            "cropped_shape":
                self.cropped_shape,

            "processed_shape":
                self.processed_shape,

            "crop_bounds":
                self.crop_bounds,
        }


def normalize_ct(
    volume: np.ndarray,
):

    volume = volume.astype(
        np.float32
    )

    mean = float(
        volume.mean()
    )

    std = float(
        volume.std()
    )

    if std > 0:

        volume = (
            volume - mean
        ) / std

    return volume.astype(
        np.float32
    )


def _find_liver_bounds(
    label: np.ndarray,
):

    if label.ndim != 3:

        raise LiverPreprocessingError(
            f"Expected 3D label, got {label.shape}"
        )

    liver_region = (
        label > 0
    )

    coords = np.argwhere(
        liver_region
    )

    if coords.size == 0:

        raise LiverPreprocessingError(
            "No liver region found in segmentation label."
        )

    z_min, y_min, x_min = coords.min(
        axis=0
    )

    z_max, y_max, x_max = coords.max(
        axis=0
    )

    return (
        int(z_min),
        int(z_max),
        int(y_min),
        int(y_max),
        int(x_min),
        int(x_max),
    )


def _apply_margin(
    bounds,
    shape,
    margin=ROI_MARGIN,
):

    (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    ) = bounds

    depth, height, width = shape

    mz, my, mx = margin

    z_min = max(
        0,
        z_min - mz,
    )

    z_max = min(
        depth - 1,
        z_max + mz,
    )

    y_min = max(
        0,
        y_min - my,
    )

    y_max = min(
        height - 1,
        y_max + my,
    )

    x_min = max(
        0,
        x_min - mx,
    )

    x_max = min(
        width - 1,
        x_max + mx,
    )

    return (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    )


def crop_to_liver_roi(
    image: np.ndarray,
    label: np.ndarray,
):

    if image.shape != label.shape:

        raise LiverPreprocessingError(
            "Image and label shapes do not match: "
            f"{image.shape} != {label.shape}"
        )

    bounds = _find_liver_bounds(
        label
    )

    bounds = _apply_margin(
        bounds,
        image.shape,
    )

    (
        z_min,
        z_max,
        y_min,
        y_max,
        x_min,
        x_max,
    ) = bounds

    image_crop = image[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    label_crop = label[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    crop_bounds = (
        (
            z_min,
            z_max,
        ),
        (
            y_min,
            y_max,
        ),
        (
            x_min,
            x_max,
        ),
    )

    return (
        image_crop,
        label_crop,
        crop_bounds,
    )


def resize_volume(
    volume: np.ndarray,
    size=TARGET_SIZE,
):

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            volume
        )
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    tensor = F.interpolate(
        tensor,
        size=size,
        mode="trilinear",
        align_corners=False,
    )

    return tensor.squeeze(
        0
    ).squeeze(
        0
    )


def resize_label(
    label: np.ndarray,
    size=TARGET_SIZE,
):

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            label
        )
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    tensor = F.interpolate(
        tensor,
        size=size,
        mode="nearest",
    )

    return tensor.squeeze(
        0
    ).squeeze(
        0
    ).long()


def preprocess_liver_volume(
    image: np.ndarray,
    label: np.ndarray,
):

    try:

        if image.ndim != 3:

            raise LiverPreprocessingError(
                f"Expected 3D image, got {image.shape}"
            )

        if label.ndim != 3:

            raise LiverPreprocessingError(
                f"Expected 3D label, got {label.shape}"
            )

        original_shape = tuple(
            int(x)
            for x in image.shape
        )

        image = normalize_ct(
            image
        )

        (
            image_crop,
            label_crop,
            crop_bounds,
        ) = crop_to_liver_roi(
            image,
            label,
        )

        cropped_shape = tuple(
            int(x)
            for x in image_crop.shape
        )

        image_resized = resize_volume(
            image_crop,
            TARGET_SIZE,
        )

        label_resized = resize_label(
            label_crop,
            TARGET_SIZE,
        )

        image_resized = image_resized.unsqueeze(
            0
        )

        return ProcessedLiverVolume(

            image=image_resized,

            label=label_resized,

            original_shape=original_shape,

            cropped_shape=cropped_shape,

            processed_shape=tuple(
                int(x)
                for x in image_resized.shape
            ),

            crop_bounds=crop_bounds,
        )

    except LiverPreprocessingError:
        raise

    except Exception as exc:

        raise LiverPreprocessingError(
            str(exc)
        ) from exc