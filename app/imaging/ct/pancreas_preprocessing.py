"""
MSD Task07 Pancreas CT Preprocessing

Input:
    image: [D, H, W]
    label: [D, H, W]

Output:
    image: [1, 128, 128, 128]
    label: [128, 128, 128]

Labels:
    0 = background
    1 = pancreas
    2 = cancer

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

HU_MIN = -125.0
HU_MAX = 275.0


class PancreasPreprocessingError(Exception):
    pass


@dataclass
class ProcessedPancreasVolume:

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


def window_pancreas_ct(
    volume: np.ndarray,
):

    volume = volume.astype(
        np.float32
    )

    volume = np.clip(
        volume,
        HU_MIN,
        HU_MAX,
    )

    volume = (
        volume - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )

    return volume.astype(
        np.float32
    )


def resize_image(
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

    return (
        tensor
        .squeeze(0)
        .squeeze(0)
        .long()
    )


def preprocess_pancreas_volume(
    image: np.ndarray,
    label: np.ndarray | None = None,
):

    try:

        if image.ndim != 3:

            raise PancreasPreprocessingError(
                f"Expected 3D CT volume, got {image.shape}"
            )

        original_shape = tuple(
            int(x)
            for x in image.shape
        )

        image = window_pancreas_ct(
            image
        )

        image = resize_image(
            image,
            TARGET_SIZE,
        )

        image = image.unsqueeze(
            0
        )

        processed_label = None

        if label is not None:

            if label.ndim != 3:

                raise PancreasPreprocessingError(
                    f"Expected 3D label, got {label.shape}"
                )

            if tuple(label.shape) != original_shape:

                raise PancreasPreprocessingError(
                    "Image and label shapes do not match: "
                    f"{original_shape} != {label.shape}"
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
                {0, 1, 2}
            )

            if unexpected:

                raise PancreasPreprocessingError(
                    f"Unexpected labels: {sorted(unexpected)}"
                )

            processed_label = resize_label(
                label,
                TARGET_SIZE,
            )

        return ProcessedPancreasVolume(

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

    except PancreasPreprocessingError:
        raise

    except Exception as exc:

        raise PancreasPreprocessingError(
            str(exc)
        ) from exc


if __name__ == "__main__":

    print(
        "Pancreas preprocessing module ready."
    )