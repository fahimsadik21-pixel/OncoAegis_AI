"""
Liver CT Segmentation Visualization

Creates an axial overlay image containing:
    - Original CT slice
    - Liver segmentation
    - Tumor segmentation

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_DIR = (
    _PROJECT_ROOT
    / "outputs"
    / "liver_visualization"
)


class LiverVisualizationError(Exception):
    pass


def _select_best_slice(
    segmentation_mask: np.ndarray,
) -> int:

    if segmentation_mask.ndim != 3:
        raise LiverVisualizationError(
            f"Expected 3D mask, got {segmentation_mask.shape}"
        )

    tumor_counts = np.count_nonzero(
        segmentation_mask == 2,
        axis=(1, 2),
    )

    if tumor_counts.max() > 0:
        return int(
            np.argmax(
                tumor_counts
            )
        )

    liver_counts = np.count_nonzero(
        segmentation_mask == 1,
        axis=(1, 2),
    )

    if liver_counts.max() > 0:
        return int(
            np.argmax(
                liver_counts
            )
        )

    return (
        segmentation_mask.shape[0]
        //
        2
    )


def _display_window(
    image_slice: np.ndarray,
):

    image_slice = image_slice.astype(
        np.float32
    )

    low = np.percentile(
        image_slice,
        1
    )

    high = np.percentile(
        image_slice,
        99
    )

    if high <= low:
        return image_slice

    image_slice = np.clip(
        image_slice,
        low,
        high,
    )

    image_slice = (
        image_slice - low
    ) / (
        high - low
    )

    return image_slice


def save_liver_overlay(
    ct_volume: np.ndarray,
    segmentation_mask: np.ndarray,
    output_path: str | Path | None = None,
    title: str = "Liver CT Segmentation",
):

    if ct_volume.ndim != 3:
        raise LiverVisualizationError(
            f"Expected 3D CT volume, got {ct_volume.shape}"
        )

    if segmentation_mask.ndim != 3:
        raise LiverVisualizationError(
            f"Expected 3D segmentation mask, got {segmentation_mask.shape}"
        )

    if ct_volume.shape != segmentation_mask.shape:
        raise LiverVisualizationError(
            "CT and segmentation mask shapes do not match: "
            f"{ct_volume.shape} != {segmentation_mask.shape}"
        )

    if output_path is None:

        DEFAULT_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            DEFAULT_OUTPUT_DIR
            / "liver_tumor_overlay.png"
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    slice_index = _select_best_slice(
        segmentation_mask
    )

    ct_slice = ct_volume[
        slice_index
    ]

    mask_slice = segmentation_mask[
        slice_index
    ]

    display_image = _display_window(
        ct_slice
    )

    liver_overlay = np.ma.masked_where(
        mask_slice != 1,
        mask_slice,
    )

    tumor_overlay = np.ma.masked_where(
        mask_slice != 2,
        mask_slice,
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        display_image,
        cmap="gray",
    )

    plt.imshow(
        liver_overlay,
        cmap="Blues",
        alpha=0.30,
        vmin=0,
        vmax=2,
    )

    plt.imshow(
        tumor_overlay,
        cmap="autumn",
        alpha=0.65,
        vmin=0,
        vmax=2,
    )

    plt.title(
        f"{title} | Axial Slice {slice_index}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    return {
        "output_path": str(
            output_path
        ),
        "slice_index": slice_index,
    }


if __name__ == "__main__":
    print(
        "Liver visualization module ready."
    )