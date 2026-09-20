"""
Pancreas CT visualization utilities.

Labels:
    0 = background
    1 = pancreas
    2 = cancer

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class PancreasVisualizationError(Exception):
    pass


def _select_slice(
    segmentation_mask: np.ndarray,
) -> int:

    cancer_counts = np.sum(
        segmentation_mask == 2,
        axis=(1, 2),
    )

    if np.max(cancer_counts) > 0:
        return int(
            np.argmax(cancer_counts)
        )

    pancreas_counts = np.sum(
        segmentation_mask > 0,
        axis=(1, 2),
    )

    if np.max(pancreas_counts) > 0:
        return int(
            np.argmax(pancreas_counts)
        )

    return int(
        segmentation_mask.shape[0]
        // 2
    )


def save_pancreas_overlay(
    volume: np.ndarray,
    segmentation_mask: np.ndarray,
    output_path: str | Path,
):

    if volume.ndim != 3:
        raise PancreasVisualizationError(
            f"Expected 3D CT volume, got {volume.shape}"
        )

    if segmentation_mask.ndim != 3:
        raise PancreasVisualizationError(
            f"Expected 3D mask, got {segmentation_mask.shape}"
        )

    if volume.shape != segmentation_mask.shape:
        raise PancreasVisualizationError(
            "CT volume and segmentation mask shapes do not match."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    slice_index = _select_slice(
        segmentation_mask
    )

    image_slice = volume[
        slice_index
    ]

    mask_slice = segmentation_mask[
        slice_index
    ]

    low = float(
        np.percentile(
            image_slice,
            1
        )
    )

    high = float(
        np.percentile(
            image_slice,
            99
        )
    )

    if high <= low:
        low = float(
            image_slice.min()
        )

        high = float(
            image_slice.max()
        )

    fig = plt.figure(
        figsize=(
            8,
            8,
        )
    )

    plt.imshow(
        image_slice,
        cmap="gray",
        vmin=low,
        vmax=high,
    )

    pancreas_overlay = np.ma.masked_where(
        mask_slice != 1,
        mask_slice,
    )

    cancer_overlay = np.ma.masked_where(
        mask_slice != 2,
        mask_slice,
    )

    if np.any(
        mask_slice == 1
    ):
        plt.imshow(
            pancreas_overlay,
            cmap="Blues",
            alpha=0.35,
        )

    if np.any(
        mask_slice == 2
    ):
        plt.imshow(
            cancer_overlay,
            cmap="autumn",
            alpha=0.65,
        )

    plt.title(
        f"Pancreas CT AI Overlay — Slice {slice_index}"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return {
        "output_path":
            str(output_path),

        "slice_index":
            slice_index,
    }