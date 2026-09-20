"""
Colon CT visualization.

Mask:
    0 = background
    1 = colon tumor model region

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ColonVisualizationError(Exception):
    pass


def _select_slice(
    segmentation_mask: np.ndarray,
) -> int:

    counts = np.sum(
        segmentation_mask == 1,
        axis=(1, 2),
    )

    if np.max(counts) > 0:
        return int(
            np.argmax(counts)
        )

    return int(
        segmentation_mask.shape[0]
        // 2
    )


def save_colon_overlay(
    volume: np.ndarray,
    segmentation_mask: np.ndarray,
    output_path: str | Path,
):

    if volume.ndim != 3:

        raise ColonVisualizationError(
            f"Expected 3D CT volume, got {volume.shape}"
        )

    if segmentation_mask.ndim != 3:

        raise ColonVisualizationError(
            f"Expected 3D mask, got {segmentation_mask.shape}"
        )

    if volume.shape != segmentation_mask.shape:

        raise ColonVisualizationError(
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
            1,
        )
    )

    high = float(
        np.percentile(
            image_slice,
            99,
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

    tumor_overlay = np.ma.masked_where(
        mask_slice != 1,
        mask_slice,
    )

    if np.any(
        mask_slice == 1
    ):

        plt.imshow(
            tumor_overlay,
            cmap="autumn",
            alpha=0.60,
        )

    plt.title(
        f"Colon CT Research Overlay — Slice {slice_index}"
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