"""
Skin lesion visualization.

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class SkinVisualizationError(Exception):
    pass


def save_skin_overlay(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    output_path: str | Path,
):

    if image.ndim != 3:

        raise SkinVisualizationError(
            f"Expected RGB image, got {image.shape}"
        )

    if segmentation_mask.ndim != 2:

        raise SkinVisualizationError(
            f"Expected 2D mask, got "
            f"{segmentation_mask.shape}"
        )

    if image.shape[:2] != segmentation_mask.shape:

        raise SkinVisualizationError(
            "Image and segmentation mask "
            "shapes do not match."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(
            8,
            8,
        )
    )

    plt.imshow(
        image
    )

    overlay = np.ma.masked_where(
        segmentation_mask != 1,
        segmentation_mask,
    )

    if np.any(
        segmentation_mask == 1
    ):

        plt.imshow(
            overlay,
            cmap="autumn",
            alpha=0.45,
        )

    plt.title(
        "Skin Lesion Research Segmentation"
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
            str(
                output_path
            )
    }