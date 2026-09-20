"""
C-NMC microscopy classification visualization.

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_cnmc_visualization(
    image: np.ndarray,
    prediction,
    output_path: str | Path,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image
    )

    plt.title(
        (
            f"{prediction.predicted_class}\n"
            f"ALL probability: "
            f"{prediction.all_probability:.3f} | "
            f"HEM probability: "
            f"{prediction.hem_probability:.3f}"
        )
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