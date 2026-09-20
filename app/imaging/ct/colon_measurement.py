"""
Colon tumor physical measurement.

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ColonMeasurementError(Exception):
    pass


@dataclass
class ColonMeasurement:

    tumor_voxels: int

    tumor_volume_mm3: float

    tumor_volume_cm3: float

    voxel_volume_mm3: float

    spacing_mm: tuple[float, float, float]

    def to_dict(self):

        return {
            "tumor_voxels":
                self.tumor_voxels,

            "tumor_volume_mm3":
                round(
                    self.tumor_volume_mm3,
                    3,
                ),

            "tumor_volume_cm3":
                round(
                    self.tumor_volume_cm3,
                    3,
                ),

            "voxel_volume_mm3":
                round(
                    self.voxel_volume_mm3,
                    6,
                ),

            "spacing_mm":
                self.spacing_mm,
        }


def measure_colon_segmentation(
    segmentation_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> ColonMeasurement:

    if segmentation_mask.ndim != 3:

        raise ColonMeasurementError(
            f"Expected 3D mask, got {segmentation_mask.shape}"
        )

    if len(spacing_mm) != 3:

        raise ColonMeasurementError(
            "spacing_mm must be (z, y, x)"
        )

    unique_labels = set(
        int(x)
        for x
        in np.unique(
            segmentation_mask
        )
    )

    unexpected = (
        unique_labels
        -
        {0, 1}
    )

    if unexpected:

        raise ColonMeasurementError(
            f"Unexpected labels: {sorted(unexpected)}"
        )

    spacing = tuple(
        float(x)
        for x
        in spacing_mm
    )

    if any(
        x <= 0
        for x
        in spacing
    ):

        raise ColonMeasurementError(
            f"Invalid spacing: {spacing}"
        )

    voxel_volume_mm3 = float(
        spacing[0]
        *
        spacing[1]
        *
        spacing[2]
    )

    tumor_voxels = int(
        np.count_nonzero(
            segmentation_mask
            ==
            1
        )
    )

    tumor_volume_mm3 = (
        tumor_voxels
        *
        voxel_volume_mm3
    )

    tumor_volume_cm3 = (
        tumor_volume_mm3
        /
        1000.0
    )

    return ColonMeasurement(

        tumor_voxels=
            tumor_voxels,

        tumor_volume_mm3=
            tumor_volume_mm3,

        tumor_volume_cm3=
            tumor_volume_cm3,

        voxel_volume_mm3=
            voxel_volume_mm3,

        spacing_mm=
            spacing,
    )