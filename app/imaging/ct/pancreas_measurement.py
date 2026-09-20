"""
Physical measurement utilities for pancreas CT segmentation.

Labels:
    0 = background
    1 = pancreas
    2 = cancer

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PancreasMeasurementError(Exception):
    pass


@dataclass
class PancreasMeasurement:

    pancreas_voxels: int

    cancer_voxels: int

    total_pancreas_region_voxels: int

    pancreas_volume_mm3: float

    cancer_volume_mm3: float

    total_pancreas_region_volume_mm3: float

    pancreas_volume_cm3: float

    cancer_volume_cm3: float

    total_pancreas_region_volume_cm3: float

    cancer_to_pancreas_ratio: float

    cancer_percentage_of_pancreas: float

    voxel_volume_mm3: float

    spacing_mm: tuple[float, float, float]

    def to_dict(self):

        return {
            "pancreas_voxels":
                self.pancreas_voxels,

            "cancer_voxels":
                self.cancer_voxels,

            "total_pancreas_region_voxels":
                self.total_pancreas_region_voxels,

            "pancreas_volume_mm3":
                round(
                    self.pancreas_volume_mm3,
                    3,
                ),

            "cancer_volume_mm3":
                round(
                    self.cancer_volume_mm3,
                    3,
                ),

            "total_pancreas_region_volume_mm3":
                round(
                    self.total_pancreas_region_volume_mm3,
                    3,
                ),

            "pancreas_volume_cm3":
                round(
                    self.pancreas_volume_cm3,
                    3,
                ),

            "cancer_volume_cm3":
                round(
                    self.cancer_volume_cm3,
                    3,
                ),

            "total_pancreas_region_volume_cm3":
                round(
                    self.total_pancreas_region_volume_cm3,
                    3,
                ),

            "cancer_to_pancreas_ratio":
                round(
                    self.cancer_to_pancreas_ratio,
                    6,
                ),

            "cancer_percentage_of_pancreas":
                round(
                    self.cancer_percentage_of_pancreas,
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


def measure_pancreas_segmentation(
    segmentation_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> PancreasMeasurement:

    if segmentation_mask.ndim != 3:

        raise PancreasMeasurementError(
            f"Expected 3D segmentation mask, "
            f"got {segmentation_mask.shape}"
        )

    if len(spacing_mm) != 3:

        raise PancreasMeasurementError(
            "spacing_mm must contain "
            "(z, y, x) spacing."
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
        {0, 1, 2}
    )

    if unexpected:

        raise PancreasMeasurementError(
            f"Unexpected segmentation labels: "
            f"{sorted(unexpected)}"
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

        raise PancreasMeasurementError(
            f"Invalid voxel spacing: {spacing}"
        )

    voxel_volume_mm3 = float(
        spacing[0]
        *
        spacing[1]
        *
        spacing[2]
    )

    pancreas_voxels = int(
        np.count_nonzero(
            segmentation_mask == 1
        )
    )

    cancer_voxels = int(
        np.count_nonzero(
            segmentation_mask == 2
        )
    )

    total_voxels = (
        pancreas_voxels
        +
        cancer_voxels
    )

    pancreas_volume_mm3 = (
        pancreas_voxels
        *
        voxel_volume_mm3
    )

    cancer_volume_mm3 = (
        cancer_voxels
        *
        voxel_volume_mm3
    )

    total_volume_mm3 = (
        total_voxels
        *
        voxel_volume_mm3
    )

    pancreas_volume_cm3 = (
        pancreas_volume_mm3
        /
        1000.0
    )

    cancer_volume_cm3 = (
        cancer_volume_mm3
        /
        1000.0
    )

    total_volume_cm3 = (
        total_volume_mm3
        /
        1000.0
    )

    if total_voxels > 0:

        cancer_ratio = (
            cancer_voxels
            /
            total_voxels
        )

        cancer_percentage = (
            cancer_ratio
            *
            100.0
        )

    else:

        cancer_ratio = 0.0
        cancer_percentage = 0.0

    return PancreasMeasurement(

        pancreas_voxels=
            pancreas_voxels,

        cancer_voxels=
            cancer_voxels,

        total_pancreas_region_voxels=
            total_voxels,

        pancreas_volume_mm3=
            pancreas_volume_mm3,

        cancer_volume_mm3=
            cancer_volume_mm3,

        total_pancreas_region_volume_mm3=
            total_volume_mm3,

        pancreas_volume_cm3=
            pancreas_volume_cm3,

        cancer_volume_cm3=
            cancer_volume_cm3,

        total_pancreas_region_volume_cm3=
            total_volume_cm3,

        cancer_to_pancreas_ratio=
            cancer_ratio,

        cancer_percentage_of_pancreas=
            cancer_percentage,

        voxel_volume_mm3=
            voxel_volume_mm3,

        spacing_mm=
            spacing,
    )