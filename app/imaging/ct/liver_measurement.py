"""
Liver CT Segmentation Measurement Utilities

Measures:
- Liver parenchyma voxels
- Tumor voxels
- Total segmented liver region
- Physical liver/tumor volume using DICOM voxel spacing
- Tumor-to-liver volume ratio

Segmentation labels:
    0 = background
    1 = liver
    2 = tumor

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class LiverMeasurementError(Exception):
    pass


@dataclass
class LiverMeasurementResult:

    liver_parenchyma_voxels: int
    tumor_voxels: int
    total_liver_region_voxels: int

    liver_parenchyma_volume_mm3: float
    tumor_volume_mm3: float
    total_liver_region_volume_mm3: float

    liver_parenchyma_volume_cm3: float
    tumor_volume_cm3: float
    total_liver_region_volume_cm3: float

    tumor_to_liver_ratio: float
    tumor_percentage_of_liver: float

    voxel_volume_mm3: float

    spacing_mm: tuple[float, float, float]

    def to_dict(self):

        return {
            "liver_parenchyma_voxels":
                self.liver_parenchyma_voxels,

            "tumor_voxels":
                self.tumor_voxels,

            "total_liver_region_voxels":
                self.total_liver_region_voxels,

            "liver_parenchyma_volume_mm3":
                round(
                    self.liver_parenchyma_volume_mm3,
                    3,
                ),

            "tumor_volume_mm3":
                round(
                    self.tumor_volume_mm3,
                    3,
                ),

            "total_liver_region_volume_mm3":
                round(
                    self.total_liver_region_volume_mm3,
                    3,
                ),

            "liver_parenchyma_volume_cm3":
                round(
                    self.liver_parenchyma_volume_cm3,
                    3,
                ),

            "tumor_volume_cm3":
                round(
                    self.tumor_volume_cm3,
                    3,
                ),

            "total_liver_region_volume_cm3":
                round(
                    self.total_liver_region_volume_cm3,
                    3,
                ),

            "tumor_to_liver_ratio":
                round(
                    self.tumor_to_liver_ratio,
                    6,
                ),

            "tumor_percentage_of_liver":
                round(
                    self.tumor_percentage_of_liver,
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


def measure_liver_segmentation(
    segmentation_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> LiverMeasurementResult:

    if segmentation_mask.ndim != 3:

        raise LiverMeasurementError(
            f"Expected 3D segmentation mask, "
            f"got shape {segmentation_mask.shape}"
        )

    if len(spacing_mm) != 3:

        raise LiverMeasurementError(
            "spacing_mm must contain "
            "(z_spacing, row_spacing, column_spacing)"
        )

    z_spacing = float(
        spacing_mm[0]
    )

    row_spacing = float(
        spacing_mm[1]
    )

    column_spacing = float(
        spacing_mm[2]
    )

    if (
        z_spacing <= 0
        or row_spacing <= 0
        or column_spacing <= 0
    ):

        raise LiverMeasurementError(
            f"Invalid voxel spacing: {spacing_mm}"
        )

    valid_labels = np.unique(
        segmentation_mask
    )

    unexpected_labels = [
        int(value)
        for value in valid_labels
        if int(value) not in (0, 1, 2)
    ]

    if unexpected_labels:

        raise LiverMeasurementError(
            f"Unexpected segmentation labels: "
            f"{unexpected_labels}"
        )

    liver_parenchyma_voxels = int(
        np.count_nonzero(
            segmentation_mask == 1
        )
    )

    tumor_voxels = int(
        np.count_nonzero(
            segmentation_mask == 2
        )
    )

    total_liver_region_voxels = (
        liver_parenchyma_voxels
        +
        tumor_voxels
    )

    voxel_volume_mm3 = (
        z_spacing
        *
        row_spacing
        *
        column_spacing
    )

    liver_parenchyma_volume_mm3 = (
        liver_parenchyma_voxels
        *
        voxel_volume_mm3
    )

    tumor_volume_mm3 = (
        tumor_voxels
        *
        voxel_volume_mm3
    )

    total_liver_region_volume_mm3 = (
        total_liver_region_voxels
        *
        voxel_volume_mm3
    )

    liver_parenchyma_volume_cm3 = (
        liver_parenchyma_volume_mm3
        /
        1000.0
    )

    tumor_volume_cm3 = (
        tumor_volume_mm3
        /
        1000.0
    )

    total_liver_region_volume_cm3 = (
        total_liver_region_volume_mm3
        /
        1000.0
    )

    if total_liver_region_voxels > 0:

        tumor_to_liver_ratio = (
            tumor_voxels
            /
            total_liver_region_voxels
        )

    else:

        tumor_to_liver_ratio = 0.0

    tumor_percentage_of_liver = (
        tumor_to_liver_ratio
        *
        100.0
    )

    return LiverMeasurementResult(

        liver_parenchyma_voxels=
            liver_parenchyma_voxels,

        tumor_voxels=
            tumor_voxels,

        total_liver_region_voxels=
            total_liver_region_voxels,

        liver_parenchyma_volume_mm3=
            liver_parenchyma_volume_mm3,

        tumor_volume_mm3=
            tumor_volume_mm3,

        total_liver_region_volume_mm3=
            total_liver_region_volume_mm3,

        liver_parenchyma_volume_cm3=
            liver_parenchyma_volume_cm3,

        tumor_volume_cm3=
            tumor_volume_cm3,

        total_liver_region_volume_cm3=
            total_liver_region_volume_cm3,

        tumor_to_liver_ratio=
            tumor_to_liver_ratio,

        tumor_percentage_of_liver=
            tumor_percentage_of_liver,

        voxel_volume_mm3=
            voxel_volume_mm3,

        spacing_mm=(
            z_spacing,
            row_spacing,
            column_spacing,
        ),
    )


if __name__ == "__main__":

    print(
        "Liver measurement module ready."
    )