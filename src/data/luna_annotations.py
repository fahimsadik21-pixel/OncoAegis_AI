"""LUNA16 nodule-annotation utilities.

LUNA16 annotations identify pulmonary nodules and their physical diameter.
They do not provide benign/malignant pathology labels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk


@dataclass(frozen=True)
class LUNANoduleAnnotation:
    case_id: str
    coord_x: float
    coord_y: float
    coord_z: float
    diameter_mm: float


def load_luna_annotations(
    csv_path: str | Path,
    *,
    case_ids: set[str] | None = None,
) -> dict[str, tuple[LUNANoduleAnnotation, ...]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"LUNA16 annotations file not found: {path}")
    grouped: dict[str, list[LUNANoduleAnnotation]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case_id = row["seriesuid"].strip()
            if case_ids is not None and case_id not in case_ids:
                continue
            grouped.setdefault(case_id, []).append(
                LUNANoduleAnnotation(
                    case_id=case_id,
                    coord_x=float(row["coordX"]),
                    coord_y=float(row["coordY"]),
                    coord_z=float(row["coordZ"]),
                    diameter_mm=float(row["diameter_mm"]),
                )
            )
    return {case_id: tuple(values) for case_id, values in grouped.items()}


def annotation_slice_ids(
    ct_path: str | Path,
    annotations: tuple[LUNANoduleAnnotation, ...],
) -> set[int]:
    """Map physical nodule coordinates to z-slices, including diameter span."""

    if not annotations:
        return set()
    image = sitk.ReadImage(str(ct_path))
    size_z = image.GetSize()[2]
    spacing_z = float(image.GetSpacing()[2])
    slice_ids: set[int] = set()
    for annotation in annotations:
        continuous_index = image.TransformPhysicalPointToContinuousIndex(
            (annotation.coord_x, annotation.coord_y, annotation.coord_z)
        )
        center = int(round(continuous_index[2]))
        radius = max(0, int(round(annotation.diameter_mm / (2.0 * spacing_z))))
        for slice_id in range(center - radius, center + radius + 1):
            if 0 <= slice_id < size_z:
                slice_ids.add(slice_id)
    return slice_ids


def annotation_voxel_centers(
    ct_path: str | Path,
    annotations: tuple[LUNANoduleAnnotation, ...],
) -> list[tuple[int, int, int, float]]:
    """Return annotation centers as x, y, z, diameter_mm voxel values."""

    if not annotations:
        return []
    image = sitk.ReadImage(str(ct_path))
    size_x, size_y, size_z = image.GetSize()
    centers: list[tuple[int, int, int, float]] = []
    for annotation in annotations:
        continuous_index = image.TransformPhysicalPointToContinuousIndex(
            (annotation.coord_x, annotation.coord_y, annotation.coord_z)
        )
        x = int(round(continuous_index[0]))
        y = int(round(continuous_index[1]))
        z = int(round(continuous_index[2]))
        if 0 <= x < size_x and 0 <= y < size_y and 0 <= z < size_z:
            centers.append((x, y, z, annotation.diameter_mm))
    return centers
