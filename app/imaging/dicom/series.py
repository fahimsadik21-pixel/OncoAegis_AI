"""Validation, spatial ordering and 3D construction for DICOM series."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class DicomSeriesError(ValueError):
    """Raised when a collection cannot form one safe DICOM image series."""


def _as_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DicomSeriesError(
            f"Invalid {field_name} value: {value!r}"
        ) from exc

    if not isfinite(number):
        raise DicomSeriesError(f"Invalid {field_name} value: {value!r}")
    return number


def _as_float_tuple(
    value: Any,
    *,
    expected_length: int,
    field_name: str,
) -> tuple[float, ...] | None:
    if value is None:
        return None

    try:
        values = tuple(
            _as_float(item, field_name)
            for item in value
        )
    except TypeError as exc:
        raise DicomSeriesError(
            f"Invalid {field_name}: expected a sequence"
        ) from exc

    if len(values) != expected_length:
        raise DicomSeriesError(
            f"Invalid {field_name}: expected {expected_length} values"
        )
    return values


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    return _as_float(value, field_name)


@dataclass
class DicomSlice:
    """A loaded DICOM image slice plus the geometry used for ordering."""

    dataset: Any
    source: str
    series_instance_uid: str
    study_instance_uid: str | None
    sop_instance_uid: str | None
    modality: str
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    orientation: tuple[float, ...] | None
    image_position: tuple[float, ...] | None
    projected_position: float | None
    instance_number: int | None
    slice_location: float | None
    slice_thickness: float | None


@dataclass
class DicomSeries:
    """Ready-to-use 3D volume and the validated series context."""

    volume: np.ndarray
    slices: list[DicomSlice]
    modality: str
    series_instance_uid: str
    study_instance_uid: str | None
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_spacing: float | None
    slice_thickness: float | None
    orientation: tuple[float, ...] | None
    ordering_method: str
    warnings: list[str]
    safe_metadata: dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.volume.shape)


def _normal_from_orientation(
    orientation: tuple[float, ...] | None,
) -> np.ndarray | None:
    if orientation is None:
        return None

    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row, column)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise DicomSeriesError(
            "ImageOrientationPatient contains parallel direction vectors"
        )
    return normal / norm


def _slice_projection(
    image_position: tuple[float, ...] | None,
    orientation: tuple[float, ...] | None,
) -> float | None:
    if image_position is None or orientation is None:
        return None
    normal = _normal_from_orientation(orientation)
    assert normal is not None
    return float(np.dot(np.asarray(image_position), normal))


def make_slice_record(dataset: Any, source: str) -> DicomSlice:
    """Extract the technical fields needed for validation and ordering."""

    series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
    if not series_uid:
        raise DicomSeriesError(f"Missing SeriesInstanceUID in {source}")

    modality = str(getattr(dataset, "Modality", "")).strip().upper()
    if not modality:
        raise DicomSeriesError(f"Missing Modality in {source}")

    try:
        rows = int(getattr(dataset, "Rows"))
        columns = int(getattr(dataset, "Columns"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise DicomSeriesError(
            f"Missing or invalid image dimensions in {source}"
        ) from exc

    if rows <= 0 or columns <= 0:
        raise DicomSeriesError(f"Invalid image dimensions in {source}")

    pixel_spacing = _as_float_tuple(
        getattr(dataset, "PixelSpacing", None),
        expected_length=2,
        field_name="PixelSpacing",
    )
    if pixel_spacing is None or any(value <= 0 for value in pixel_spacing):
        raise DicomSeriesError(
            f"Missing or invalid PixelSpacing in {source}"
        )

    orientation = _as_float_tuple(
        getattr(dataset, "ImageOrientationPatient", None),
        expected_length=6,
        field_name="ImageOrientationPatient",
    )
    image_position = _as_float_tuple(
        getattr(dataset, "ImagePositionPatient", None),
        expected_length=3,
        field_name="ImagePositionPatient",
    )

    return DicomSlice(
        dataset=dataset,
        source=source,
        series_instance_uid=series_uid,
        study_instance_uid=(
            str(getattr(dataset, "StudyInstanceUID", "")).strip() or None
        ),
        sop_instance_uid=(
            str(getattr(dataset, "SOPInstanceUID", "")).strip() or None
        ),
        modality=modality,
        rows=rows,
        columns=columns,
        pixel_spacing=(pixel_spacing[0], pixel_spacing[1]),
        orientation=orientation,
        image_position=image_position,
        projected_position=_slice_projection(image_position, orientation),
        instance_number=_optional_int(
            getattr(dataset, "InstanceNumber", None)
        ),
        slice_location=_optional_float(
            getattr(dataset, "SliceLocation", None),
            "SliceLocation",
        ),
        slice_thickness=_optional_float(
            getattr(dataset, "SliceThickness", None),
            "SliceThickness",
        ),
    )


def _all_close(
    values: Iterable[tuple[float, ...]],
    *,
    tolerance: float = 1e-4,
) -> bool:
    value_list = list(values)
    if not value_list:
        return True
    reference = np.asarray(value_list[0], dtype=np.float64)
    return all(
        np.allclose(reference, np.asarray(value), atol=tolerance, rtol=0)
        for value in value_list[1:]
    )


def _sort_slices(
    slices: list[DicomSlice],
) -> tuple[list[DicomSlice], str, list[str]]:
    warnings: list[str] = []

    if all(item.projected_position is not None for item in slices):
        ordered = sorted(
            slices,
            key=lambda item: float(item.projected_position),
        )
        ordering_method = "image_position_patient_projection"
    elif all(item.instance_number is not None for item in slices):
        ordered = sorted(
            slices,
            key=lambda item: int(item.instance_number),
        )
        ordering_method = "instance_number"
        warnings.append(
            "ImagePositionPatient/Orientation unavailable; used InstanceNumber"
        )
    elif all(item.slice_location is not None for item in slices):
        ordered = sorted(
            slices,
            key=lambda item: float(item.slice_location),
        )
        ordering_method = "slice_location"
        warnings.append(
            "ImagePositionPatient/Orientation unavailable; used SliceLocation"
        )
    else:
        ordered = sorted(
            slices,
            key=lambda item: Path(item.source).name.lower(),
        )
        ordering_method = "filename"
        warnings.append(
            "Spatial ordering metadata unavailable; used filename order"
        )

    positions = [
        item.projected_position
        for item in ordered
        if item.projected_position is not None
    ]
    if len(positions) == len(ordered) and len(positions) > 2:
        differences = np.diff(np.asarray(positions, dtype=np.float64))
        if np.any(differences <= 0):
            raise DicomSeriesError(
                "Duplicate or non-increasing slice positions detected"
            )

        spacing = float(np.median(differences))
        if not np.allclose(
            differences,
            spacing,
            atol=max(0.01, spacing * 0.05),
            rtol=0,
        ):
            warnings.append(
                "Slice spacing is non-uniform; median spacing will be reported"
            )

    return ordered, ordering_method, warnings


def validate_and_order_slices(
    slices: list[DicomSlice],
    *,
    require_ct: bool = True,
) -> tuple[list[DicomSlice], str, list[str]]:
    """Validate that slices can safely be treated as one image series."""

    if not slices:
        raise DicomSeriesError("No image slices were provided")

    modalities = {item.modality for item in slices}
    if require_ct and modalities != {"CT"}:
        raise DicomSeriesError(
            "CT series required; received modalities: "
            + ", ".join(sorted(modalities))
        )
    if len(modalities) != 1:
        raise DicomSeriesError(
            "Mixed DICOM modalities cannot form one series: "
            + ", ".join(sorted(modalities))
        )

    series_uids = {item.series_instance_uid for item in slices}
    if len(series_uids) != 1:
        raise DicomSeriesError(
            "Multiple SeriesInstanceUID values were provided; "
            "upload one series at a time"
        )

    study_uids = {
        item.study_instance_uid
        for item in slices
        if item.study_instance_uid is not None
    }
    if len(study_uids) > 1:
        raise DicomSeriesError(
            "Multiple StudyInstanceUID values were provided; "
            "upload one study series at a time"
        )

    dimensions = {(item.rows, item.columns) for item in slices}
    if len(dimensions) != 1:
        raise DicomSeriesError(
            "All slices must have identical Rows and Columns"
        )

    if not _all_close(item.pixel_spacing for item in slices):
        raise DicomSeriesError("All slices must have identical PixelSpacing")

    orientations = [
        item.orientation
        for item in slices
        if item.orientation is not None
    ]
    if orientations and len(orientations) != len(slices):
        raise DicomSeriesError(
            "ImageOrientationPatient is missing from some slices"
        )
    if orientations and not _all_close(orientations):
        raise DicomSeriesError(
            "All slices must have the same ImageOrientationPatient"
        )

    sop_uids = [
        item.sop_instance_uid
        for item in slices
        if item.sop_instance_uid is not None
    ]
    if len(sop_uids) != len(set(sop_uids)):
        raise DicomSeriesError("Duplicate SOPInstanceUID values detected")

    return _sort_slices(slices)


def build_volume(slices: list[DicomSlice]) -> np.ndarray:
    """Stack slices and apply DICOM rescale slope/intercept to float32 data."""

    pixel_arrays: list[np.ndarray] = []
    for item in slices:
        try:
            pixels = np.asarray(item.dataset.pixel_array)
        except Exception as exc:
            raise DicomSeriesError(
                f"Could not decode pixel data in {item.source}: {exc}"
            ) from exc

        if pixels.ndim != 2:
            raise DicomSeriesError(
                f"Expected a 2D CT slice in {item.source}; "
                f"received shape {pixels.shape}"
            )
        if pixels.shape != (item.rows, item.columns):
            raise DicomSeriesError(
                f"Pixel array shape mismatch in {item.source}: "
                f"expected {(item.rows, item.columns)}, got {pixels.shape}"
            )

        slope = _as_float(
            getattr(item.dataset, "RescaleSlope", 1.0),
            "RescaleSlope",
        )
        intercept = _as_float(
            getattr(item.dataset, "RescaleIntercept", 0.0),
            "RescaleIntercept",
        )
        pixel_arrays.append(
            pixels.astype(np.float32, copy=False) * slope + intercept
        )

    return np.stack(pixel_arrays, axis=0).astype(np.float32, copy=False)


def calculate_slice_spacing(
    slices: list[DicomSlice],
) -> float | None:
    positions = [
        item.projected_position
        for item in slices
        if item.projected_position is not None
    ]
    if len(positions) >= 2:
        differences = np.abs(np.diff(np.asarray(positions)))
        positive = differences[differences > 1e-6]
        if positive.size:
            return float(np.median(positive))

    thicknesses = [
        item.slice_thickness
        for item in slices
        if item.slice_thickness is not None and item.slice_thickness > 0
    ]
    return float(np.median(thicknesses)) if thicknesses else None
