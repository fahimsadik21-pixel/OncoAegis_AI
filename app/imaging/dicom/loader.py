"""Load one validated DICOM image series from disk or uploaded bytes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator

import pydicom

from app.imaging.dicom.metadata import build_safe_metadata
from app.imaging.dicom.series import (
    DicomSeries,
    DicomSeriesError,
    build_volume,
    calculate_slice_spacing,
    make_slice_record,
    validate_and_order_slices,
)


def _iter_paths(source: str | Path) -> Iterator[Path]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"DICOM source does not exist: {path}")
    if path.is_file():
        yield path
        return
    yield from (item for item in path.rglob("*") if item.is_file())


def _read_dataset(data: bytes, source: str):
    try:
        return pydicom.dcmread(
            BytesIO(data),
            force=False,
        )
    except (pydicom.errors.InvalidDicomError, EOFError) as exc:
        raise DicomSeriesError(f"Not a readable DICOM file: {source}") from exc


def _read_path(path: Path):
    try:
        return pydicom.dcmread(str(path), force=False)
    except (pydicom.errors.InvalidDicomError, EOFError) as exc:
        raise DicomSeriesError(f"Not a readable DICOM file: {path}") from exc


def _build_series(datasets: Iterable[tuple[str, object]]) -> DicomSeries:
    records = []
    skipped: list[str] = []

    for source, dataset in datasets:
        if "PixelData" not in dataset:
            skipped.append(f"{source}: no PixelData")
            continue
        try:
            records.append(make_slice_record(dataset, source))
        except DicomSeriesError as exc:
            skipped.append(f"{source}: {exc}")

    if not records:
        detail = f" Details: {skipped[:3]}" if skipped else ""
        raise DicomSeriesError(f"No usable DICOM image slices found.{detail}")

    # Do not silently merge unrelated acquisitions. The caller can upload a
    # single series or use the series UID to filter before calling this layer.
    series_uids = {item.series_instance_uid for item in records}
    if len(series_uids) != 1:
        counts = {
            uid: sum(item.series_instance_uid == uid for item in records)
            for uid in sorted(series_uids)
        }
        raise DicomSeriesError(
            "Input contains multiple DICOM series; upload one series at a time. "
            f"Slice counts by series: {counts}"
        )

    ordered, ordering_method, warnings = validate_and_order_slices(records)
    volume = build_volume(ordered)
    first = ordered[0]
    slice_spacing = calculate_slice_spacing(ordered)
    thicknesses = [
        item.slice_thickness
        for item in ordered
        if item.slice_thickness is not None and item.slice_thickness > 0
    ]
    slice_thickness = (
        float(thicknesses[0]) if thicknesses else None
    )

    if skipped:
        warnings.append(
            f"Skipped {len(skipped)} non-image or invalid file(s)"
        )

    safe_metadata = build_safe_metadata(
        (item.dataset for item in ordered),
        slice_count=len(ordered),
        rows=first.rows,
        columns=first.columns,
        pixel_spacing=first.pixel_spacing,
        slice_spacing=slice_spacing,
        slice_thickness=slice_thickness,
        orientation=first.orientation,
        ordering_method=ordering_method,
        warnings=warnings,
    )

    result = DicomSeries(
        volume=volume,
        slices=ordered,
        modality=first.modality,
        series_instance_uid=first.series_instance_uid,
        study_instance_uid=first.study_instance_uid,
        rows=first.rows,
        columns=first.columns,
        pixel_spacing=first.pixel_spacing,
        slice_spacing=slice_spacing,
        slice_thickness=slice_thickness,
        orientation=first.orientation,
        ordering_method=ordering_method,
        warnings=warnings,
        safe_metadata=safe_metadata,
    )
    return result


def load_dicom_series(
    source: str | Path,
) -> DicomSeries:
    """Load and validate all image slices under a file or directory."""

    datasets: list[tuple[str, object]] = []
    for path in _iter_paths(source):
        try:
            dataset = _read_path(path)
        except DicomSeriesError:
            # A DICOM directory commonly contains README/sidecar files. They
            # are ignored and a warning is added only if an image series loads.
            continue
        datasets.append((str(path), dataset))

    return _build_series(datasets)


def load_dicom_series_from_bytes(
    files: Iterable[tuple[str, bytes]],
) -> DicomSeries:
    """Load a DICOM series from ``(filename, bytes)`` upload entries."""

    datasets: list[tuple[str, object]] = []
    for filename, data in files:
        try:
            dataset = _read_dataset(data, filename)
        except DicomSeriesError:
            continue
        datasets.append((filename, dataset))

    return _build_series(datasets)
