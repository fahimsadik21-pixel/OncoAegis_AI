"""
Generic Liver CT DICOM I/O

Purpose:
- Read an uploaded/received CT DICOM series
- Sort slices correctly
- Convert pixel values to Hounsfield Units
- Preserve voxel spacing and basic metadata
- Build a 3D CT volume for liver inference

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


class LiverDicomIOError(Exception):
    pass


@dataclass
class LiverDicomVolume:

    volume: np.ndarray
    spacing_mm: tuple[float, float, float]

    patient_id: str | None
    study_instance_uid: str | None
    series_instance_uid: str | None

    modality: str
    slice_count: int

    rows: int
    columns: int

    source_directory: str

    def to_dict(self):

        return {
            "spacing_mm": self.spacing_mm,
            "patient_id": self.patient_id,
            "study_instance_uid": self.study_instance_uid,
            "series_instance_uid": self.series_instance_uid,
            "modality": self.modality,
            "slice_count": self.slice_count,
            "rows": self.rows,
            "columns": self.columns,
            "source_directory": self.source_directory,
        }


def _safe_float(
    value: Any,
    default: float,
) -> float:

    try:
        return float(value)
    except Exception:
        return default


def _read_possible_dicom(
    path: Path,
):

    try:

        ds = pydicom.dcmread(
            str(path),
            force=True,
        )

        if not hasattr(
            ds,
            "PixelData",
        ):
            return None

        if not hasattr(
            ds,
            "Rows",
        ):
            return None

        if not hasattr(
            ds,
            "Columns",
        ):
            return None

        return ds

    except Exception:
        return None


def _slice_position(
    ds,
    fallback_index: int,
):

    image_position = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    if (
        image_position is not None
        and len(image_position) >= 3
    ):

        try:
            return (
                0,
                float(
                    image_position[2]
                ),
            )
        except Exception:
            pass

    instance_number = getattr(
        ds,
        "InstanceNumber",
        None,
    )

    if instance_number is not None:

        try:
            return (
                1,
                float(
                    instance_number
                ),
            )
        except Exception:
            pass

    slice_location = getattr(
        ds,
        "SliceLocation",
        None,
    )

    if slice_location is not None:

        try:
            return (
                2,
                float(
                    slice_location
                ),
            )
        except Exception:
            pass

    return (
        3,
        float(
            fallback_index
        ),
    )


def _extract_spacing(
    slices,
) -> tuple[float, float, float]:

    first_ds = slices[0][1]

    pixel_spacing = getattr(
        first_ds,
        "PixelSpacing",
        None,
    )

    if (
        pixel_spacing is not None
        and len(pixel_spacing) >= 2
    ):

        row_spacing = _safe_float(
            pixel_spacing[0],
            1.0,
        )

        column_spacing = _safe_float(
            pixel_spacing[1],
            1.0,
        )

    else:

        row_spacing = 1.0
        column_spacing = 1.0

    z_spacing = None

    if len(slices) >= 2:

        z_positions = []

        for _, ds, _ in slices:

            image_position = getattr(
                ds,
                "ImagePositionPatient",
                None,
            )

            if (
                image_position is not None
                and len(image_position) >= 3
            ):

                try:
                    z_positions.append(
                        float(
                            image_position[2]
                        )
                    )
                except Exception:
                    pass

        if len(z_positions) >= 2:

            differences = np.abs(
                np.diff(
                    z_positions
                )
            )

            differences = differences[
                differences > 0
            ]

            if len(differences) > 0:

                z_spacing = float(
                    np.median(
                        differences
                    )
                )

    if z_spacing is None:

        z_spacing = _safe_float(
            getattr(
                first_ds,
                "SpacingBetweenSlices",
                getattr(
                    first_ds,
                    "SliceThickness",
                    1.0,
                ),
            ),
            1.0,
        )

    return (
        z_spacing,
        row_spacing,
        column_spacing,
    )


def load_liver_ct_dicom_series(
    directory: str | Path,
) -> LiverDicomVolume:

    directory = Path(
        directory
    )

    if not directory.exists():

        raise LiverDicomIOError(
            f"DICOM path does not exist: {directory}"
        )

    if not directory.is_dir():

        raise LiverDicomIOError(
            "Expected a directory containing a CT DICOM series."
        )

    candidates = [
        path
        for path in directory.rglob("*")
        if path.is_file()
    ]

    if not candidates:

        raise LiverDicomIOError(
            f"No files found in: {directory}"
        )

    slices = []

    for index, path in enumerate(
        candidates
    ):

        ds = _read_possible_dicom(
            path
        )

        if ds is None:
            continue

        modality = str(
            getattr(
                ds,
                "Modality",
                "",
            )
        ).upper()

        if modality and modality != "CT":
            continue

        slices.append(
            (
                path,
                ds,
                index,
            )
        )

    if not slices:

        raise LiverDicomIOError(
            "No readable CT DICOM slices were found."
        )

    series_groups = {}

    for item in slices:

        ds = item[1]

        series_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "unknown-series",
            )
        )

        series_groups.setdefault(
            series_uid,
            [],
        ).append(
            item
        )

    selected_series_uid = max(
        series_groups,
        key=lambda uid: len(
            series_groups[uid]
        ),
    )

    slices = series_groups[
        selected_series_uid
    ]

    slices.sort(
        key=lambda item: _slice_position(
            item[1],
            item[2],
        )
    )

    first_ds = slices[0][1]

    rows = int(
        first_ds.Rows
    )

    columns = int(
        first_ds.Columns
    )

    volume_slices = []

    for path, ds, _ in slices:

        pixel = ds.pixel_array.astype(
            np.float32
        )

        if pixel.shape != (
            rows,
            columns,
        ):

            raise LiverDicomIOError(
                f"Inconsistent DICOM matrix size in {path}: "
                f"{pixel.shape} != {(rows, columns)}"
            )

        slope = _safe_float(
            getattr(
                ds,
                "RescaleSlope",
                1.0,
            ),
            1.0,
        )

        intercept = _safe_float(
            getattr(
                ds,
                "RescaleIntercept",
                0.0,
            ),
            0.0,
        )

        pixel = (
            pixel * slope
            +
            intercept
        )

        volume_slices.append(
            pixel
        )

    volume = np.stack(
        volume_slices,
        axis=0,
    ).astype(
        np.float32
    )

    spacing_mm = _extract_spacing(
        slices
    )

    return LiverDicomVolume(

        volume=volume,

        spacing_mm=spacing_mm,

        patient_id=(
            str(
                getattr(
                    first_ds,
                    "PatientID",
                    "",
                )
            )
            or None
        ),

        study_instance_uid=(
            str(
                getattr(
                    first_ds,
                    "StudyInstanceUID",
                    "",
                )
            )
            or None
        ),

        series_instance_uid=(
            str(
                getattr(
                    first_ds,
                    "SeriesInstanceUID",
                    "",
                )
            )
            or None
        ),

        modality="CT",

        slice_count=int(
            volume.shape[0]
        ),

        rows=int(
            volume.shape[1]
        ),

        columns=int(
            volume.shape[2]
        ),

        source_directory=str(
            directory
        ),
    )


if __name__ == "__main__":

    print(
        "Liver CT DICOM I/O module ready."
    )