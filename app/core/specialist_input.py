"""Safe normalization of uploaded inputs for the universal specialist path."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.imaging.dicom.loader import (
    DicomSeriesError,
    load_dicom_series_from_bytes,
)
from app.registry.model_registry import (
    ModelSpec,
    get_model_registry,
    normalize_modality,
    normalize_organ,
)


class SpecialistInputError(ValueError):
    """Raised when uploaded data cannot satisfy a model input contract."""


# These limits protect the API process from accidental or malicious oversized
# uploads while still allowing ordinary clinical CT/MRI series.  They are
# input-contract limits, not model-quality claims.
MAX_SPECIALIST_FILES = 4096
MAX_SPECIALIST_FILE_BYTES = 256 * 1024 * 1024
MAX_SPECIALIST_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_VOLUME_ELEMENTS = 128_000_000
MAX_RASTER_PIXELS = 16_000_000
MAX_FLOW_EVENTS_PER_TUBE = 2_000_000

FLOW_COLUMNS = (
    "FS Lin",
    "SS Log",
    "FL1 Log",
    "FL2 Log",
    "FL3 Log",
    "FL4 Log",
    "FL5 Log",
)


@dataclass(frozen=True)
class UploadedSpecialistFile:
    name: str
    content: bytes


@dataclass(frozen=True)
class NormalizedSpecialistInput:
    data: Any
    metadata: dict[str, Any]


def parse_spacing_mm(value: str | None) -> tuple[float, float, float] | None:
    if value is None or not value.strip():
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise SpecialistInputError(
            "spacing_mm must contain exactly three comma-separated values"
        )
    try:
        spacing = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise SpecialistInputError(
            "spacing_mm must contain numeric values"
        ) from exc
    if any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise SpecialistInputError(
            "spacing_mm values must be positive and finite"
        )
    return spacing


def _suffix(name: str) -> str:
    lowered = str(name).lower()
    if lowered.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(str(name)).suffix.lower()


def _validate_uploaded_entries(entries: list[UploadedSpecialistFile]) -> None:
    if len(entries) > MAX_SPECIALIST_FILES:
        raise SpecialistInputError(
            f"Too many uploaded files; maximum is {MAX_SPECIALIST_FILES}"
        )

    total_bytes = 0
    for entry in entries:
        name = str(entry.name or "")
        content = entry.content
        if "\x00" in name:
            raise SpecialistInputError("Uploaded filename contains an invalid character")
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise SpecialistInputError("Uploaded file content must be bytes")
        size = len(content)
        if size == 0:
            raise SpecialistInputError("Uploaded file is empty")
        if size > MAX_SPECIALIST_FILE_BYTES:
            raise SpecialistInputError(
                f"Uploaded file exceeds the {MAX_SPECIALIST_FILE_BYTES} byte limit"
            )
        total_bytes += size

    if total_bytes > MAX_SPECIALIST_TOTAL_BYTES:
        raise SpecialistInputError(
            f"Total upload size exceeds the {MAX_SPECIALIST_TOTAL_BYTES} byte limit"
        )


def _validate_volume_array(array: np.ndarray, *, label: str) -> np.ndarray:
    if array.ndim not in (3, 4):
        raise SpecialistInputError(
            f"{label} must be a 3D or 4D numeric volume, got {array.shape}"
        )
    if any(int(size) <= 0 for size in array.shape):
        raise SpecialistInputError(f"{label} has an empty dimension")
    if int(array.size) > MAX_VOLUME_ELEMENTS:
        raise SpecialistInputError(
            f"{label} is too large ({int(array.size)} elements)"
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise SpecialistInputError(f"{label} must contain numeric values")
    if not np.isfinite(array).all():
        raise SpecialistInputError(f"{label} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _validate_model_volume_shape(spec: ModelSpec, volume: np.ndarray) -> None:
    if spec.modality == "CT" and volume.ndim != 3:
        raise SpecialistInputError(
            f"Model {spec.model_id} requires a 3D CT volume, got {volume.shape}"
        )
    if spec.modality == "MRI" and volume.ndim != 4:
        raise SpecialistInputError(
            f"Model {spec.model_id} requires a 4D multi-channel MRI volume, got {volume.shape}"
        )


def _single_file(files: list[UploadedSpecialistFile]) -> UploadedSpecialistFile:
    if len(files) != 1:
        raise SpecialistInputError("This specialist requires exactly one file")
    return files[0]


def _load_numpy(file: UploadedSpecialistFile) -> np.ndarray:
    try:
        value = np.load(BytesIO(file.content), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise SpecialistInputError(
            "Could not read NumPy input"
        ) from exc
    if not isinstance(value, np.ndarray):
        raise SpecialistInputError("Only a single NumPy array is supported")
    if value.dtype == object:
        raise SpecialistInputError("Object arrays are not accepted")
    return _validate_volume_array(np.asarray(value), label="NumPy input")


def _load_nifti(
    file: UploadedSpecialistFile,
    workspace: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    target = workspace / f"uploaded_volume{_suffix(file.name)}"
    target.write_bytes(file.content)
    try:
        image = nib.load(str(target))
        shape = tuple(int(value) for value in image.shape)
        if len(shape) not in (3, 4) or any(value <= 0 for value in shape):
            raise SpecialistInputError(
                f"NIfTI input must be a non-empty 3D or 4D volume, got {shape}"
            )
        if int(np.prod(shape)) > MAX_VOLUME_ELEMENTS:
            raise SpecialistInputError(
                f"NIfTI input is too large ({int(np.prod(shape))} elements)"
            )
        volume = np.asarray(image.get_fdata(), dtype=np.float32)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    except Exception as exc:
        raise SpecialistInputError(
            "Could not read NIfTI input"
        ) from exc
    volume = _validate_volume_array(volume, label="NIfTI input")
    if len(zooms) != 3 or any(value <= 0 or not np.isfinite(value) for value in zooms):
        raise SpecialistInputError("NIfTI spacing is missing or invalid")
    return volume, zooms


def _decode_rgb(file: UploadedSpecialistFile) -> np.ndarray:
    try:
        with Image.open(BytesIO(file.content)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_RASTER_PIXELS:
                raise SpecialistInputError(
                    f"Raster input is too large: {width}x{height}"
                )
            value = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise SpecialistInputError(
            "Could not read raster image"
        ) from exc
    if value.ndim != 3 or value.shape[2] != 3:
        raise SpecialistInputError("Raster input must decode to an RGB image")
    return value


def _raster_to_single_slice_volume(
    raster: np.ndarray,
    *,
    modality: str,
) -> np.ndarray:
    """Adapt an ordinary exported scan image to the 3D model contract.

    This is intentionally labelled as a single-slice approximation in the
    returned metadata.  It makes patient-held PNG/JPEG exports usable without
    pretending that one screenshot contains the full CT/MRI series.
    """

    grayscale = np.asarray(
        Image.fromarray(raster).convert("L").resize((64, 64)),
        dtype=np.float32,
    )
    grayscale /= 255.0
    volume = np.repeat(grayscale[:, :, np.newaxis], 64, axis=2)
    if modality == "MRI":
        return np.repeat(volume[:, :, :, np.newaxis], 4, axis=3)
    return np.moveaxis(volume, 2, 0)


def _validate_flow_csv(file: UploadedSpecialistFile) -> int:
    try:
        text = bytes(file.content).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SpecialistInputError("Flow-cytometry CSV is not UTF-8") from exc

    reader = csv.DictReader(text.splitlines())
    headers = tuple((header or "").strip() for header in (reader.fieldnames or ()))
    if headers != FLOW_COLUMNS:
        raise SpecialistInputError(
            f"Flow-cytometry CSV must contain columns: {', '.join(FLOW_COLUMNS)}"
        )

    events = 0
    for row in reader:
        events += 1
        if events > MAX_FLOW_EVENTS_PER_TUBE:
            raise SpecialistInputError(
                f"Flow-cytometry CSV exceeds {MAX_FLOW_EVENTS_PER_TUBE} events"
            )
        if None in row:
            raise SpecialistInputError(
                "Flow-cytometry CSV contains an extra field"
            )
        try:
            values = [float(row[column]) for column in FLOW_COLUMNS]
        except (TypeError, ValueError) as exc:
            raise SpecialistInputError(
                "Flow-cytometry CSV contains a non-numeric event"
            ) from exc
        if not np.isfinite(values).all():
            raise SpecialistInputError("Flow-cytometry CSV contains non-finite values")

    if events == 0:
        raise SpecialistInputError("Flow-cytometry CSV contains no events")
    return events


def _validate_declared_metadata(
    spec: ModelSpec,
    *,
    modality: str | None,
    organ: str | None,
) -> None:
    if modality and normalize_modality(modality) != spec.modality:
        raise SpecialistInputError(
            f"Declared modality does not match model {spec.model_id}"
        )
    if organ and normalize_organ(organ) != spec.organ:
        raise SpecialistInputError(
            f"Declared organ does not match model {spec.model_id}"
        )


def normalize_specialist_input(
    *,
    model_id: str,
    files: Iterable[UploadedSpecialistFile],
    workspace: str | Path,
    modality: str | None = None,
    organ: str | None = None,
    spacing_mm: tuple[float, float, float] | None = None,
) -> NormalizedSpecialistInput:
    """Normalize one uploaded request without retaining raw files."""

    entries = list(files)
    if not entries:
        raise SpecialistInputError("At least one input file is required")
    _validate_uploaded_entries(entries)
    entries = [
        UploadedSpecialistFile(
            name=str(entry.name or ""),
            content=bytes(entry.content),
        )
        for entry in entries
    ]

    spec = get_model_registry().get_spec(model_id)
    if spec is None:
        raise SpecialistInputError(f"Unknown specialist model: {model_id}")
    if not spec.checkpoint_available:
        raise SpecialistInputError(
            f"Specialist checkpoint is unavailable: {model_id}"
        )
    _validate_declared_metadata(spec, modality=modality, organ=organ)

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    suffixes = [_suffix(entry.name) for entry in entries]
    safe_metadata: dict[str, Any] = {
        "modality": spec.modality,
        "organ": spec.organ,
        "input_type": spec.input_type,
        "file_count": len(entries),
        "extensions": sorted(set(suffixes)),
    }

    if spec.modality == "CT":
        dicom_suffixes = {"", ".dcm", ".dicom", ".ima"}
        looks_like_dicom_series = all(
            suffix in dicom_suffixes for suffix in suffixes
        )
        if looks_like_dicom_series:
            try:
                series = load_dicom_series_from_bytes(
                    [(entry.name, entry.content) for entry in entries]
                )
            except (DicomSeriesError, ValueError) as exc:
                raise SpecialistInputError(
                    "Uploaded files do not form one valid CT DICOM series"
                ) from exc
            volume = _validate_volume_array(
                np.asarray(series.volume),
                label="DICOM CT volume",
            )
            _validate_model_volume_shape(spec, volume)
            resolved_spacing = (
                float(series.slice_spacing or series.slice_thickness or 1.0),
                float(series.pixel_spacing[0]),
                float(series.pixel_spacing[1]),
            )
            safe_metadata.update(
                {
                    "input_format": "DICOM",
                    "spacing_mm": resolved_spacing,
                    "spacing_source": "DICOM",
                    "series_shape": series.shape,
                    "ordering_method": series.ordering_method,
                    "intensity_units": "HU",
                    "hu_conversion_applied": True,
                    "intensity_range_hu": (
                        float(np.min(volume)),
                        float(np.max(volume)),
                    ),
                }
            )
            return NormalizedSpecialistInput(volume, safe_metadata)

        file = _single_file(entries)
        if _suffix(file.name) in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            raster = _decode_rgb(file)
            volume = _raster_to_single_slice_volume(raster, modality="CT")
            _validate_model_volume_shape(spec, volume)
            safe_metadata.update(
                {
                    "input_format": "raster_scan_export",
                    "shape": volume.shape,
                    "source_image_shape": raster.shape,
                    "single_slice_approximation": True,
                    "spacing_source": "not_available_from_image",
                    "intensity_units": "display_pixels",
                    "hu_conversion_applied": False,
                }
            )
            return NormalizedSpecialistInput(volume, safe_metadata)
        if _suffix(file.name) == ".npy":
            volume = _load_numpy(file)
            _validate_model_volume_shape(spec, volume)
            safe_metadata.update(
                {
                    "input_format": "NumPy",
                    "shape": volume.shape,
                    "intensity_units": "unknown",
                    "hu_conversion_applied": False,
                    "spacing_source": "request" if spacing_mm else "not_provided",
                }
            )
            if spacing_mm is not None:
                safe_metadata["spacing_mm"] = spacing_mm
            return NormalizedSpecialistInput(volume, safe_metadata)
        if _suffix(file.name) == ".nii.gz" or _suffix(file.name) == ".nii":
            volume, nifti_spacing = _load_nifti(file, workspace_path)
            _validate_model_volume_shape(spec, volume)
            safe_metadata.update(
                {
                    "input_format": "NIfTI",
                    "shape": volume.shape,
                    "spacing_mm": spacing_mm or nifti_spacing,
                    "spacing_source": "request" if spacing_mm else "NIfTI header",
                    "intensity_units": "unknown",
                    "hu_conversion_applied": False,
                }
            )
            return NormalizedSpecialistInput(volume, safe_metadata)
        raise SpecialistInputError(
            "CT input must be a DICOM series, .npy volume, or NIfTI volume"
        )

    if spec.modality == "MRI":
        file = _single_file(entries)
        if _suffix(file.name) in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            raster = _decode_rgb(file)
            volume = _raster_to_single_slice_volume(raster, modality="MRI")
            _validate_model_volume_shape(spec, volume)
            safe_metadata.update(
                {
                    "input_format": "raster_scan_export",
                    "shape": volume.shape,
                    "source_image_shape": raster.shape,
                    "single_slice_approximation": True,
                    "spacing_source": "not_available_from_image",
                    "intensity_units": "display_pixels",
                }
            )
            return NormalizedSpecialistInput(volume, safe_metadata)
        if _suffix(file.name) == ".npy":
            volume = _load_numpy(file)
            _validate_model_volume_shape(spec, volume)
            safe_metadata.update(
                {
                    "input_format": "NumPy",
                    "shape": volume.shape,
                    "spacing_source": "request" if spacing_mm else "not_provided",
                    "intensity_units": "unknown",
                }
            )
            if spacing_mm is not None:
                safe_metadata["spacing_mm"] = spacing_mm
            return NormalizedSpecialistInput(volume, safe_metadata)
        if _suffix(file.name) == ".nii.gz" or _suffix(file.name) == ".nii":
            volume, nifti_spacing = _load_nifti(file, workspace_path)
            _validate_model_volume_shape(spec, volume)
            safe_metadata.update(
                {
                    "input_format": "NIfTI",
                    "shape": volume.shape,
                    "spacing_mm": spacing_mm or nifti_spacing,
                    "spacing_source": "request" if spacing_mm else "NIfTI header",
                    "intensity_units": "unknown",
                }
            )
            return NormalizedSpecialistInput(volume, safe_metadata)
        raise SpecialistInputError("MRI input must be a NumPy or NIfTI volume")

    if spec.modality in {"ULTRASOUND", "DERMOSCOPY", "MICROSCOPY"}:
        file = _single_file(entries)
        raster = _decode_rgb(file)
        safe_metadata.update(
            {
                "input_format": "raster_image",
                "shape": raster.shape,
                "pixel_count": int(raster.shape[0] * raster.shape[1]),
            }
        )
        if model_id == "tn3k_thyroid_nodule_segmentation":
            extension = Path(file.name).suffix.lower() or ".img"
            target = workspace_path / f"uploaded_image{extension}"
            target.write_bytes(file.content)
            return NormalizedSpecialistInput(target, safe_metadata)
        if model_id.startswith("busi_"):
            return NormalizedSpecialistInput(file.content, safe_metadata)
        return NormalizedSpecialistInput(raster, safe_metadata)

    if spec.modality == "FLOW_CYTOMETRY":
        if len(entries) != 8 or any(suffix != ".csv" for suffix in suffixes):
            raise SpecialistInputError(
                "Flow-cytometry analysis requires exactly eight CSV tube files"
            )
        paths = []
        event_counts = []
        for index, entry in enumerate(entries, start=1):
            event_counts.append(_validate_flow_csv(entry))
            target = workspace_path / f"tube_{index}.csv"
            target.write_bytes(entry.content)
            paths.append(target)
        safe_metadata["input_format"] = "CSV"
        safe_metadata["tube_count"] = len(paths)
        safe_metadata["event_counts"] = event_counts
        safe_metadata["validation"] = "schema_and_numeric"
        return NormalizedSpecialistInput(paths, safe_metadata)

    raise SpecialistInputError(
        f"No unified input normalizer exists for modality {spec.modality}"
    )
