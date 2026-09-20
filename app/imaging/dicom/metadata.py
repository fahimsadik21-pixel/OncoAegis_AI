"""Safe, model-facing metadata extraction for DICOM series.

The DICOM objects are still available to the loader for technical validation,
but this module deliberately exposes only metadata needed by an imaging
pipeline. Patient names, IDs, dates, accession numbers and other identifying
fields are never copied into the returned summary.
"""

from __future__ import annotations

from typing import Any, Iterable


_BODY_REGION_ALIASES = {
    "CHEST": "CHEST",
    "THORAX": "CHEST",
    "LUNG": "CHEST",
    "ABDOMEN": "ABDOMEN",
    "PELVIS": "PELVIS",
    "HEAD": "HEAD",
    "BRAIN": "HEAD",
    "NECK": "NECK",
    "SPINE": "SPINE",
    "BREAST": "BREAST",
}


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_value(datasets: Iterable[Any], attribute: str) -> Any:
    for dataset in datasets:
        value = getattr(dataset, attribute, None)
        if value not in (None, ""):
            return value
    return None


def normalize_body_region(value: Any) -> str | None:
    """Convert common DICOM body-part values to stable routing labels."""

    text = _clean_text(value)
    if text is None:
        return None

    normalized = " ".join(text.upper().replace("_", " ").split())
    direct = _BODY_REGION_ALIASES.get(normalized)
    if direct:
        return direct
    for alias, region in _BODY_REGION_ALIASES.items():
        if alias in normalized.split():
            return region
    return normalized


def build_safe_metadata(
    datasets: Iterable[Any],
    *,
    slice_count: int,
    rows: int,
    columns: int,
    pixel_spacing: tuple[float, float],
    slice_spacing: float | None,
    slice_thickness: float | None,
    orientation: tuple[float, ...] | None,
    ordering_method: str,
    warnings: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a technical metadata summary without patient identifiers."""

    dataset_list = list(datasets)
    modality = _clean_text(_first_value(dataset_list, "Modality"))
    body_part = _first_value(dataset_list, "BodyPartExamined")
    if body_part in (None, ""):
        body_part = _first_value(dataset_list, "StudyDescription")
    if body_part in (None, ""):
        body_part = _first_value(dataset_list, "SeriesDescription")
    body_region = normalize_body_region(body_part)
    manufacturer = _clean_text(_first_value(dataset_list, "Manufacturer"))
    model_name = _clean_text(
        _first_value(dataset_list, "ManufacturerModelName")
    )

    return {
        "modality": modality.upper() if modality else "UNKNOWN",
        "body_region": body_region,
        "manufacturer": manufacturer,
        "manufacturer_model": model_name,
        "slice_count": slice_count,
        "rows": rows,
        "columns": columns,
        "pixel_spacing_mm": [pixel_spacing[0], pixel_spacing[1]],
        "slice_spacing_mm": slice_spacing,
        "slice_thickness_mm": slice_thickness,
        "orientation": list(orientation) if orientation else None,
        "ordering_method": ordering_method,
        "warnings": list(warnings),
        "patient_identifiers_in_response": False,
    }
