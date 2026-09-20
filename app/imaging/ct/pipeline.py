"""End-to-end CT preparation pipeline before specialist models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.anatomy_router import AnatomyRoute, route_body_region
from app.imaging.ct.preprocessing import (
    CTPreprocessingError,
    PreprocessedCT,
    preprocess_ct_volume,
)
from app.imaging.dicom.loader import (
    DicomSeries,
    load_dicom_series,
    load_dicom_series_from_bytes,
)


@dataclass
class CTPipelineResult:
    """Validated CT data ready for a future anatomy/specialist model."""

    series: DicomSeries
    preprocessed: PreprocessedCT
    anatomy_route: AnatomyRoute

    @property
    def model_input(self) -> dict:
        """Return only model-safe data; raw DICOM datasets stay out of it."""

        return {
            "volume": self.preprocessed.volume,
            "metadata": self.series.safe_metadata,
            "original_spacing_mm": self.preprocessed.original_spacing_mm,
            "output_spacing_mm": self.preprocessed.output_spacing_mm,
            "candidate_organs": self.anatomy_route.candidate_organs,
        }


def _prepare_series(
    series: DicomSeries,
    *,
    target_spacing_mm: tuple[float, float, float] | None = None,
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> CTPipelineResult:
    slice_spacing = series.slice_spacing or series.slice_thickness
    if slice_spacing is None:
        raise CTPreprocessingError(
            "CT series has no usable slice spacing or slice thickness"
        )

    preprocessed = preprocess_ct_volume(
        series.volume,
        pixel_spacing_mm=series.pixel_spacing,
        slice_spacing_mm=slice_spacing,
        target_spacing_mm=target_spacing_mm,
        window_center=window_center,
        window_width=window_width,
    )
    body_region = series.safe_metadata.get("body_region")
    anatomy_route = route_body_region(body_region)

    return CTPipelineResult(
        series=series,
        preprocessed=preprocessed,
        anatomy_route=anatomy_route,
    )


def run_ct_pipeline_from_bytes(
    files: list[tuple[str, bytes]],
    *,
    target_spacing_mm: tuple[float, float, float] | None = None,
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> CTPipelineResult:
    """Run the preparation pipeline for uploaded DICOM slices."""

    series = load_dicom_series_from_bytes(files)
    return _prepare_series(
        series,
        target_spacing_mm=target_spacing_mm,
        window_center=window_center,
        window_width=window_width,
    )


def run_ct_pipeline(
    source: str | Path,
    *,
    target_spacing_mm: tuple[float, float, float] | None = None,
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> CTPipelineResult:
    """Run the preparation pipeline for a local DICOM file or directory."""

    series = load_dicom_series(source)
    return _prepare_series(
        series,
        target_spacing_mm=target_spacing_mm,
        window_center=window_center,
        window_width=window_width,
    )
