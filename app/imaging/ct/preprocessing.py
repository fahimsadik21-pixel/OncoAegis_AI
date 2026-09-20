"""Dataset-independent CT preprocessing primitives.

These operations prepare a validated DICOM volume for a future model. They
do not make a cancer diagnosis and their default window is only a general
technical normalization, not a clinically calibrated decision threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.ndimage import zoom


class CTPreprocessingError(ValueError):
    """Raised when a CT volume or spacing cannot be preprocessed safely."""


@dataclass
class PreprocessedCT:
    """Normalized CT volume and the geometry needed by downstream models."""

    volume: np.ndarray
    original_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    output_spacing_mm: tuple[float, float, float]
    hu_window: tuple[float, float]
    clipped_fraction: float


def _validate_spacing(
    spacing: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    if len(spacing) != 3:
        raise CTPreprocessingError(f"{name} must contain three values")

    values = tuple(float(value) for value in spacing)
    if any(not isfinite(value) or value <= 0 for value in values):
        raise CTPreprocessingError(f"{name} must contain positive finite values")
    return values


def _resample(
    volume: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> np.ndarray:
    output_shape = tuple(
        max(1, int(round(size * current / target)))
        for size, current, target in zip(
            volume.shape,
            current_spacing,
            target_spacing,
        )
    )
    zoom_factors = tuple(
        output / current
        for output, current in zip(output_shape, volume.shape)
    )
    if output_shape == volume.shape:
        return volume.astype(np.float32, copy=True)

    return zoom(
        volume,
        zoom=zoom_factors,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)


def preprocess_ct_volume(
    volume: np.ndarray,
    *,
    pixel_spacing_mm: tuple[float, float],
    slice_spacing_mm: float,
    target_spacing_mm: tuple[float, float, float] | None = None,
    window_center: float = 40.0,
    window_width: float = 400.0,
) -> PreprocessedCT:
    """Resample and normalize a CT volume for model input.

    Axis order is ``(slice, row, column)``. Pixel spacing follows DICOM's
    ``(row, column)`` convention, so the full spacing tuple is
    ``(slice_spacing, row_spacing, column_spacing)``.
    """

    array = np.asarray(volume)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise CTPreprocessingError(
            "CT volume must be a non-empty 3D array"
        )

    if len(pixel_spacing_mm) != 2:
        raise CTPreprocessingError(
            "pixel_spacing_mm must contain row and column spacing"
        )
    original_spacing = _validate_spacing(
        (
            float(slice_spacing_mm),
            float(pixel_spacing_mm[0]),
            float(pixel_spacing_mm[1]),
        ),
        "pixel spacing",
    )
    target_spacing = _validate_spacing(
        target_spacing_mm or original_spacing,
        "target_spacing_mm",
    )

    if not isfinite(window_center) or not isfinite(window_width):
        raise CTPreprocessingError("HU window values must be finite")
    if window_width <= 0:
        raise CTPreprocessingError("window_width must be greater than zero")

    working = array.astype(np.float32, copy=False)
    if not np.isfinite(working).all():
        raise CTPreprocessingError("CT volume contains non-finite values")

    original_shape = tuple(int(size) for size in working.shape)
    resampled = _resample(
        working,
        original_spacing,
        target_spacing,
    )

    lower = float(window_center - window_width / 2.0)
    upper = float(window_center + window_width / 2.0)
    clipped_fraction = float(
        np.mean((resampled < lower) | (resampled > upper))
    )
    normalized = np.clip(resampled, lower, upper)
    normalized = ((normalized - lower) / (upper - lower)).astype(
        np.float32,
        copy=False,
    )

    return PreprocessedCT(
        volume=normalized,
        original_shape=original_shape,
        output_shape=tuple(int(size) for size in normalized.shape),
        original_spacing_mm=original_spacing,
        output_spacing_mm=target_spacing,
        hu_window=(lower, upper),
        clipped_fraction=clipped_fraction,
    )

