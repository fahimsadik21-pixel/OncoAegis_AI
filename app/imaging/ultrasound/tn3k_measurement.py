from __future__ import annotations

import numpy as np


def measure_binary_mask(mask: np.ndarray) -> dict:
    """
    Pixel-based measurements only.

    TN3K JPG images do not provide reliable physical pixel spacing,
    therefore no mm/cm measurement is claimed.
    """

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    binary = mask > 0

    height, width = binary.shape

    area_pixels = int(binary.sum())
    total_pixels = int(height * width)

    area_percent = (
        100.0 * area_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    ys, xs = np.where(binary)

    if len(xs) == 0:
        return {
            "detected_region": False,
            "area_pixels": 0,
            "image_area_pixels": total_pixels,
            "area_percent": 0.0,
            "bounding_box_pixels": None,
            "width_pixels": 0,
            "height_pixels": 0,
            "physical_measurement_available": False,
        }

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    return {
        "detected_region": True,

        "area_pixels": area_pixels,

        "image_area_pixels": total_pixels,

        "area_percent": float(area_percent),

        "bounding_box_pixels": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },

        "width_pixels": int(bbox_width),

        "height_pixels": int(bbox_height),

        "physical_measurement_available": False,

        "physical_measurement_note": (
            "TN3K image files do not provide validated physical "
            "pixel spacing, so pixel dimensions are reported only."
        ),
    }