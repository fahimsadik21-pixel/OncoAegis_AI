"""
Pancreas CT result export.

Exports:
    compressed segmentation NPZ
    JSON metadata

Research use only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PancreasOutputExportError(Exception):
    pass


OUTPUT_DIR = Path(
    "outputs/pancreas_results"
)


def _json_safe(
    value,
):

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_safe(x)
            for x in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            _json_safe(x)
            for x in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                _json_safe(v)
            for k, v
            in value.items()
        }

    return value


def export_pancreas_result(
    segmentation_mask: np.ndarray,
    spacing_mm,
    case_id: str,
    prediction,
    measurement=None,
):

    if segmentation_mask.ndim != 3:

        raise PancreasOutputExportError(
            f"Expected 3D segmentation mask, "
            f"got {segmentation_mask.shape}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        OUTPUT_DIR
        /
        f"{case_id}_segmentation.npz"
    )

    metadata_path = (
        OUTPUT_DIR
        /
        f"{case_id}_result.json"
    )

    spacing = tuple(
        float(x)
        for x
        in spacing_mm
    )

    np.savez_compressed(
        mask_path,
        segmentation_mask=
            segmentation_mask.astype(
                np.uint8
            ),
        spacing_mm=
            np.asarray(
                spacing,
                dtype=np.float32,
            ),
    )

    if hasattr(
        prediction,
        "to_dict",
    ):
        prediction_data = (
            prediction.to_dict()
        )
    else:
        prediction_data = prediction

    if measurement is None:
        measurement_data = None

    elif hasattr(
        measurement,
        "to_dict",
    ):
        measurement_data = (
            measurement.to_dict()
        )

    else:
        measurement_data = measurement

    metadata = {
        "case_id":
            case_id,

        "modality":
            "CT",

        "organ":
            "pancreas",

        "dataset":
            "MSD_Task07_Pancreas",

        "labels": {
            "0":
                "background",

            "1":
                "pancreas",

            "2":
                "cancer_model_region",
        },

        "spacing_mm":
            spacing,

        "prediction":
            prediction_data,

        "measurement":
            measurement_data,

        "status":
            "research_model_prediction",

        "safety": {
            "clinical_diagnosis":
                False,

            "malignancy_confirmation":
                False,

            "expert_review_required":
                True,

            "model_clinically_validated":
                False,
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(
                metadata
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "mask_file":
            str(
                mask_path.resolve()
            ),

        "metadata_file":
            str(
                metadata_path.resolve()
            ),
    }