"""
Skin lesion segmentation result export.

Research use only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SkinOutputExportError(Exception):
    pass


OUTPUT_DIR = Path(
    "outputs/skin_results"
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


def export_skin_result(
    segmentation_mask: np.ndarray,
    case_id: str,
    prediction,
):

    if segmentation_mask.ndim != 2:

        raise SkinOutputExportError(
            f"Expected 2D segmentation mask, "
            f"got {segmentation_mask.shape}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        OUTPUT_DIR
        /
        f"{case_id}_segmentation.npy"
    )

    metadata_path = (
        OUTPUT_DIR
        /
        f"{case_id}_result.json"
    )

    np.save(
        mask_path,
        segmentation_mask.astype(
            np.uint8
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

    metadata = {

        "case_id":
            case_id,

        "modality":
            "DERMOSCOPY",

        "organ":
            "skin",

        "dataset":
            "ISIC2016",

        "task":
            "lesion_segmentation",

        "labels": {
            "0":
                "background",

            "1":
                "skin_lesion_model_region",
        },

        "prediction":
            prediction_data,

        "status":
            "research_model_prediction",

        "safety": {

            "clinical_diagnosis":
                False,

            "melanoma_classification":
                False,

            "malignancy_confirmation":
                False,

            "cancer_exclusion":
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