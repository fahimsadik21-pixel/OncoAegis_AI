"""
Liver CT AI Output Export

Stores:
- Segmentation mask as compressed NPZ
- DICOM-derived spacing
- Prediction / measurement metadata as JSON

This keeps the AI pipeline independent of whether
the end user understands DICOM, NIfTI, or other formats.

Research use only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_DIR = (
    _PROJECT_ROOT
    / "outputs"
    / "liver_results"
)


class LiverOutputExportError(Exception):
    pass


def _safe_name(
    value: str,
) -> str:

    value = str(value).strip()

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    return (
        value.strip("._")
        or "liver_case"
    )


def _json_safe(
    value: Any,
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
        return str(value)

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
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    return value


def export_liver_ai_result(
    segmentation_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    case_id: str,
    prediction: dict | None = None,
    measurement: dict | None = None,
    dicom_metadata: dict | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
):

    if segmentation_mask.ndim != 3:

        raise LiverOutputExportError(
            "Expected a 3D segmentation mask, "
            f"got {segmentation_mask.shape}"
        )

    if len(spacing_mm) != 3:

        raise LiverOutputExportError(
            "spacing_mm must contain "
            "(z, row, column)"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_case_id = _safe_name(
        case_id
    )

    mask_path = (
        output_dir
        / f"{safe_case_id}_segmentation.npz"
    )

    metadata_path = (
        output_dir
        / f"{safe_case_id}_result.json"
    )

    mask = segmentation_mask.astype(
        np.uint8
    )

    np.savez_compressed(
        mask_path,
        segmentation_mask=mask,
        spacing_mm=np.asarray(
            spacing_mm,
            dtype=np.float32,
        ),
    )

    metadata = {

        "case_id":
            safe_case_id,

        "segmentation_labels": {
            "0": "background",
            "1": "liver",
            "2": "tumor",
        },

        "mask_shape": [
            int(x)
            for x in mask.shape
        ],

        "spacing_mm": [
            float(x)
            for x in spacing_mm
        ],

        "prediction":
            prediction or {},

        "measurement":
            measurement or {},

        "dicom_metadata":
            dicom_metadata or {},

        "files": {
            "segmentation_mask":
                str(mask_path),
        },

        "status":
            "research_model_output",
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(metadata),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return {

        "mask_file":
            str(mask_path),

        "metadata_file":
            str(metadata_path),
    }


def load_exported_liver_mask(
    mask_path: str | Path,
):

    mask_path = Path(
        mask_path
    )

    if not mask_path.exists():

        raise LiverOutputExportError(
            f"Mask file not found: {mask_path}"
        )

    data = np.load(
        mask_path
    )

    mask = data[
        "segmentation_mask"
    ].astype(
        np.uint8
    )

    spacing_mm = tuple(
        float(x)
        for x in data[
            "spacing_mm"
        ]
    )

    return (
        mask,
        spacing_mm,
    )


if __name__ == "__main__":

    print(
        "Liver output export module ready."
    )