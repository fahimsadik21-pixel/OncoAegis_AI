"""
C-NMC hematology result export.

Research use only.
"""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_DIR = Path(
    "outputs/hematology_results"
)


def export_cnmc_result(
    case_id: str,
    prediction,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        OUTPUT_DIR
        /
        f"{case_id}_result.json"
    )

    metadata = {

        "case_id":
            case_id,

        "modality":
            "MICROSCOPY",

        "organ":
            "blood",

        "dataset":
            "C-NMC 2019",

        "task":
            "ALL_vs_HEM_cell_classification",

        "prediction":
            prediction.to_dict(),

        "class_mapping": {
            "0":
                "HEM-like",

            "1":
                "ALL-like",
        },

        "status":
            "research_model_prediction",

        "safety": {

            "clinical_diagnosis":
                False,

            "patient_level_leukemia_diagnosis":
                False,

            "leukemia_confirmation":
                False,

            "leukemia_exclusion":
                False,

            "expert_review_required":
                True,

            "model_clinically_validated":
                False,

            "flow_cytometry_or_pathology_required_for_confirmation":
                True,
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "metadata_file":
            str(
                metadata_path.resolve()
            )
    }