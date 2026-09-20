from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.adapters.base import BaseSpecialistAdapter

from app.schemas.analysis_result import (
    StandardAnalysisResult,
    MeasurementObject,
    SafetyStatus,
    SpecialistMetadata,
)


class ColonCTAdapter(BaseSpecialistAdapter):
    """
    Converts MSD Colon CT research output
    into StandardAnalysisResult.
    """

    adapter_id = "colon_ct_adapter"


    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "colon_ct",
            "dataset": "MSD_Task10_Colon",
        }


    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        if hasattr(prediction, "colon_detected"):

            findings.append(
                {
                    "type": "colon_segmentation",
                    "detected":
                        prediction.colon_detected,
                }
            )


        if hasattr(prediction, "tumor_detected"):

            findings.append(
                {
                    "type": "colon_tumor_region_research_finding",
                    "detected":
                        prediction.tumor_detected,
                }
            )


        measurements = []


        if hasattr(prediction, "tumor_volume"):

            measurements.append(
                MeasurementObject(
                    name="colon_region_volume",
                    value=prediction.tumor_volume,
                    unit="voxels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Physical measurement requires "
                        "validated CT spacing."
                    ),
                )
            )


        limitations = [

            "MSD Colon dataset is a research dataset.",

            "Segmentation output does not confirm colorectal cancer.",

            "Automated CT findings require expert radiology review.",

        ]


        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),


            input=input_metadata or {

                "modality": "CT",

                "organ": "colon",

                "input_type": "DICOM/NIfTI",

            },


            specialist=SpecialistMetadata(

                specialist_id="colon_ct",

                model_id=(
                    "msd_colon_tumor_segmentation"
                ),

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline"
                ),

                dataset="MSD_Task10_Colon",

                task="segmentation",

                checkpoint=(
                    "checkpoints/msd_colon_tumor/"
                    "colon_tumor_best.pt"
                ),

            ),


            findings=findings,


            measurements=measurements,


            uncertainty={

                "interpretation":
                    "research segmentation output",

                "clinical_calibration":
                    "not clinically validated",

            },


            limitations=limitations,


            safety=SafetyStatus(

                research_only=True,

                clinical_diagnosis=False,

                expert_review_required=True,

                message=(

                    "Colon CT model output is "
                    "research evidence only. "
                    "It cannot confirm colorectal cancer."

                ),

            ),


            provenance={

                "dataset":
                    "MSD_Task10_Colon",

                "model":
                    "MSD Colon Tumor CT Segmentation",

            },


            artifacts={},


            expert_review=True,

        )