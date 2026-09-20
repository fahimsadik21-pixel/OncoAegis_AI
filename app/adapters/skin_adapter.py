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


class SkinAdapter(BaseSpecialistAdapter):
    """
    Converts ISIC2016 dermoscopy research output
    into StandardAnalysisResult.
    """

    adapter_id = "skin_adapter"


    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "skin_dermoscopy",
            "dataset": "ISIC2016",
        }


    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        if hasattr(prediction, "lesion_detected"):

            findings.append(
                {
                    "type": "skin_lesion_region",
                    "detected":
                        prediction.lesion_detected,
                }
            )


        if hasattr(prediction, "lesion_pixels"):

            findings.append(
                {
                    "type": "lesion_area_estimation",
                    "value":
                        prediction.lesion_pixels,
                }
            )


        measurements = []


        if hasattr(prediction, "lesion_pixels"):

            measurements.append(
                MeasurementObject(
                    name="lesion_pixel_area",
                    value=prediction.lesion_pixels,
                    unit="pixels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Physical lesion size requires "
                        "image calibration."
                    ),
                )
            )

        if hasattr(prediction, "lesion_percentage"):
            findings.append(
                {
                    "type": "skin_lesion_fraction",
                    "percentage": prediction.lesion_percentage,
                }
            )


        limitations = [

            "ISIC2016 is a research dermoscopy dataset.",

            "Segmentation does not confirm melanoma or skin cancer.",

            "Dermatologist evaluation is required.",

        ]


        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),


            input=input_metadata or {

                "modality": "DERMOSCOPY",

                "organ": "skin",

                "input_type": "RGB image",

            },


            specialist=SpecialistMetadata(

                specialist_id="skin_dermoscopy",

                model_id=(
                    "isic2016_skin_lesion_segmentation"
                ),

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline"
                ),

                dataset="ISIC2016",

                task="segmentation",

                checkpoint=(
                    "checkpoints/"
                    "isic2016_segmentation/"
                    "isic2016_skin_segmentation_best.pt"
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

                    "Skin lesion AI output is "
                    "research evidence only. "
                    "It cannot confirm skin cancer."

                ),

            ),


            provenance={

                "dataset":
                    "ISIC2016",

                "model":
                    "ISIC2016 Skin Lesion Segmentation",

            },


            artifacts={},


            expert_review=True,

        )
