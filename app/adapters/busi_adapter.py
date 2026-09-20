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


class BUSIAdapter(BaseSpecialistAdapter):
    """
    Converts BUSI ultrasound research output
    into the universal OncoAegis result contract.
    """

    adapter_id = "busi_adapter"

    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "breast_ultrasound",
            "dataset": "BUSI",
        }

    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:

        findings = [
            {
                "type": "breast_lesion_research_prediction",
                "predicted_class": prediction.predicted_class,
                "class_probabilities": prediction.class_probabilities,
            }
        ]

        measurements = [
            MeasurementObject(
                name="lesion_area_fraction",
                value=prediction.lesion_area_fraction,
                unit="fraction_of_image",
                approximate=True,
                physical_measurement_available=False,
                limitation=(
                    "Pixel fraction only. "
                    "Physical lesion size requires calibrated spacing."
                ),
            )
        ]

        limitations = list(prediction.warnings)

        limitations.extend(
            [
                "BUSI dataset labels are research categories.",
                "Ultrasound AI output cannot confirm malignancy.",
                "Clinical correlation and expert review are required.",
            ]
        )

        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),

            input=input_metadata or {
                "modality": "ULTRASOUND",
                "organ": "breast",
                "input_type": "raster_image",
            },

            specialist=SpecialistMetadata(
                specialist_id="busi_breast_ultrasound",

                model_id="busi_breast_classifier",

                model_version=prediction.model_version,

                dataset="BUSI",

                task="classification + segmentation",

                checkpoint=(
                    "checkpoints/busi/"
                    "busi_multitask_unet_best.pt"
                ),
            ),

            findings=findings,

            measurements=measurements,

            uncertainty={
                "model_confidence": prediction.model_confidence,
                "confidence_calibration": "uncalibrated",
                "interpretation": "research prediction only",
            },

            limitations=limitations,

            safety=SafetyStatus(
                research_only=True,

                clinical_diagnosis=False,

                expert_review_required=True,

                message=(
                    "BUSI model output is a research finding. "
                    "It does not confirm breast cancer. "
                    "Expert medical review is required."
                ),
            ),

            provenance={
                "dataset": "BUSI",
                "model": prediction.model_name,
                "warnings": prediction.warnings,
            },

            artifacts={},

            expert_review=True,
        )