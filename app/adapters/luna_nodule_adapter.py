from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.adapters.base import BaseSpecialistAdapter
from app.schemas.analysis_result import (
    MeasurementObject,
    SafetyStatus,
    SpecialistMetadata,
    StandardAnalysisResult,
)


class LUNANoduleAdapter(BaseSpecialistAdapter):
    """Convert LUNA16 annotated-nodule research output."""

    adapter_id = "luna_nodule_adapter"

    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "lung_nodule_patch_classifier",
            "dataset": "LUNA16",
        }

    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:
        probabilities = dict(
            getattr(prediction, "class_probabilities", {})
        )
        warnings = list(getattr(prediction, "warnings", ()))
        return StandardAnalysisResult(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            input=input_metadata or {
                "modality": "CT",
                "organ": "lung",
                "input_type": "nodule-centred 2D CT patch",
            },
            specialist=SpecialistMetadata(
                specialist_id="luna16_nodule_classifier",
                model_id="luna16_nodule_detector",
                model_version=getattr(
                    prediction, "model_version", "research-baseline"
                ),
                dataset="LUNA16",
                task="annotated-nodule detection",
                checkpoint=(
                    "checkpoints/luna16_nodule/"
                    "luna16_nodule_classifier_best.pt"
                ),
            ),
            findings=[
                {
                    "type": "lung_annotated_nodule_research_prediction",
                    "detected": bool(
                        getattr(prediction, "nodule_detected", False)
                    ),
                    "predicted_class": getattr(
                        prediction, "predicted_class", None
                    ),
                    "class_probabilities": probabilities,
                }
            ],
            measurements=[
                MeasurementObject(
                    name="nodule_model_confidence",
                    value=getattr(prediction, "confidence", None),
                    unit="probability",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation="Uncalibrated research-model score.",
                )
            ],
            uncertainty={
                "interpretation": "annotated-nodule research output",
                "clinical_calibration": "not clinically validated",
            },
            limitations=warnings + [
                "LUNA16 does not provide pathology-confirmed malignancy labels.",
                "A positive nodule-like score cannot confirm lung cancer.",
                "Expert radiology review is required.",
            ],
            safety=SafetyStatus(
                research_only=True,
                clinical_diagnosis=False,
                expert_review_required=True,
                message=(
                    "LUNA16 nodule output is research evidence only; "
                    "it cannot confirm lung cancer."
                ),
            ),
            provenance={
                "dataset": "LUNA16",
                "model": "LUNA16 Annotated-Nodule Patch Classifier",
            },
            artifacts={},
            expert_review=True,
        )
