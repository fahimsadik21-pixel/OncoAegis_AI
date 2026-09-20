from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.adapters.base import BaseSpecialistAdapter

from app.schemas.analysis_result import (
    StandardAnalysisResult,
    SafetyStatus,
    SpecialistMetadata,
)


class FlowCAPAdapter(BaseSpecialistAdapter):
    """
    Converts DREAM6 FlowCAP-II AML research output
    into StandardAnalysisResult.
    """

    adapter_id = "flowcap_adapter"


    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "flow_cytometry",
            "dataset": "DREAM6_FlowCAP_II_AML",
        }


    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        is_mapping = isinstance(prediction, dict)
        prediction_block = (
            prediction.get("prediction", {})
            if is_mapping
            else prediction
        )
        model_block = prediction.get("model", {}) if is_mapping else {}

        def value(name: str, default=None):
            if is_mapping:
                return prediction_block.get(name, default)
            return getattr(prediction, name, default)

        findings = []


        if value("combined_score") is not None:

            findings.append(
                {
                    "type": "patient_level_research_score",
                    "value": value("combined_score"),
                }
            )


        if value("tube_scores") is not None:

            findings.append(
                {
                    "type": "tube_level_scores",
                    "values": value("tube_scores"),
                }
            )


        limitations = [

            "DREAM6 FlowCAP-II AML is a research dataset.",

            "Model score does not confirm AML diagnosis.",

            "Flow cytometry interpretation requires hematopathology review.",

        ]


        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),


            input=input_metadata or {

                "modality": "FLOW_CYTOMETRY",

                "organ": "blood",

                "input_type": "CSV",

            },


            specialist=SpecialistMetadata(

                specialist_id="flowcap_aml",

                model_id=(
                    "flowcap_aml_patient_classifier"
                ),

                model_version=(
                    model_block.get("model_version", "research-baseline")
                    if is_mapping
                    else getattr(prediction, "model_version", "research-baseline")
                ),

                dataset="DREAM6_FlowCAP_II_AML",

                task="classification",

                checkpoint=(
                    "datasets/hematology/"
                    "flowcap_aml/"
                    "Dream6-binary/"
                    "final-classifier.xml"
                ),

            ),


            findings=findings,


            measurements=[],


            uncertainty={

                "interpretation":
                    "research flow-cytometry output",

                "clinical_calibration":
                    "not clinically validated",

            },


            limitations=limitations,


            safety=SafetyStatus(

                research_only=True,

                clinical_diagnosis=False,

                expert_review_required=True,

                message=(

                    "FlowCAP AML model output "
                    "is research evidence only. "
                    "It cannot confirm AML."

                ),

            ),


            provenance={

                "dataset":
                    "DREAM6_FlowCAP_II_AML",

                "model": model_block.get(
                    "name", "FlowCAP AML Classifier"
                ),

            },


            artifacts={},


            expert_review=True,

        )
