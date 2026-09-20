from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.adapters.base import BaseSpecialistAdapter

from app.schemas.analysis_result import (
    StandardAnalysisResult,
    SafetyStatus,
    SpecialistMetadata,
)


class CNMCAdapter(BaseSpecialistAdapter):
    """
    Converts C-NMC 2019 leukemia microscopy research output
    into StandardAnalysisResult.
    """

    adapter_id = "cnmc_adapter"


    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "hematology_microscopy",
            "dataset": "C-NMC_2019",
        }


    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        if hasattr(prediction, "predicted_class"):

            findings.append(
                {
                    "type": "blood_cell_research_classification",
                    "predicted_class":
                        prediction.predicted_class,
                }
            )


        if hasattr(prediction, "hem_probability"):

            findings.append(
                {
                    "type": "cell_class_probabilities",
                    "probabilities": {
                        "HEM-like": prediction.hem_probability,
                        "ALL-like": prediction.all_probability,
                    },
                    "confidence": prediction.confidence,
                }
            )


        limitations = [

            "C-NMC 2019 is a research microscopy dataset.",

            "Cell classification output cannot diagnose leukemia.",

            "Hematology expert review is required.",

        ]


        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),


            input=input_metadata or {

                "modality": "MICROSCOPY",

                "organ": "blood",

                "input_type": "cell_image",

            },


            specialist=SpecialistMetadata(

                specialist_id="cnmc2019",

                model_id=(
                    "cnmc2019_all_cell_classifier"
                ),

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline"
                ),

                dataset="C-NMC 2019",

                task="classification",

                checkpoint=(
                    "checkpoints/cnmc2019/"
                    "cnmc2019_classifier_best.pt"
                ),

            ),


            findings=findings,


            measurements=[],


            uncertainty={

                "interpretation":
                    "research cell classification",

                "clinical_calibration":
                    "not clinically validated",

            },


            limitations=limitations,


            safety=SafetyStatus(

                research_only=True,

                clinical_diagnosis=False,

                expert_review_required=True,

                message=(

                    "C-NMC microscopy model output "
                    "is research evidence only. "
                    "It cannot confirm leukemia."

                ),

            ),


            provenance={

                "dataset":
                    "C-NMC 2019",

                "model":
                    "ALL Cell Classifier",

            },


            artifacts={},


            expert_review=True,

        )
