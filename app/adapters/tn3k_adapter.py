from uuid import uuid4
from app.adapters.base import BaseSpecialistAdapter

from app.schemas.analysis_result import (
    StandardAnalysisResult,
    MeasurementObject,
    SafetyStatus,
    SpecialistMetadata,
)

from app.schemas.evidence import EvidenceObject



class TN3KAdapter(BaseSpecialistAdapter):
    """
    Converts TN3K specialist output into
    universal OncoAegis format.
    """

    adapter_id = "tn3k_adapter"
    specialist_id = "tn3k_thyroid_adapter"


    def convert(
        self,
        tn3k_result: dict,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        model_info = tn3k_result.get(
            "model",
            {}
        )


        measurements = (
            tn3k_result.get(
                "measurements",
                {}
            )
        )


        segmentation = (
            tn3k_result.get(
                "segmentation",
                {}
            )
        )


        evidence = []


        evidence.append(
            EvidenceObject(
                evidence_id=str(uuid4()),

                source_type="imaging_model",

                source_reference=(
                    model_info.get("id")
                ),

                statement=(
                    "Research ultrasound segmentation "
                    "model predicted a thyroid region."
                ),

                normalized_finding=(
                    "thyroid_nodule_region"
                ),

                polarity=(
                    "present"
                    if segmentation.get(
                        "detected_region"
                    )
                    else "absent"
                ),

                confidence=(
                    segmentation.get(
                        "mean_probability_inside_predicted_region"
                    )
                ),

                model_derived=True,

                clinically_confirmed=False,

                limitations=[
                    "Segmentation output only.",
                    "Does not confirm thyroid cancer."
                ],
            )
        )


        output_measurements = []


        if measurements.get(
            "area_pixels"
        ) is not None:

            output_measurements.append(
                MeasurementObject(
                    name="nodule_area_pixels",

                    value=measurements.get(
                        "area_pixels"
                    ),

                    unit="pixels",

                    approximate=True,

                    physical_measurement_available=False,

                    limitation=(
                        "Physical spacing unavailable."
                    ),
                )
            )


        safety = SafetyStatus(

            research_only=True,

            clinical_diagnosis=False,

            expert_review_required=True,

            message=(
                "Research ultrasound segmentation. "
                "Does not diagnose thyroid cancer."
            ),
        )


        specialist = SpecialistMetadata(

            specialist_id=self.specialist_id,

            model_id=model_info.get(
                "id"
            ),

            model_version="1.0",

            dataset=model_info.get(
                "dataset"
            ),

            checkpoint=model_info.get(
                "checkpoint"
            ),

            task="segmentation",
        )


        return StandardAnalysisResult(

            analysis_id=str(uuid4()),

            input=input_metadata or tn3k_result.get("input", {}),

            specialist=specialist,

            findings=[
                {
                    "type":
                    "thyroid_nodule_region",
                    
                    "detected":
                    segmentation.get(
                        "detected_region",
                        False
                    )
                }
            ],

            measurements=output_measurements,

            evidence=evidence,

            limitations=[
                "Research-only model.",
                "Requires expert review."
            ],

            safety=safety,

            artifacts=(
                tn3k_result.get(
                    "outputs",
                    {}
                )
            ),

            expert_review=True,
        )
