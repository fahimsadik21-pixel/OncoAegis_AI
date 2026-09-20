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


class LUNAAdapter(BaseSpecialistAdapter):
    """
    Converts LUNA16 lung CT research output
    into the universal OncoAegis result contract.
    """

    adapter_id = "luna_adapter"


    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "lung_ct",
            "dataset": "LUNA16",
        }


    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        # Lung segmentation result
        if hasattr(prediction, "positive_voxel_count"):
            findings.append(
                {
                    "type": "lung_segmentation",
                    "detected": prediction.positive_voxel_count > 0,
                    "positive_slice_count": prediction.positive_slice_count,
                    "positive_voxel_count": prediction.positive_voxel_count,
                    "segmented_fraction": prediction.segmented_fraction,
                }
            )


        measurements = []


        if hasattr(prediction, "positive_voxel_count"):

            measurements.append(
                MeasurementObject(
                    name="lung_segmented_voxel_volume",
                    value=prediction.positive_voxel_count,
                    unit="voxels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Physical volume requires validated spacing "
                        "and calibration."
                    ),
                )
            )

        if hasattr(prediction, "segmented_volume_mm3"):
            measurements.append(
                MeasurementObject(
                    name="lung_segmented_volume",
                    value=prediction.segmented_volume_mm3,
                    unit="mm3",
                    approximate=True,
                    physical_measurement_available=True,
                    limitation="Research-model mask volume using supplied spacing.",
                )
            )


        limitations = [
            "LUNA16 is a research dataset.",
            "Output does not confirm lung cancer.",
            "Annotated nodules are research findings only.",
            "Expert radiology review is required.",
        ]


        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),


            input=input_metadata or {
                "modality": "CT",
                "organ": "lung",
                "input_type": "DICOM/NIfTI",
            },


            specialist=SpecialistMetadata(

                specialist_id="luna16_lung_ct",

                model_id="luna16_lung_segmentation",

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline"
                ),

                dataset="LUNA16",

                task="lung segmentation",

                checkpoint=(
                    "checkpoints/luna16/"
                    "luna16_unet_best.pt"
                ),

            ),


            findings=findings,


            measurements=measurements,


            uncertainty={

                "interpretation":
                    "research output only",

                "clinical_calibration":
                    "not clinically calibrated",

            },


            limitations=limitations,


            safety=SafetyStatus(

                research_only=True,

                clinical_diagnosis=False,

                expert_review_required=True,

                message=(
                    "LUNA16 output cannot confirm "
                    "lung cancer. Expert review required."
                ),

            ),


            provenance={

                "dataset": "LUNA16",

                "model":
                    "LUNA16 Lung Segmentation Model",

            },


            artifacts={},


            expert_review=True,

        )
