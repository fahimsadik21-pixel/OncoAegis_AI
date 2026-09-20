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


class BrainMRIAdapter(BaseSpecialistAdapter):
    """
    Converts MSD Brain Tumor MRI research output
    into StandardAnalysisResult.
    """

    adapter_id = "brain_mri_adapter"


    def metadata(self) -> dict:
        return {
            "adapter_id": self.adapter_id,
            "specialist": "brain_mri",
            "dataset": "MSD_Brain_Tumor",
        }


    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        if hasattr(prediction, "tumor_detected"):

            findings.append(
                {
                    "type": "brain_tumor_region",
                    "detected":
                        prediction.tumor_detected,
                }
            )


        if hasattr(prediction, "tumor_volume_voxels"):

            findings.append(
                {
                    "type": "tumor_volume_estimation",
                    "value":
                        prediction.tumor_volume_voxels,
                }
            )


        measurements = []


        if hasattr(prediction, "tumor_volume_voxels"):

            measurements.append(
                MeasurementObject(
                    name="tumor_volume",
                    value=prediction.tumor_volume_voxels,
                    unit="voxels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Physical volume requires validated "
                        "MRI spacing information."
                    ),
                )
            )


        limitations = [

            "MSD Brain Tumor dataset is a research dataset.",

            "Segmentation output does not confirm malignancy.",

            "MRI AI findings require expert neuroradiology review.",

        ]


        return StandardAnalysisResult(

            analysis_id=str(uuid.uuid4()),

            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),


            input=input_metadata or {

                "modality": "MRI",

                "organ": "brain",

                "input_type": "NIfTI",

            },


            specialist=SpecialistMetadata(

                specialist_id="brain_mri",

                model_id=(
                    "msd_brain_tumor_segmentation"
                ),

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline"
                ),

                dataset="MSD_Brain_Tumor",

                task="segmentation",

                checkpoint=(
                    "checkpoints/msd_brain_tumor/"
                    "msd_brain_tumor_best.pt"
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

                    "Brain MRI model output is a "
                    "research finding only. "
                    "It cannot confirm brain tumor diagnosis."

                ),

            ),


            provenance={

                "dataset":
                    "MSD_Brain_Tumor",

                "model":
                    "MSD Brain Tumor MRI Segmentation",

            },


            artifacts={},


            expert_review=True,

        )
