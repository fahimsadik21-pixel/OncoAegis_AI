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



class PancreasCTAdapter(BaseSpecialistAdapter):
    """
    Converts MSD Pancreas CT research output
    into StandardAnalysisResult.
    """

    adapter_id = "pancreas_ct_adapter"



    def metadata(self) -> dict:

        return {
            "adapter_id": self.adapter_id,
            "specialist": "pancreas_ct",
            "dataset": "MSD_Pancreas",
        }



    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        # -----------------------------
        # Pancreas segmentation finding
        # -----------------------------

        findings.append(
            {
                "type": "pancreas_segmentation",
                "detected": getattr(
                    prediction,
                    "pancreas_detected",
                    False,
                ),
                "pancreas_voxels": getattr(
                    prediction,
                    "pancreas_voxels",
                    0,
                ),
            }
        )



        # -----------------------------
        # Cancer/tumor region finding
        # -----------------------------

        findings.append(
            {
                "type":
                    "pancreatic_tumor_region_research_finding",

                "detected":
                    getattr(
                        prediction,
                        "cancer_detected",
                        False,
                    ),

                "cancer_voxels":
                    getattr(
                        prediction,
                        "cancer_voxels",
                        0,
                    ),
            }
        )



        measurements = []



        if hasattr(
            prediction,
            "pancreas_voxels",
        ):

            measurements.append(
                MeasurementObject(
                    name="pancreas_voxel_volume",
                    value=prediction.pancreas_voxels,
                    unit="voxels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Voxel count only. "
                        "Physical volume requires CT spacing."
                    ),
                )
            )



        if hasattr(
            prediction,
            "cancer_voxels",
        ):

            measurements.append(
                MeasurementObject(
                    name="cancer_region_voxel_volume",
                    value=prediction.cancer_voxels,
                    unit="voxels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Voxel count only. "
                        "Physical tumor volume requires validated spacing."
                    ),
                )
            )



        limitations = [

            "MSD Pancreas is a research dataset.",

            "Segmentation findings do not confirm pancreatic cancer.",

            "Expert radiology review is required.",

        ]



        return StandardAnalysisResult(


            analysis_id=str(uuid.uuid4()),


            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),



            input=input_metadata or {

                "modality": "CT",

                "organ": "pancreas",

                "input_type": "DICOM/NIfTI",

            },



            specialist=SpecialistMetadata(

                specialist_id="pancreas_ct",

                model_id=(
                    "msd_pancreas_tumor_segmentation"
                ),

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline",
                ),

                dataset="MSD_Pancreas",

                task=(
                    "two-stage pancreas "
                    "and tumor segmentation"
                ),

                checkpoint=(
                    "checkpoints/msd_pancreas_tumor_fast/"
                    "pancreas_tumor_fast_best.pt"
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
                    "Pancreas CT AI output is research evidence only. "
                    "It cannot confirm pancreatic cancer."
                ),

            ),



            provenance={

                "dataset":
                    "MSD_Pancreas",

                "model":
                    "MSD Task07 Two-Stage Pancreas Analysis",

                "crop_bounds":
                    getattr(
                        prediction,
                        "crop_bounds",
                        None,
                    ),

                "original_shape":
                    getattr(
                        prediction,
                        "original_shape",
                        None,
                    ),

            },



            artifacts={},

            expert_review=True,

        )
