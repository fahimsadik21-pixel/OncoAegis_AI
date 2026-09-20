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



class LiverCTAdapter(BaseSpecialistAdapter):
    """
    Converts IRCADb01 Liver CT research output
    into StandardAnalysisResult.
    """

    adapter_id = "liver_ct_adapter"



    def metadata(self) -> dict:

        return {
            "adapter_id": self.adapter_id,
            "specialist": "liver_ct",
            "dataset": "IRCADb01",
        }




    def convert(
        self,
        prediction,
        *,
        input_metadata: dict | None = None,
    ) -> StandardAnalysisResult:


        findings = []


        findings.append(
            {
                "type": "liver_segmentation",

                "detected": getattr(
                    prediction,
                    "liver_detected",
                    False,
                ),

                "liver_voxels": getattr(
                    prediction,
                    "liver_voxels",
                    0,
                ),
            }
        )



        findings.append(
            {
                "type":
                    "liver_tumor_region_research_finding",

                "detected": getattr(
                    prediction,
                    "tumor_detected",
                    False,
                ),

                "tumor_voxels": getattr(
                    prediction,
                    "tumor_voxels",
                    0,
                ),
            }
        )



        measurements = []



        if hasattr(
            prediction,
            "liver_voxels",
        ):

            measurements.append(
                MeasurementObject(
                    name="liver_voxel_volume",
                    value=prediction.liver_voxels,
                    unit="voxels",
                    approximate=True,
                    physical_measurement_available=False,
                    limitation=(
                        "Voxel count only. "
                        "Physical liver volume requires CT spacing."
                    ),
                )
            )



        if hasattr(
            prediction,
            "tumor_voxels",
        ):

            measurements.append(
                MeasurementObject(
                    name="liver_tumor_voxel_volume",
                    value=prediction.tumor_voxels,
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

            "IRCADb01 is a research dataset.",

            "Segmentation output does not confirm liver cancer.",

            "Automated liver CT findings require expert review.",

        ]



        return StandardAnalysisResult(


            analysis_id=str(uuid.uuid4()),


            timestamp=datetime.now(
                timezone.utc
            ).isoformat(),



            input=input_metadata or {

                "modality": "CT",

                "organ": "liver",

                "input_type": "DICOM/NIfTI",

            },



            specialist=SpecialistMetadata(

                specialist_id="liver_ct",

                model_id=(
                    "ircadb01_liver_tumor_segmentation"
                ),

                model_version=getattr(
                    prediction,
                    "model_version",
                    "research-baseline",
                ),

                dataset="IRCADb01",

                task=(
                    "liver and tumor segmentation"
                ),

                checkpoint=(
                    "checkpoints/liver_tumor_stage2/"
                    "liver_tumor_stage2_best.pt"
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
                    "Liver CT AI output is research evidence only. "
                    "It cannot confirm liver cancer."
                ),

            ),



            provenance={

                "dataset":
                    "IRCADb01",

                "model":
                    "IRCADb01 Liver CT Tumor Segmentation",

                "crop_bounds":
                    getattr(
                        prediction,
                        "crop_bounds",
                        None,
                    ),

                "locator_fallback_used":
                    getattr(
                        prediction,
                        "locator_fallback_used",
                        None,
                    ),

            },



            artifacts={},


            expert_review=True,

        )