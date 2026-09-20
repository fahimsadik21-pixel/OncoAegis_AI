"""API schemas for medical imaging inspection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.results import CancerAnalysisResult, Modality


class CTSeriesInspection(BaseModel):
    input_type: str = "dicom_ct_series"
    detected_modality: Modality = Modality.CT
    series_slices: int = Field(ge=1)
    volume_shape: tuple[int, int, int]
    body_region: str | None = None
    pixel_spacing_mm: tuple[float, float]
    slice_spacing_mm: float | None = None
    slice_thickness_mm: float | None = None
    ordering_method: str
    volume_ready: bool = True
    preprocessing_applied: bool = True
    preprocessed_volume_shape: tuple[int, int, int]
    preprocessed_spacing_mm: tuple[float, float, float]
    hu_window: tuple[float, float]
    clipped_fraction: float = Field(ge=0.0, le=1.0)
    candidate_organs: tuple[str, ...] = ()
    anatomy_route_status: str
    patient_identifiers_in_response: bool = False
    warnings: list[str] = Field(default_factory=list)


class CTSeriesAnalysisResponse(BaseModel):
    filenames_received: int
    inspection: CTSeriesInspection
    safe_metadata: dict[str, Any]
    status: str = "input_inspected"
    next_stage: str = "anatomy_model_router"


class LUNACTModelPrediction(BaseModel):
    input_volume_shape: tuple[int, int, int]
    output_mask_shape: tuple[int, int, int]
    model_input_size: tuple[int, int]
    positive_slice_count: int = Field(ge=0)
    positive_voxel_count: int = Field(ge=0)
    segmented_fraction: float = Field(ge=0.0, le=1.0)
    segmented_volume_mm3: float = Field(ge=0.0)
    model_name: str
    model_version: str
    warnings: list[str] = Field(default_factory=list)


class CTSpecialistAnalysisResponse(BaseModel):
    filenames_received: int
    inspection: CTSeriesInspection
    prediction: LUNACTModelPrediction
    analysis: CancerAnalysisResult
    status: str = "ct_specialist_prediction"
    next_stage: str = "expert_review"


class UltrasoundInspection(BaseModel):
    input_type: str = "ultrasound_image"
    detected_modality: Modality = Modality.ULTRASOUND
    original_image_shape: tuple[int, int]
    processed_image_shape: tuple[int, int]
    segmentation_available: bool = True
    lesion_area_fraction: float = Field(ge=0.0, le=1.0)
    patient_identifiers_in_response: bool = False
    warnings: list[str] = Field(default_factory=list)


class BUSIModelPrediction(BaseModel):
    predicted_class: str
    class_probabilities: dict[str, float]
    model_confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str


class UltrasoundAnalysisResponse(BaseModel):
    filename: str
    inspection: UltrasoundInspection
    prediction: BUSIModelPrediction
    analysis: CancerAnalysisResult
    status: str = "ultrasound_model_prediction"
    next_stage: str = "expert_review"
