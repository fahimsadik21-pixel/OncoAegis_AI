"""API schemas for document evidence and multimodal fusion."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.results import CancerAnalysisResult
from app.schemas.imaging import BUSIModelPrediction, CTSeriesInspection, LUNACTModelPrediction


class ClinicalEvidenceResponse(BaseModel):
    source_type: str
    evidence_type: str
    statement: str
    polarity: str
    confirmation_status: str
    organ: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class DocumentAnalysisResponse(BaseModel):
    filename: str
    document_type: str
    text_characters: int
    needs_ocr: bool
    evidence: list[ClinicalEvidenceResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    structured_fields: dict[str, object] = Field(default_factory=dict)
    extraction_quality: str = "limited_rule_based"
    ocr_status: str = "not_required"
    status: str = "document_evidence_extracted"
    next_stage: str = "evidence_fusion"


class ImageEvidenceInput(BaseModel):
    modality: str
    finding: str = Field(min_length=1)
    organ: str | None = None
    risk_level: str = "indeterminate"
    source: str = "imaging_model"
    evidence_kind: str = "clinical_finding"


class DocumentEvidenceInput(BaseModel):
    source_type: str
    evidence_type: str = "cancer_related_finding"
    statement: str = Field(min_length=1)
    polarity: str = "uncertain"
    confirmation_status: str = "not_confirmed"
    organ: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class FusionRequest(BaseModel):
    image_evidence: list[ImageEvidenceInput] = Field(default_factory=list)
    document_evidence: list[DocumentEvidenceInput] = Field(default_factory=list)


class FusionAnalysisResponse(BaseModel):
    analysis: CancerAnalysisResult
    agreement: str
    evidence_conflict: bool
    conflict_description: str | None = None
    reasoning: dict[str, object] = Field(default_factory=dict)


class MultimodalUltrasoundResponse(BaseModel):
    image_filename: str
    prediction: BUSIModelPrediction
    document: DocumentAnalysisResponse | None = None
    fusion: FusionAnalysisResponse
    status: str = "multimodal_analysis_complete"
    next_stage: str = "expert_review"


class MultimodalCTResponse(BaseModel):
    filenames_received: int
    inspection: CTSeriesInspection
    prediction: LUNACTModelPrediction
    document: DocumentAnalysisResponse | None = None
    fusion: FusionAnalysisResponse
    status: str = "multimodal_analysis_complete"
    next_stage: str = "expert_review"
