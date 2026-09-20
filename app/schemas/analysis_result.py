from typing import Any, Optional
from pydantic import BaseModel, Field


class MeasurementObject(BaseModel):
    """
    Universal automated measurement output.
    """

    name: str

    value: Optional[Any] = None

    unit: Optional[str] = None

    approximate: bool = True

    physical_measurement_available: bool = False

    limitation: Optional[str] = None



class SafetyStatus(BaseModel):
    """
    Safety and clinical limitation layer.
    """

    research_only: bool = True

    clinical_diagnosis: bool = False

    expert_review_required: bool = True

    message: str



class SpecialistMetadata(BaseModel):
    """
    Model provenance.
    """

    specialist_id: str

    model_id: Optional[str] = None

    model_version: Optional[str] = None

    dataset: Optional[str] = None

    checkpoint: Optional[str] = None

    task: Optional[str] = None



class StandardAnalysisResult(BaseModel):
    """
    Universal final output contract of OncoAegis AI.
    All specialists should eventually return this format.
    """

    analysis_id: str

    case_id: Optional[str] = None

    timestamp: Optional[str] = None


    input: dict[str, Any] = Field(
        default_factory=dict
    )


    specialist: Optional[SpecialistMetadata] = None


    findings: list[Any] = Field(
        default_factory=list
    )


    measurements: list[MeasurementObject] = Field(
        default_factory=list
    )


    evidence: list[Any] = Field(
        default_factory=list
    )


    uncertainty: dict[str, Any] = Field(
        default_factory=dict
    )


    limitations: list[str] = Field(
        default_factory=list
    )


    conflicts: list[str] = Field(
        default_factory=list
    )


    safety: Optional[SafetyStatus] = None


    provenance: dict[str, Any] = Field(
        default_factory=dict
    )


    artifacts: dict[str, Any] = Field(
        default_factory=dict
    )


    expert_review: bool = True