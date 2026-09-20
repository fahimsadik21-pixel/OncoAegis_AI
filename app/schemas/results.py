from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Modality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    ULTRASOUND = "ULTRASOUND"
    DOCUMENT = "DOCUMENT"
    TEXT = "TEXT"
    UNKNOWN = "UNKNOWN"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class Finding(BaseModel):
    organ: Optional[str] = None
    finding_type: str
    location: Optional[str] = None

    detection_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    classification: Optional[str] = None

    size_mm: Optional[float] = None
    volume_mm3: Optional[float] = None

    segmentation_available: bool = False


class Evidence(BaseModel):
    source: str
    description: str
    supports_finding: Optional[bool] = None


class Uncertainty(BaseModel):
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    abstained: bool = False
    reason: Optional[str] = None


class CancerAnalysisResult(BaseModel):
    modality: Modality

    body_region: Optional[str] = None
    suspected_organ: Optional[str] = None

    findings: List[Finding] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)

    evidence_conflict: bool = False
    conflict_description: Optional[str] = None

    uncertainty: Uncertainty = Field(default_factory=Uncertainty)

    final_status: str = "analysis_pending"

    model_used: Optional[str] = None
    model_version: Optional[str] = None
