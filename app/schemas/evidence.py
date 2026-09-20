from typing import Any, Optional
from pydantic import BaseModel, Field


class EvidenceObject(BaseModel):
    """
    Universal evidence representation.

    Every medical statement in OncoAegis
    should preserve provenance.
    """

    evidence_id: str

    source_type: str
    """
    Examples:
    - imaging_model
    - pathology_report
    - radiology_report
    - laboratory_report
    - user_input
    - knowledge_database
    """

    source_reference: Optional[str] = None

    statement: str

    normalized_finding: Optional[str] = None


    polarity: str = "uncertain"
    """
    present
    absent
    uncertain
    conflicting
    """


    certainty: Optional[str] = None

    date: Optional[str] = None

    location: Optional[str] = None


    supporting_text_span: Optional[str] = None


    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )


    model_derived: bool = False

    clinically_confirmed: bool = False


    provenance: dict[str, Any] = Field(
        default_factory=dict
    )


    limitations: list[str] = Field(
        default_factory=list
    )