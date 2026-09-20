from typing import Any, Optional
from pydantic import BaseModel, Field


class StandardMedicalSample(BaseModel):
    """
    Universal input contract for all medical AI specialists.
    """

    sample_id: str

    case_id: Optional[str] = None

    patient_id: Optional[str] = None

    modality: str

    anatomy: Optional[str] = None

    task: Optional[str] = None

    input_type: str

    image_path: Optional[str] = None

    volume_path: Optional[str] = None

    mask_path: Optional[str] = None

    label: Optional[Any] = None

    spacing: Optional[list[float]] = None

    orientation: Optional[list[float]] = None

    origin: Optional[list[float]] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    source: Optional[str] = None

    provenance: dict[str, Any] = Field(
        default_factory=dict
    )

    extras: dict[str, Any] = Field(
        default_factory=dict
    )