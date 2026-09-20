"""Public educational cancer-information endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.cancer_info.service import get_cancer_information_service
from app.schemas.cancer_information import (
    CancerInformationRequest,
    CancerInformationResponse,
)


router = APIRouter(prefix="/information", tags=["cancer-information"])


@router.post("/cancer", response_model=CancerInformationResponse)
def ask_about_cancer(request: CancerInformationRequest) -> CancerInformationResponse:
    try:
        return get_cancer_information_service().answer(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/cancer/status")
def cancer_information_status() -> dict[str, object]:
    """Expose safe readiness metadata without exposing API keys."""

    import os

    configured = bool(
        os.getenv("ONCOAEGIS_CANCER_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    return {
        "status": "configured_ai_available" if configured else "safe_fallback_available",
        "configured_ai": configured,
        "diagnostic_conclusion": False,
        "expert_review_required": True,
    }
