"""API contracts for general cancer-information questions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CancerInformationRequest(BaseModel):
    cancer: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=3, max_length=2400)
    language: Literal["en", "bn"] = "en"


class InformationSection(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=5000)
    bullets: list[str] = Field(default_factory=list, max_length=12)


class InformationSource(BaseModel):
    title: str
    url: str
    source_type: str


class CancerInformationResponse(BaseModel):
    request_id: str
    cancer: str
    answer: str
    sections: list[InformationSection]
    follow_up_questions: list[str]
    urgent_guidance: str
    sources: list[InformationSource]
    mode: Literal["safe_fallback", "offline_knowledge_base", "configured_ai"]
    specialist_registry_match: bool
    diagnostic_conclusion: bool = False
    expert_review_required: bool = True
