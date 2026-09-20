"""Safety-aware fusion of imaging and clinical-document evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.schemas.results import (
    CancerAnalysisResult,
    Evidence,
    Finding,
    Modality,
    Uncertainty,
)
from app.clinical.document_intelligence import ClinicalEvidence
from app.clinical.reasoning import build_medical_reasoning


@dataclass(frozen=True)
class ImageEvidence:
    modality: str
    finding: str
    organ: str | None = None
    risk_level: str = "indeterminate"
    source: str = "imaging_model"
    evidence_kind: str = "clinical_finding"


@dataclass(frozen=True)
class FusionResult:
    analysis: CancerAnalysisResult
    agreement: str
    evidence_conflict: bool
    conflict_description: str | None
    reasoning: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "analysis": self.analysis.model_dump(mode="json"),
            "agreement": self.agreement,
            "evidence_conflict": self.evidence_conflict,
            "conflict_description": self.conflict_description,
            "reasoning": self.reasoning,
        }


def _modality(value: str) -> Modality:
    normalized = value.strip().upper()
    try:
        return Modality(normalized)
    except ValueError:
        return Modality.UNKNOWN


def _image_is_suspicious(item: ImageEvidence) -> bool:
    if item.evidence_kind.strip().lower() in {"technical", "anatomy"}:
        return False
    risk = item.risk_level.strip().lower()
    finding = item.finding.lower()
    return risk in {"suspicious", "high_risk", "high-risk", "indeterminate"} or any(
        term in finding for term in ("lesion", "mass", "nodule", "suspicious")
    )


def _normalise_organ(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(value.strip().lower().replace("_", " ").split()) or None


def _relevant_documents(
    documents: tuple[ClinicalEvidence, ...],
    images: tuple[ImageEvidence, ...],
) -> tuple[ClinicalEvidence, ...]:
    """Keep evidence for the same known organ when an image target is known."""

    image_organs = {
        organ
        for organ in (_normalise_organ(item.organ) for item in images)
        if organ
    }
    if not image_organs:
        return documents
    return tuple(
        item
        for item in documents
        if _normalise_organ(item.organ) is None
        or _normalise_organ(item.organ) in image_organs
    )


def fuse_evidence(
    image_evidence: Iterable[ImageEvidence] = (),
    document_evidence: Iterable[ClinicalEvidence] = (),
) -> FusionResult:
    """Fuse evidence and explicitly withhold conclusions on conflict."""

    images = tuple(image_evidence)
    documents = _relevant_documents(tuple(document_evidence), images)
    suspicious_images = tuple(item for item in images if _image_is_suspicious(item))
    suspicious_documents = tuple(
        item
        for item in documents
        if item.confirmation_status == "imaging_suspicious_not_confirmed"
        and item.polarity == "supports"
    )
    negative_images = tuple(
        item
        for item in images
        if item.risk_level.strip().lower()
        in {"normal", "negative", "no_suspicious_finding"}
    )
    negative_documents = tuple(
        item
        for item in documents
        if item.confirmation_status == "imaging_no_suspicious_finding"
        and item.polarity == "refutes"
    )
    malignant_pathology = tuple(
        item
        for item in documents
        if item.confirmation_status == "pathology_confirmed_malignancy"
        and item.polarity == "supports"
    )
    benign_pathology = tuple(
        item
        for item in documents
        if item.confirmation_status == "pathology_confirmed_benign"
        and item.polarity == "refutes"
    )
    conflict = bool(
        (suspicious_images or suspicious_documents) and benign_pathology
        or (malignant_pathology and benign_pathology)
        or (negative_images or negative_documents) and malignant_pathology
    )

    if conflict:
        agreement = "conflict"
        final_status = "evidence_conflict_definitive_conclusion_withheld"
        conflict_description = (
            "Imaging/document evidence disagrees; definitive conclusion is withheld."
        )
        uncertainty = Uncertainty(
            abstained=True,
            reason="Conflicting evidence requires expert adjudication.",
        )
    elif malignant_pathology:
        agreement = "pathology_supports_malignancy"
        final_status = "pathology_confirmed_malignancy_evidence_requires_expert_review"
        conflict_description = None
        uncertainty = Uncertainty(
            abstained=False,
            reason="Malignancy status is supported only by explicit pathology evidence.",
        )
    elif suspicious_images or suspicious_documents:
        agreement = "imaging_only_or_unconfirmed"
        final_status = "imaging_suspicious_not_confirmed_requires_expert_review"
        conflict_description = None
        uncertainty = Uncertainty(
            abstained=True,
            reason="Imaging can indicate suspicion but cannot confirm malignancy.",
        )
    elif benign_pathology:
        agreement = "pathology_supports_benignity"
        final_status = "pathology_benign_evidence_requires_expert_review"
        conflict_description = None
        uncertainty = Uncertainty(
            abstained=False,
            reason="Interpretation remains subject to clinical review.",
        )
    else:
        agreement = "insufficient_or_indeterminate_evidence"
        final_status = "insufficient_evidence_requires_expert_review"
        conflict_description = None
        uncertainty = Uncertainty(
            abstained=True,
            reason="No definitive evidence combination was detected.",
        )

    first_image = images[0] if images else None
    modality = _modality(first_image.modality) if first_image else Modality.UNKNOWN
    organ = first_image.organ if first_image else None
    findings = [
        Finding(
            organ=item.organ,
            finding_type="imaging_evidence",
            classification=item.risk_level,
        )
        for item in images
    ]
    evidence = [
        Evidence(
            source=item.source,
            description=item.finding,
            supports_finding=True,
        )
        for item in images
    ]
    evidence.extend(
        Evidence(
            source=item.source_type,
            description=item.statement,
            supports_finding=(
                True if item.polarity == "supports"
                else False if item.polarity == "refutes"
                else None
            ),
        )
        for item in documents
    )
    analysis = CancerAnalysisResult(
        modality=modality,
        suspected_organ=organ,
        findings=findings,
        evidence=evidence,
        evidence_conflict=conflict,
        conflict_description=conflict_description,
        uncertainty=uncertainty,
        final_status=final_status,
        model_used=(
            ", ".join(sorted({item.source for item in images}))
            if images
            else None
        ),
    )
    reasoning = build_medical_reasoning(
        images=images,
        documents=documents,
        agreement=agreement,
        conflict=conflict,
    )
    return FusionResult(
        analysis=analysis,
        agreement=agreement,
        evidence_conflict=conflict,
        conflict_description=conflict_description,
        reasoning=reasoning,
    )
