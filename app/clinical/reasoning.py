"""Evidence-first medical reasoning for multimodal result presentation.

This layer explains available evidence and safe next steps. It never upgrades
an imaging finding into a cancer diagnosis and abstains when evidence is weak
or conflicting.
"""

from __future__ import annotations

from typing import Iterable

from app.clinical.document_intelligence import ClinicalEvidence


def build_medical_reasoning(
    *,
    images: Iterable[object] = (),
    documents: Iterable[ClinicalEvidence] = (),
    agreement: str,
    conflict: bool,
) -> dict[str, object]:
    image_items = tuple(images)
    document_items = tuple(documents)
    sources = []
    if image_items:
        sources.append("imaging_model")
    sources.extend(sorted({item.source_type.lower() for item in document_items}))

    if conflict:
        interpretation = "Evidence sources disagree; automated definitive interpretation is withheld."
        disposition = "expert_review_required"
        urgency = "review_promptly_with_the_relevant_clinician"
    elif any(item.confirmation_status == "pathology_confirmed_malignancy" for item in document_items):
        interpretation = "The supplied document contains explicit pathology terminology; imaging remains supportive context."
        disposition = "pathology_evidence_present_expert_review_required"
        urgency = "follow_the_treating_team_or_pathology_service"
    elif image_items or document_items:
        interpretation = "Available evidence describes findings or concern, but it does not establish malignancy on its own."
        disposition = "research_output_requires_expert_review"
        urgency = "discuss_with_a_qualified_clinician"
    else:
        interpretation = "There is not enough structured evidence for a reliable interpretation."
        disposition = "insufficient_evidence"
        urgency = "provide_relevant_report_or_clinical_context"

    return {
        "interpretation": interpretation,
        "agreement": agreement,
        "evidence_sources": sources,
        "evidence_count": len(image_items) + len(document_items),
        "disposition": disposition,
        "recommended_next_step": urgency,
        "questions_for_expert_review": [
            "Which finding is clinically relevant after reviewing the original images and report?",
            "Is pathology or another confirmatory test available and appropriate?",
            "What follow-up interval is appropriate for this individual case?",
        ],
        "safety": {
            "research_only": True,
            "diagnosis_confirmed": False,
            "imaging_alone_confirms_cancer": False,
            "automated_treatment_advice": False,
        },
    }
