"""Conservative clinical-document evidence extraction.

This module is a deterministic foundation for the future medical NLP layer.
It extracts structured evidence without treating free text as a diagnosis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import fitz


class ClinicalDocumentError(ValueError):
    """Raised when a clinical document cannot be read."""


@dataclass(frozen=True)
class ClinicalEvidence:
    source_type: str
    evidence_type: str
    statement: str
    polarity: str
    confirmation_status: str
    organ: str | None = None
    matched_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "evidence_type": self.evidence_type,
            "statement": self.statement,
            "polarity": self.polarity,
            "confirmation_status": self.confirmation_status,
            "organ": self.organ,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class DocumentAnalysis:
    filename: str
    document_type: str
    text_characters: int
    needs_ocr: bool
    evidence: tuple[ClinicalEvidence, ...]
    warnings: tuple[str, ...]
    structured_fields: dict[str, object] = field(default_factory=dict)
    extraction_quality: str = "limited_rule_based"
    ocr_status: str = "not_required"
    status: str = "document_evidence_extracted"

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "document_type": self.document_type,
            "text_characters": self.text_characters,
            "needs_ocr": self.needs_ocr,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
            "structured_fields": self.structured_fields,
            "extraction_quality": self.extraction_quality,
            "ocr_status": self.ocr_status,
            "status": self.status,
        }


def _is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def _extract_text(filename: str, content: bytes) -> tuple[str, bool]:
    if not content:
        raise ClinicalDocumentError("Uploaded clinical document is empty")
    if _is_pdf(filename):
        try:
            document = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ClinicalDocumentError("Uploaded PDF cannot be opened") from exc
        page_text: list[str] = []
        for page in document:
            page_text.append(page.get_text())
        text = "\n".join(page_text).strip()
        page_count = document.page_count
        document.close()
        average = len(text) / page_count if page_count else 0.0
        return text, bool(page_count and average < 80)
    suffix = Path(filename).suffix.lower()
    if suffix != ".txt":
        raise ClinicalDocumentError("Only PDF and TXT clinical documents are supported")
    return content.decode("utf-8", errors="ignore").strip(), False


def _detect_document_type(filename: str, text: str) -> str:
    filename_signal = filename.lower()
    text_signal = text[:4000].lower()
    if re.search(
        r"pathology|histopath|cytology|biopsy[ _-]+report|biopsy[ _-]+result",
        filename_signal,
    ) or re.search(
        r"pathology[ _-]+report|histopathology|cytology[ _-]+report|"
        r"biopsy[ _-]+(?:report|result)|specimen[ _-]+diagnosis",
        text_signal,
    ):
        return "PATHOLOGY"
    if re.search(
        r"radiology|radiograph|ct[ _-]+scan|computed[ _-]+tomography|"
        r"mri|ultrasound|sonograph",
        filename_signal,
    ) or re.search(
        r"radiology[ _-]+report|imaging[ _-]+report|"
        r"impression:|ct[ _-]+(?:chest|abdomen|scan)|"
        r"computed[ _-]+tomography|ultrasound[ _-]+report",
        text_signal,
    ):
        return "RADIOLOGY"
    signal = f"{filename_signal} {text_signal}"
    if re.search(r"laboratory|laboratory test|lab report|blood test|cbc|hematology", signal):
        return "LABORATORY"
    if re.search(r"genetic|molecular|mutation|sequencing", signal):
        return "GENETIC"
    if re.search(r"clinical note|progress note|discharge|medical history", signal):
        return "CLINICAL_NOTE"
    return "UNKNOWN_DOCUMENT"


def _detect_organ(text: str) -> str | None:
    lower = text.lower()
    organs = (
        ("lung", ("lung", "pulmonary")),
        ("breast", ("breast", "mammary")),
        ("liver", ("liver", "hepatic")),
        ("kidney", ("kidney", "renal")),
        ("pancreas", ("pancreas", "pancreatic")),
        ("thyroid", ("thyroid",)),
        ("prostate", ("prostate",)),
        ("colon", ("colon", "colorectal")),
    )
    for organ, terms in organs:
        if any(term in lower for term in terms):
            return organ
    return None


def _matches(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    lower = text.lower()
    return tuple(pattern for pattern in patterns if re.search(pattern, lower))


def _extract_evidence(document_type: str, text: str) -> tuple[ClinicalEvidence, ...]:
    organ = _detect_organ(text)
    negative_malignancy = (
        r"no evidence of malignancy",
        r"negative for malignancy",
        r"negative for cancer",
        r"benign",
        r"non[- ]malignant",
        r"no malignancy identified",
    )
    positive_malignancy = (
        r"\bmalignan(?:t|cy)\b",
        r"\bcarcinoma\b",
        r"\badenocarcinoma\b",
        r"\bsarcoma\b",
        r"\blymphoma\b",
        r"\bmetast(?:asis|atic)\b",
        r"\binvasive\b",
    )
    negative_suspicion = (
        r"no suspicious lesion",
        r"no suspicious mass",
        r"no focal lesion",
        r"no pulmonary nodule",
    )
    positive_suspicion = (
        r"suspicious for malignancy",
        r"highly suspicious",
        r"suspicious lesion",
        r"indeterminate lesion",
        r"\bmass\b",
        r"\blesion\b",
        r"\bnodule\b",
    )

    negative = _matches(text, negative_malignancy)
    positive = _matches(text, positive_malignancy)
    if document_type in {"PATHOLOGY", "BIOPSY"}:
        if negative:
            return (
                ClinicalEvidence(
                    source_type=document_type,
                    evidence_type="malignancy",
                    statement="Pathology text contains benign or negative-for-malignancy evidence.",
                    polarity="refutes",
                    confirmation_status="pathology_confirmed_benign",
                    organ=organ,
                    matched_terms=negative,
                ),
            )
        if positive:
            return (
                ClinicalEvidence(
                    source_type=document_type,
                    evidence_type="malignancy",
                    statement="Pathology text contains explicit malignant/neoplastic terminology.",
                    polarity="supports",
                    confirmation_status="pathology_confirmed_malignancy",
                    organ=organ,
                    matched_terms=positive,
                ),
            )

    no_suspicion = _matches(text, negative_suspicion)
    suspicion = _matches(text, positive_suspicion)
    if document_type == "RADIOLOGY":
        if no_suspicion:
            return (
                ClinicalEvidence(
                    source_type=document_type,
                    evidence_type="lesion",
                    statement="Radiology text contains a negative suspicious-finding statement.",
                    polarity="refutes",
                    confirmation_status="imaging_no_suspicious_finding",
                    organ=organ,
                    matched_terms=no_suspicion,
                ),
            )
        if suspicion:
            return (
                ClinicalEvidence(
                    source_type=document_type,
                    evidence_type="lesion",
                    statement="Radiology text contains a lesion, mass or nodule concern.",
                    polarity="supports",
                    confirmation_status="imaging_suspicious_not_confirmed",
                    organ=organ,
                    matched_terms=suspicion,
                ),
            )

    lower = text.lower()
    general_terms = _matches(
        text,
        (
            r"tumou?r marker",
            r"oncogenic",
            r"mutation",
            r"neoplasm",
            r"cancer",
        ),
    )
    if general_terms or negative or positive:
        return (
            ClinicalEvidence(
                source_type=document_type,
                evidence_type="cancer_related_finding",
                statement="Clinical text contains cancer-related terminology requiring specialist review.",
                polarity="uncertain",
                confirmation_status="not_confirmed",
                organ=organ,
                matched_terms=general_terms + negative + positive,
            ),
        )
    return ()


def _extract_structured_fields(text: str, document_type: str) -> dict[str, object]:
    """Extract bounded fields useful for reasoning while preserving spans.

    This is intentionally deterministic. It does not invent values and does
    not turn a stage, marker, or measurement into a diagnosis.
    """

    measurements = []
    for match in re.finditer(
        r"(?P<label>lesion|mass|nodule|tumou?r|size|diameter|volume)\s*[:=-]?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|ml|cm3|mm3)",
        text,
        flags=re.IGNORECASE,
    ):
        measurements.append({
            "label": match.group("label").lower(),
            "value": float(match.group("value")),
            "unit": match.group("unit").lower(),
            "source_span": match.group(0)[:160],
        })

    biomarkers = []
    for match in re.finditer(
        r"\b(AFP|PSA|CA[- ]?125|CA[- ]?19[- ]?9|CEA|HER2|ER|PR|Ki[- ]?67|LDH)\b"
        r"\s*(?:=|:|-)\s*([<>]?\s*[\w.+-]+(?:\s*[\w/%.-]+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        biomarkers.append({
            "name": match.group(1).upper().replace(" ", ""),
            "value": match.group(2).strip(),
            "source_span": match.group(0)[:160],
        })

    stage = None
    stage_match = re.search(r"\b(?:stage|tnm)\s*[:=-]?\s*([IViv1-4][A-Za-z0-9+.-]*)", text)
    if stage_match:
        stage = stage_match.group(1).upper()

    dates = sorted(set(re.findall(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", text)))
    sections = []
    for line in text.splitlines():
        clean = " ".join(line.split())
        if clean and len(clean) <= 120 and re.match(r"^[A-Z][A-Za-z /_-]{2,40}:$", clean):
            sections.append(clean[:-1].strip().lower())

    return {
        "source_type": document_type,
        "organ": _detect_organ(text),
        "measurements": measurements[:40],
        "biomarkers": biomarkers[:40],
        "stage": stage,
        "dates": dates[:20],
        "sections": sections[:30],
    }


def analyze_clinical_document(
    filename: str,
    content: bytes,
) -> DocumentAnalysis:
    """Extract limited, structured evidence while retaining safe uncertainty."""

    text, needs_ocr = _extract_text(filename, content)
    document_type = _detect_document_type(filename, text)
    evidence = _extract_evidence(document_type, text)
    structured_fields = _extract_structured_fields(text, document_type)
    warnings = [
        "Rule-based extraction is not a diagnosis; expert clinical review is required.",
    ]
    ocr_status = "not_required"
    extraction_quality = "rule_based_structured"
    if needs_ocr:
        ocr_status = "required_not_run"
        extraction_quality = "limited_text_layer"
        warnings.append(
            "Very little embedded PDF text was found; OCR is required before relying on this document."
        )
    if not evidence:
        warnings.append("No structured cancer-related evidence was detected.")
    return DocumentAnalysis(
        filename=filename,
        document_type=document_type,
        text_characters=len(text),
        needs_ocr=needs_ocr,
        evidence=evidence,
        warnings=tuple(warnings),
        structured_fields=structured_fields,
        extraction_quality=extraction_quality,
        ocr_status=ocr_status,
    )
