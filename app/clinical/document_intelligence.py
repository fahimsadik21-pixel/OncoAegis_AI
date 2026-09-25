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
    plain_language_summary: str = ""
    key_findings: tuple[str, ...] = ()
    questions_for_care_team: tuple[str, ...] = ()

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
            "plain_language_summary": self.plain_language_summary,
            "key_findings": list(self.key_findings),
            "questions_for_care_team": list(self.questions_for_care_team),
        }


def _is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def _is_raster_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg"}


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
    if _is_raster_image(filename):
        # The deployment intentionally does not invent text from an image.  A
        # later OCR service can fill this boundary without changing the API.
        return "", True
    if suffix != ".txt":
        raise ClinicalDocumentError("Supported report files are PDF, TXT, PNG, JPG, and JPEG")
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

    diagnosis_lines = []
    for line in text.splitlines():
        clean = " ".join(line.split())
        if clean and re.search(r"\b(final diagnosis|diagnosis|impression|conclusion|comment)\b", clean, re.IGNORECASE):
            diagnosis_lines.append(clean[:300])

    specimen = []
    for match in re.finditer(r"\b(?:specimen|tissue|site)\s*[:=-]\s*([^\n]{2,160})", text, re.IGNORECASE):
        specimen.append(match.group(1).strip())

    histology = []
    for match in re.finditer(
        r"\b(?:invasive ductal carcinoma|invasive lobular carcinoma|adenocarcinoma|"
        r"squamous cell carcinoma|carcinoma in situ|lymphoma|sarcoma|melanoma|"
        r"neuroendocrine tumor|benign [a-z -]{2,60})\b",
        text,
        re.IGNORECASE,
    ):
        histology.append(match.group(0))

    grade_match = re.search(r"\b(?:nottingham\s+)?grade\s*[:=-]?\s*([1-3I]{1,8})", text, re.IGNORECASE)
    margin_match = re.search(r"\b(margins?\s*(?:are|:)?\s*(?:negative|positive|clear|involved|free)[^\n.]{0,100})", text, re.IGNORECASE)
    node_match = re.search(r"\b(\d+\s*(?:of|/)\s*\d+\s*(?:lymph\s+)?nodes?[^\n.]{0,100})", text, re.IGNORECASE)

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
        "diagnosis_lines": diagnosis_lines[:8],
        "specimen": list(dict.fromkeys(specimen))[:8],
        "histology_terms": list(dict.fromkeys(histology))[:12],
        "grade": grade_match.group(1).upper() if grade_match else None,
        "margin_statement": margin_match.group(1).strip() if margin_match else None,
        "lymph_node_statement": node_match.group(1).strip() if node_match else None,
    }


def _report_explanation(
    document_type: str,
    structured_fields: dict[str, object],
    evidence: tuple[ClinicalEvidence, ...],
    needs_ocr: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Turn only extracted phrases into a plain-language, non-diagnostic guide."""

    if needs_ocr:
        return (
            "This report appears to be image-based, so readable text was not available for safe automated explanation. Upload a text-searchable PDF or TXT copy, or ask the issuing clinic for a digital report. The original image has not been interpreted as a diagnosis.",
            ("No report text was extracted from this image-based file.",),
            ("Can I have a text-searchable PDF or the written report?", "Which clinician should explain the original report with me?"),
        )

    statuses = {item.confirmation_status for item in evidence}
    if "pathology_confirmed_malignancy" in statuses:
        summary = (
            "The extracted pathology wording includes explicit malignant or neoplastic terminology. "
            "Pathology is strong evidence, but the signed report and pathologist should confirm the exact type, site, grade, margins, biomarkers, and stage-related details."
        )
    elif "pathology_confirmed_benign" in statuses:
        summary = (
            "The extracted pathology wording includes benign or negative-for-malignancy language. "
            "This should still be confirmed against the signed report and considered together with imaging and the sampled site."
        )
    elif "imaging_suspicious_not_confirmed" in statuses:
        summary = (
            "The report contains a radiology phrase describing a lesion, mass, nodule, or suspicious pattern. "
            "Imaging wording alone does not confirm cancer; the report's recommendation and any required pathology or follow-up imaging matter."
        )
    elif "imaging_no_suspicious_finding" in statuses:
        summary = (
            "The extracted radiology wording includes a negative suspicious-finding phrase. "
            "It should be read with the full report, the reason for the scan, and the clinician's assessment."
        )
    else:
        summary = (
            "The available report text was organized into structured details, but no definitive cancer-related statement was safely extracted. "
            "The original report and the ordering clinician remain the source for interpretation."
        )

    findings: list[str] = []
    for item in evidence:
        findings.append(item.statement)
    for value in structured_fields.get("diagnosis_lines", []):
        findings.append(f"Report line: {value}")
    for value in structured_fields.get("histology_terms", []):
        findings.append(f"Mentioned tissue term: {value}")
    stage = structured_fields.get("stage")
    if stage:
        findings.append(f"Reported stage expression: {stage} (not independently interpreted)")
    for measurement in structured_fields.get("measurements", [])[:3]:
        if isinstance(measurement, dict):
            findings.append(f"Reported measurement: {measurement.get('source_span', 'measurement mentioned')}")
    if not findings:
        findings.append("No diagnosis line, measurement, stage, or evidence phrase was reliably extracted.")

    if document_type == "PATHOLOGY":
        questions = (
            "What is the exact final diagnosis and the organ/site sampled?",
            "Do grade, margins, lymph nodes, or biomarkers change the next step?",
            "Does this pathology agree with the imaging and clinical findings?",
        )
    elif document_type == "RADIOLOGY":
        questions = (
            "Which imaging finding is most important, and how concerning is it?",
            "Does the report recommend follow-up imaging, a specialist review, or biopsy?",
            "How does this scan compare with previous imaging?",
        )
    else:
        questions = (
            "Which lines of this report matter most for my care?",
            "Does this result need repeat testing, a specialist review, or pathology?",
            "What result or symptom would need urgent attention?",
        )
    return summary, tuple(dict.fromkeys(findings))[:8], questions


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
        if _is_raster_image(filename):
            warnings.append("This image-based report needs OCR or a text-searchable copy before its wording can be explained.")
        else:
            warnings.append("Very little embedded PDF text was found; OCR is required before relying on this document.")
    if not evidence:
        warnings.append("No structured cancer-related evidence was detected.")
    summary, key_findings, questions = _report_explanation(document_type, structured_fields, evidence, needs_ocr)
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
        plain_language_summary=summary,
        key_findings=key_findings,
        questions_for_care_team=questions,
    )
