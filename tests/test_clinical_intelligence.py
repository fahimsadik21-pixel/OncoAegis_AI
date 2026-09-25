from __future__ import annotations

import asyncio
import unittest
from io import BytesIO

from fastapi import UploadFile

from app.clinical.document_intelligence import analyze_clinical_document
from app.clinical.fusion import ImageEvidence, fuse_evidence
from app.main import analyze_document, analyze_fusion
from app.schemas.clinical import (
    DocumentAnalysisResponse,
    DocumentEvidenceInput,
    FusionAnalysisResponse,
    FusionRequest,
    ImageEvidenceInput,
)


class TestClinicalIntelligence(unittest.TestCase):
    def test_pathology_malignancy_is_marked_as_confirmed_evidence(self):
        result = analyze_clinical_document(
            "breast_pathology_report.txt",
            b"Specimen diagnosis: invasive carcinoma identified.",
        )
        self.assertEqual(result.document_type, "PATHOLOGY")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(
            result.evidence[0].confirmation_status,
            "pathology_confirmed_malignancy",
        )

    def test_radiology_biopsy_recommendation_is_not_pathology(self):
        result = analyze_clinical_document(
            "radiology_report.txt",
            b"CT chest: suspicious lung lesion; recommend biopsy.",
        )
        self.assertEqual(result.document_type, "RADIOLOGY")
        self.assertEqual(
            result.evidence[0].confirmation_status,
            "imaging_suspicious_not_confirmed",
        )

    def test_pathology_report_gets_plain_language_summary_and_fields(self):
        result = analyze_clinical_document(
            "breast_pathology_report.txt",
            b"Final diagnosis: Invasive ductal carcinoma. Specimen: Left breast. "
            b"Tumor size: 18 mm. Nottingham grade: 2. ER: positive. Margins: negative.",
        )
        self.assertIn("pathology", result.plain_language_summary.lower())
        self.assertTrue(result.key_findings)
        self.assertTrue(result.questions_for_care_team)
        self.assertEqual(result.structured_fields["grade"], "2")
        self.assertIn(
            "invasive ductal carcinoma",
            [term.lower() for term in result.structured_fields["histology_terms"]],
        )

    def test_image_report_has_an_honest_ocr_boundary(self):
        result = analyze_clinical_document("pathology_photo.jpg", b"upload-bytes")
        self.assertTrue(result.needs_ocr)
        self.assertEqual(result.ocr_status, "required_not_run")
        self.assertIn("image-based", result.plain_language_summary.lower())

    def test_conflicting_benign_pathology_with_suspicious_imaging_abstains(self):
        pathology = analyze_clinical_document(
            "liver_pathology.txt",
            b"Specimen diagnosis: benign lesion. Negative for malignancy.",
        )
        result = fuse_evidence(
            [
                ImageEvidence(
                    modality="CT",
                    organ="liver",
                    finding="suspicious liver lesion",
                    risk_level="suspicious",
                )
            ],
            pathology.evidence,
        )
        self.assertTrue(result.evidence_conflict)
        self.assertTrue(result.analysis.uncertainty.abstained)
        self.assertEqual(
            result.analysis.final_status,
            "evidence_conflict_definitive_conclusion_withheld",
        )

    def test_imaging_only_never_confirms_malignancy(self):
        result = fuse_evidence(
            [
                ImageEvidence(
                    modality="ULTRASOUND",
                    organ="breast",
                    finding="suspicious breast mass",
                    risk_level="high_risk",
                )
            ]
        )
        self.assertTrue(result.analysis.uncertainty.abstained)
        self.assertFalse(result.analysis.evidence_conflict)
        self.assertEqual(
            result.analysis.final_status,
            "imaging_suspicious_not_confirmed_requires_expert_review",
        )

    def test_radiology_document_suspicion_conflicts_with_benign_pathology(self):
        radiology = analyze_clinical_document(
            "radiology_report.txt",
            b"CT abdomen: suspicious liver lesion; recommend biopsy.",
        )
        pathology = analyze_clinical_document(
            "liver_pathology.txt",
            b"Specimen diagnosis: benign lesion.",
        )
        result = fuse_evidence(
            document_evidence=radiology.evidence + pathology.evidence,
        )
        self.assertTrue(result.evidence_conflict)
        self.assertTrue(result.analysis.uncertainty.abstained)

    def test_document_and_fusion_api_contracts(self):
        upload = UploadFile(
            filename="radiology_report.txt",
            file=BytesIO(b"CT chest: indeterminate lung nodule."),
        )
        document_response = asyncio.run(analyze_document(upload))
        parsed_document = DocumentAnalysisResponse.model_validate(document_response)
        self.assertEqual(parsed_document.document_type, "RADIOLOGY")

        fusion_response = analyze_fusion(
            FusionRequest(
                image_evidence=[
                    ImageEvidenceInput(
                        modality="CT",
                        organ="lung",
                        finding="suspicious lung nodule",
                        risk_level="suspicious",
                    )
                ],
                document_evidence=[
                    DocumentEvidenceInput(
                        source_type="PATHOLOGY",
                        evidence_type="malignancy",
                        statement="Pathology contains explicit malignant terminology.",
                        polarity="supports",
                        confirmation_status="pathology_confirmed_malignancy",
                    )
                ],
            )
        )
        parsed_fusion = FusionAnalysisResponse.model_validate(fusion_response)
        self.assertEqual(
            parsed_fusion.agreement,
            "pathology_supports_malignancy",
        )


if __name__ == "__main__":
    unittest.main()
