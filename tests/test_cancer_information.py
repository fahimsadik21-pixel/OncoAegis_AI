from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.api.cancer_information_api import ask_about_cancer, cancer_information_status
from app.schemas.cancer_information import CancerInformationRequest
from app.cancer_info.service import CancerInformationService, normalize_topic


class CancerInformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = CancerInformationService()

    def test_known_topic_normalizes_to_registry_match(self):
        topic = normalize_topic("lung")
        self.assertEqual(topic.key, "lung_cancer")
        self.assertEqual(topic.registry_key, "lung_cancer")

    def test_fallback_answers_in_english_with_safety_fields_and_sources(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.service.answer(
                CancerInformationRequest(
                    cancer="breast cancer",
                    question="What are common symptoms and how is it evaluated?",
                )
            )
        self.assertEqual(response.mode, "safe_fallback")
        self.assertFalse(response.diagnostic_conclusion)
        self.assertTrue(response.expert_review_required)
        self.assertGreaterEqual(len(response.sections), 4)
        self.assertTrue(any("cancer.gov" in source.url for source in response.sources))
        self.assertTrue(any("medlineplus.gov" in source.url for source in response.sources))

    def test_bangla_request_returns_bangla_content(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.service.answer(
                CancerInformationRequest(
                    cancer="লিভার ক্যান্সার",
                    question="লক্ষণ এবং পরীক্ষা সম্পর্কে জানতে চাই",
                    language="bn",
                )
            )
        self.assertEqual(response.mode, "safe_fallback")
        self.assertTrue(any("\u0980" <= char <= "\u09ff" for char in response.answer))
        self.assertFalse(response.diagnostic_conclusion)

    def test_unknown_cancer_is_still_educational_and_not_diagnostic(self):
        with patch.dict(os.environ, {}, clear=True):
            response = ask_about_cancer(
                CancerInformationRequest(
                    cancer="a rare cancer subtype",
                    question="Can you explain treatment options?",
                )
            )
        self.assertEqual(response.cancer, "A rare cancer subtype")
        self.assertFalse(response.diagnostic_conclusion)
        self.assertTrue(response.expert_review_required)

    def test_status_never_claims_diagnosis(self):
        with patch.dict(os.environ, {}, clear=True):
            status = cancer_information_status()
        self.assertEqual(status["status"], "safe_fallback_available")
        self.assertFalse(status["configured_ai"])
        self.assertFalse(status["diagnostic_conclusion"])

    def test_provider_failure_degrades_to_safe_fallback(self):
        with patch.dict(os.environ, {"ONCOAEGIS_CANCER_AI_API_KEY": "test-key"}, clear=True), patch(
            "app.cancer_info.service.urlopen", side_effect=OSError("provider unavailable")
        ):
            response = self.service.answer(
                CancerInformationRequest(
                    cancer="thyroid cancer",
                    question="What should I know about testing?",
                )
            )
        self.assertEqual(response.mode, "safe_fallback")
        self.assertFalse(response.diagnostic_conclusion)


if __name__ == "__main__":
    unittest.main()
