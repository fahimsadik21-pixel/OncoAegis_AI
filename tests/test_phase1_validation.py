import unittest

import torch

from app.registry.model_registry import get_model_registry
from evaluation.phase1_validation import (
    EVALUATION_SPECS,
    _binary_auroc,
    _classification_report,
    _segmentation_report,
    evaluate_model,
)


class TestPhase1Validation(unittest.TestCase):
    def test_every_registered_model_has_a_phase1_spec(self):
        registered = {spec.model_id for spec in get_model_registry().list()}
        self.assertEqual(registered, set(EVALUATION_SPECS))

    def test_segmentation_metrics_handle_empty_and_positive_cases(self):
        prediction = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[1, 1], [0, 0]],
            ],
            dtype=torch.bool,
        )
        target = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
            ],
            dtype=torch.bool,
        )
        report = _segmentation_report(prediction, target)
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["positive_sample_count"], 1)
        self.assertAlmostEqual(report["metrics"]["dice"], (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(report["positive_case_metrics"]["dice"], 2.0 / 3.0)

    def test_binary_auroc_is_none_without_both_classes(self):
        self.assertIsNone(_binary_auroc([0.1, 0.2], [0, 0]))
        self.assertAlmostEqual(_binary_auroc([0.1, 0.9], [0, 1]), 1.0)

    def test_classification_report_is_json_safe(self):
        report = _classification_report(
            torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
            torch.tensor([0, 1]),
            ("negative", "positive"),
        )
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["auroc"], 1.0)

    def test_flowcap_is_not_marked_evaluable_without_ground_truth(self):
        report = evaluate_model("flowcap_aml_patient_classifier", device="cpu")
        self.assertEqual(report["status"], "not_evaluable")
        self.assertFalse(report["ground_truth_available"])
        self.assertEqual(report["metrics"], {})


if __name__ == "__main__":
    unittest.main()
