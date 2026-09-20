from __future__ import annotations

import unittest

from app.registry.cancer_registry import get_cancer_capabilities
from app.registry.model_registry import get_all_models, get_model


class TestRegistries(unittest.TestCase):
    def test_models(self):
        models = get_all_models()
        self.assertIn("luna16_lung_segmentation", models)
        self.assertIn("busi_breast_classifier", models)
        self.assertEqual(
            models["luna16_nodule_detector"]["input"],
            "nodule-centred 2D CT patch",
        )

    def test_cancers(self):
        cancers = get_cancer_capabilities()
        self.assertIn("lung_cancer", cancers)
        self.assertIn("blood_cancer", cancers)
        self.assertIn(
            "ircadb01_liver_tumor_segmentation",
            cancers["liver_cancer"]["models"],
        )

    def test_legacy_single_model_lookup_remains_serializable(self):
        model = get_model("busi_breast_classifier")
        self.assertIsNotNone(model)
        self.assertEqual(model["modality"], "ULTRASOUND")
        self.assertFalse(model["clinical_validation"])


if __name__ == "__main__":
    unittest.main()
