from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from app.adapters.flowcap_adapter import FlowCAPAdapter
from app.adapters.registry import list_adapters
from app.adapters.tn3k_adapter import TN3KAdapter
from app.core.execution_engine import ExecutionEngine
from app.registry.model_registry import get_model_registry


class TestSpecialistIntegration(unittest.TestCase):
    def test_every_registered_model_has_an_adapter(self):
        adapters = list_adapters()
        model_ids = {spec.model_id for spec in get_model_registry().list()}
        self.assertEqual(model_ids, set(adapters))

    def test_execution_engine_passes_spacing_and_selected_model_id(self):
        class FakeLiverService:
            def analyze(self, volume, *, spacing_mm):
                self.spacing_mm = spacing_mm
                return SimpleNamespace(
                    liver_detected=True,
                    tumor_detected=False,
                    liver_voxels=12,
                    tumor_voxels=0,
                    model_version="test",
                    crop_bounds=None,
                    spacing_seen=self.spacing_mm,
                )

        from app.adapters.liver_ct_adapter import LiverCTAdapter

        service = FakeLiverService()
        with patch(
            "app.core.execution_engine.get_or_create_specialist",
            return_value=service,
        ), patch(
            "app.core.execution_engine.get_adapter",
            return_value=LiverCTAdapter(),
        ):
            result = ExecutionEngine().execute(
                model_id="ircadb01_liver_tumor_segmentation",
                input_data=np.zeros((2, 2, 2), dtype=np.float32),
                input_metadata={"spacing_mm": (2.0, 1.5, 1.0)},
            )

        self.assertEqual(service.spacing_mm, (2.0, 1.5, 1.0))
        self.assertEqual(
            result.specialist.model_id,
            "ircadb01_liver_tumor_segmentation",
        )
        self.assertFalse(result.safety.clinical_diagnosis)

    def test_execution_engine_releases_cached_model_on_constrained_host(self):
        class FakeLiverService:
            def analyze(self, volume, *, spacing_mm):
                return SimpleNamespace(
                    liver_detected=True,
                    tumor_detected=False,
                    liver_voxels=12,
                    tumor_voxels=0,
                    model_version="test",
                    crop_bounds=None,
                )

        from app.adapters.liver_ct_adapter import LiverCTAdapter

        with patch(
            "app.core.execution_engine.get_or_create_specialist",
            return_value=FakeLiverService(),
        ), patch(
            "app.core.execution_engine.get_adapter",
            return_value=LiverCTAdapter(),
        ), patch(
            "app.core.execution_engine.model_eviction_enabled",
            return_value=True,
        ), patch(
            "app.core.execution_engine.release_specialist",
        ) as release_specialist:
            ExecutionEngine().execute(
                model_id="ircadb01_liver_tumor_segmentation",
                input_data=np.zeros((2, 2, 2), dtype=np.float32),
            )

        release_specialist.assert_called_once_with(
            "ircadb01_liver_tumor_segmentation"
        )

    def test_flowcap_mapping_adapter_preserves_scores(self):
        result = FlowCAPAdapter().convert(
            {
                "model": {"name": "DREAM6 test"},
                "prediction": {
                    "combined_score": 0.25,
                    "tube_scores": [0.1] * 8,
                },
            }
        )
        self.assertEqual(
            result.findings[0]["type"],
            "patient_level_research_score",
        )
        self.assertEqual(result.findings[0]["value"], 0.25)
        self.assertFalse(result.safety.clinical_diagnosis)

    def test_tn3k_adapter_accepts_universal_metadata(self):
        result = TN3KAdapter().convert(
            {
                "model": {
                    "id": "tn3k_thyroid_nodule_segmentation",
                    "dataset": "TN3K",
                },
                "segmentation": {
                    "detected_region": True,
                    "mean_probability_inside_predicted_region": 0.8,
                },
                "measurements": {"area_pixels": 10},
                "input": {"modality": "ULTRASOUND"},
            },
            input_metadata={"modality": "ULTRASOUND", "organ": "thyroid"},
        )
        self.assertEqual(result.input["organ"], "thyroid")
        self.assertEqual(result.measurements[0].value, 10)
        self.assertFalse(result.safety.clinical_diagnosis)


if __name__ == "__main__":
    unittest.main()
