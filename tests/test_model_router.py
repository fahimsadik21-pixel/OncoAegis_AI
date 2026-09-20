from __future__ import annotations

import unittest

from app.core.model_router import route_specialist_model
from app.main import app, cancer_capabilities, models, route_model, system_status
from app.schemas.routing import ModelRouteRequest, ModelRouteResponse


class TestModelRouter(unittest.TestCase):
    def test_selects_luna_segmentation_from_aliases(self):
        decision = route_specialist_model(
            modality="ct",
            organ="Lung",
            task="segment",
            require_available=False,
        )
        self.assertEqual(decision.route_status, "model_selected")
        self.assertEqual(decision.selected_model_id, "luna16_lung_segmentation")
        self.assertFalse(decision.malignancy_confirmation)

    def test_selects_busi_classifier_for_breast_capability(self):
        decision = route_specialist_model(
            modality="Ultrasound",
            organ="breast",
            task="classification",
            cancer_type="breast-cancer",
            require_available=False,
        )
        self.assertEqual(decision.route_status, "model_selected")
        self.assertEqual(decision.selected_model_id, "busi_breast_classifier")
        self.assertEqual(
            decision.safety_status,
            "research_model_requires_expert_review",
        )

    def test_requires_task_when_multiple_models_match(self):
        decision = route_specialist_model(
            modality="CT",
            organ="lung",
            require_available=False,
        )
        self.assertEqual(decision.route_status, "task_required")
        self.assertIsNone(decision.selected_model_id)
        self.assertEqual(
            set(decision.candidate_model_ids),
            {"luna16_lung_segmentation", "luna16_nodule_detector"},
        )

    def test_registered_liver_specialist_is_selected(self):
        decision = route_specialist_model(
            modality="CT",
            organ="liver",
            cancer_type="liver_cancer",
            task="segmentation",
            require_available=False,
        )
        self.assertEqual(decision.route_status, "model_selected")
        self.assertEqual(
            decision.selected_model_id,
            "ircadb01_liver_tumor_segmentation",
        )

    def test_route_api_contract_and_registry_endpoints(self):
        response = route_model(
            ModelRouteRequest(
                modality="US",
                organ="breast",
                task="classification",
                cancer_type="breast_cancer",
                require_available=False,
            )
        )
        parsed = ModelRouteResponse.model_validate(response)
        self.assertEqual(parsed.selected_model_id, "busi_breast_classifier")
        def paths(routes):
            values = set()
            for route in routes:
                path = getattr(route, "path", None)
                if path:
                    values.add(path)
                nested = getattr(route, "routes", None)
                if nested:
                    values.update(paths(nested))
            return values

        self.assertIn("/route/model", paths(app.routes))
        model_ids = {entry["model_id"] for entry in models()["models"]}
        self.assertIn("ircadb01_liver_tumor_segmentation", model_ids)
        self.assertGreaterEqual(len(model_ids), 13)
        self.assertIn("lung_cancer", cancer_capabilities()["capabilities"])

    def test_system_status_reports_baseline_readiness_and_safety(self):
        status = system_status()
        self.assertEqual(status["status"], "ready_for_registered_baselines")
        self.assertEqual(status["components"]["dicom_series_engine"], "ready")
        self.assertEqual(status["components"]["clinical_document_evidence"], "rule_based_foundation")
        self.assertFalse(status["safety"]["imaging_alone_confirms_malignancy"])


if __name__ == "__main__":
    unittest.main()
