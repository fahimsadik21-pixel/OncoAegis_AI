from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile
from PIL import Image

from app.schemas.analysis_result import (
    SafetyStatus,
    SpecialistMetadata,
    StandardAnalysisResult,
)
from app.storage.case_store import (
    CaseDeletedError,
    CaseStore,
    InvalidCaseIDError,
    compute_input_hash,
    set_case_store,
)


def _result(analysis_id: str, *, modality: str = "ULTRASOUND"):
    return StandardAnalysisResult(
        analysis_id=analysis_id,
        input={"modality": modality, "organ": "breast"},
        specialist=SpecialistMetadata(
            specialist_id="test-specialist",
            model_id="busi_breast_classifier",
            model_version="research-test-v1",
            checkpoint="checkpoints/missing.pt",
            dataset="BUSI",
        ),
        findings=[{"type": "research_finding", "detected": False}],
        safety=SafetyStatus(
            research_only=True,
            clinical_diagnosis=False,
            expert_review_required=True,
            message="Research output; expert review required.",
        ),
    )


def _upload(name: str = "image.png", content: bytes = b"image-bytes"):
    return SimpleNamespace(name=name, content=content)


class TestCaseStore(unittest.TestCase):
    def _store(self):
        directory = tempfile.TemporaryDirectory()
        store = CaseStore(Path(directory.name) / "cases.sqlite3")
        self.addCleanup(directory.cleanup)
        return store

    def test_store_result_tracks_hashes_model_version_and_audit(self):
        store = self._store()
        saved = store.store_result(
            _result("analysis-1"),
            files=[_upload()],
            actor_id="reviewer-1",
            case_id="case-1",
        )

        self.assertEqual(saved["case"]["case_id"], "case-1")
        self.assertEqual(saved["case"]["result_count"], 1)
        self.assertEqual(saved["result"]["model_version"], "research-test-v1")
        self.assertEqual(len(saved["result"]["input_hash"]), 64)
        self.assertEqual(saved["result"]["input_hash_algorithm"], "sha256")

        loaded = store.get_result("case-1", "analysis-1")
        self.assertEqual(loaded["result"]["case_id"], "case-1")
        self.assertEqual(
            loaded["result"]["provenance"]["input_hash"],
            saved["result"]["input_hash"],
        )
        actions = {
            event["action"] for event in store.list_audit("case-1")["events"]
        }
        self.assertIn("case.created", actions)
        self.assertIn("result.stored", actions)
        self.assertTrue(store.verify_audit_chain()["valid"])

    def test_case_history_accepts_multiple_results_and_tracks_context(self):
        store = self._store()
        store.store_result(
            _result("analysis-1"),
            files=[_upload()],
            case_id="case-history",
        )
        second = _result("analysis-2", modality="CT")
        store.store_result(
            second,
            files=[_upload("volume.npy", b"volume")],
            case_id="case-history",
        )

        history = store.get_case("case-history")
        self.assertEqual(history["case"]["result_count"], 2)
        self.assertEqual(history["case"]["modality"], "MULTIMODAL")
        self.assertEqual(len(history["results"]), 2)

    def test_delete_purges_result_payload_but_keeps_tombstone_and_audit(self):
        store = self._store()
        store.store_result(
            _result("analysis-delete"),
            files=[_upload()],
            case_id="case-delete",
        )

        deleted = store.delete_case(
            "case-delete",
            actor_id="privacy-admin",
            reason="user_requested",
        )
        self.assertEqual(deleted["status"], "deleted")
        self.assertTrue(deleted["data_purged"])
        with self.assertRaises(CaseDeletedError):
            store.get_result("case-delete", "analysis-delete")

        tombstone = store.get_case("case-delete", include_deleted=True)
        self.assertEqual(tombstone["case"]["result_count"], 0)
        self.assertTrue(tombstone["case"]["data_purged"])
        actions = {
            event["action"] for event in store.list_audit("case-delete")["events"]
        }
        self.assertIn("case.deleted", actions)
        self.assertTrue(store.verify_audit_chain()["valid"])

    def test_retention_job_purges_expired_cases(self):
        store = self._store()
        store.store_result(
            _result("analysis-retention"),
            files=[_upload()],
            case_id="case-retention",
        )
        assert store.db_path is not None
        connection = sqlite3.connect(store.db_path)
        try:
            connection.execute(
                "UPDATE cases SET retention_expires_at = ? WHERE case_id = ?",
                ("2000-01-01T00:00:00+00:00", "case-retention"),
            )
            connection.commit()
        finally:
            connection.close()

        report = store.purge_expired_cases(now="2026-01-01T00:00:00+00:00")
        self.assertEqual(report["purged_case_ids"], ["case-retention"])
        self.assertTrue(store.verify_audit_chain()["valid"])

    def test_audit_tampering_is_detected(self):
        store = self._store()
        store.store_result(
            _result("analysis-tamper"),
            files=[_upload()],
            case_id="case-tamper",
        )
        assert store.db_path is not None
        connection = sqlite3.connect(store.db_path)
        try:
            connection.execute(
                "UPDATE audit_events SET event_hash = ? WHERE sequence = 1",
                ("tampered",),
            )
            connection.commit()
        finally:
            connection.close()
        verification = store.verify_audit_chain()
        self.assertFalse(verification["valid"])
        self.assertEqual(verification["reason"], "event_hash_mismatch")

    def test_case_id_validation_and_filename_privacy(self):
        store = self._store()
        with self.assertRaises(InvalidCaseIDError):
            store.store_result(
                _result("analysis-invalid"),
                files=[_upload("patient-secret.png")],
                case_id="../patient",
            )
        first = compute_input_hash([_upload("patient-A.png", b"same")])
        second = compute_input_hash([_upload("patient-B.png", b"same")])
        self.assertEqual(first, second)

    def test_unified_specialist_endpoint_persists_result(self):
        import app.main as app_main

        store = self._store()
        set_case_store(store)
        self.addCleanup(lambda: set_case_store(None))

        class FakeOrchestrator:
            def execute_analysis(self, *, model_id, input_data, input_metadata):
                return _result("analysis-api")

        async def run_endpoint():
            image = BytesIO()
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image, format="PNG")
            return await app_main.analyze_specialist(
                model_id="busi_breast_classifier",
                files=[
                    UploadFile(
                        filename="breast.png",
                        file=BytesIO(image.getvalue()),
                    )
                ],
                modality="ULTRASOUND",
                organ="breast",
                task=None,
                cancer_type=None,
                case_id="case-api",
                spacing_mm=None,
                actor_id="api-test",
            )

        with patch.object(app_main, "AnalysisOrchestrator", FakeOrchestrator):
            result = asyncio.run(run_endpoint())

        self.assertEqual(result.case_id, "case-api")
        stored = store.get_result("case-api", "analysis-api")
        self.assertEqual(stored["storage"]["model_id"], "busi_breast_classifier")


if __name__ == "__main__":
    unittest.main()
