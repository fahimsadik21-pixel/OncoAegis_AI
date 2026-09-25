from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from app.deployment.checkpoint_bootstrap import checkpoint_is_ready
from app.deployment.resource_policy import model_eviction_enabled
from app.registry.model_registry import ModelSpec


class TestDeploymentPolicy(unittest.TestCase):
    def test_git_lfs_pointer_is_not_accepted_as_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:test\n")
            self.assertFalse(checkpoint_is_ready(path))

    def test_real_weight_sized_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"model-state" * 150)
            self.assertTrue(checkpoint_is_ready(path))

    def test_lfs_pointer_is_not_reported_as_a_loadable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:example\nsize 999999\n",
                encoding="utf-8",
            )
            with patch("app.registry.model_registry._PROJECT_ROOT", Path(directory)):
                spec = ModelSpec(
                    model_id="temporary_model",
                    name="Temporary model",
                    modality="CT",
                    organ="lung",
                    tasks=("segmentation",),
                    input_type="test",
                    outputs=("test",),
                    checkpoint_path="checkpoint.pt",
                )
                self.assertFalse(spec.checkpoint_ready)
                self.assertEqual(spec.checkpoint_status, "checkpoint_missing")

    def test_multistage_model_requires_every_registered_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_two = root / "checkpoints" / "liver_tumor_stage2"
            stage_two.mkdir(parents=True)
            (stage_two / "liver_tumor_stage2_best.pt").write_bytes(
                b"model-state" * 150
            )
            with patch("app.registry.model_registry._PROJECT_ROOT", root):
                spec = ModelSpec(
                    model_id="ircadb01_liver_tumor_segmentation",
                    name="Temporary liver model",
                    modality="CT",
                    organ="liver",
                    tasks=("segmentation",),
                    input_type="test",
                    outputs=("test",),
                    checkpoint_path=(
                        "checkpoints/liver_tumor_stage2/"
                        "liver_tumor_stage2_best.pt"
                    ),
                )
                self.assertFalse(spec.checkpoint_ready)
                self.assertEqual(spec.checkpoint_status, "checkpoint_missing")

    def test_hosted_platform_defaults_to_model_eviction(self):
        with patch.dict(
            os.environ,
            {"RAILWAY_ENVIRONMENT": "production"},
            clear=True,
        ):
            self.assertTrue(model_eviction_enabled())

    def test_local_platform_keeps_models_warm_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(model_eviction_enabled())


if __name__ == "__main__":
    unittest.main()
