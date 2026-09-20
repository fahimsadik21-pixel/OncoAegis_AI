from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image
import SimpleITK as sitk

from src.data.busi_dataset import BUSIDataset, discover_busi_samples
from src.data.loaders import create_luna16_loaders
from src.data.luna_dataset import LUNA16SliceDataset
from src.data.luna_nodule_dataset import LUNA16NoduleSliceDataset
from src.data.splits import split_ids
from src.data.registry import get_dataset_registry
from src.data.universal import StandardMedicalSample, UniversalDataset
from evaluation.metrics import classification_report
from src.models.busi_models import BUSIMultiTaskUNet
from src.models.unet import UNet
from app.imaging.ultrasound.busi_inference import BUSIModelService, run_busi_ultrasound
from app.schemas.imaging import UltrasoundAnalysisResponse


class TestDatasetPipelines(unittest.TestCase):
    def test_luna_slice_dataset_binary_masks_and_case_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case_id in ("case_a", "case_b", "case_c"):
                np.save(root / f"{case_id}_ct.npy", np.zeros((2, 8, 8), dtype=np.float32))
                mask = np.zeros((2, 8, 8), dtype=np.uint8)
                mask[1, 2:4, 2:4] = 5
                np.save(root / f"{case_id}_mask.npy", mask)

            dataset = LUNA16SliceDataset(root, image_size=None)
            self.assertEqual(len(dataset), 6)
            _, target = dataset[1]
            self.assertEqual(set(torch.unique(target).tolist()), {0.0, 1.0})
            train, validation = split_ids(("case_a", "case_b", "case_c"), seed=7)
            self.assertTrue(set(train).isdisjoint(validation))

    def test_luna_nodule_dataset_builds_balanced_centered_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_id = "case_a"
            np.save(root / f"{case_id}_ct.npy", np.zeros((2, 32, 32), dtype=np.float32))
            mask = np.ones((2, 32, 32), dtype=np.uint8)
            np.save(root / f"{case_id}_mask.npy", mask)
            raw_image = sitk.GetImageFromArray(np.zeros((2, 32, 32), dtype=np.int16))
            raw_image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(raw_image, str(root / f"{case_id}.mhd"))
            (root / "annotations.csv").write_text(
                "seriesuid,coordX,coordY,coordZ,diameter_mm\ncase_a,12,10,0,4\n",
                encoding="utf-8",
            )
            dataset = LUNA16NoduleSliceDataset(
                root,
                root / "annotations.csv",
                root,
                image_size=(32, 32),
                patch_size=16,
                negative_ratio=1,
            )
            labels = [int(dataset[index][1]) for index in range(len(dataset))]
            self.assertEqual(labels.count(1), 1)
            self.assertEqual(labels.count(0), 1)
            self.assertEqual(dataset[0][0].shape, (1, 32, 32))

    def test_busi_merges_multiple_masks_and_returns_class_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for class_name in ("normal", "benign", "malignant"):
                (root / class_name).mkdir()
            image = np.full((10, 12), 100, dtype=np.uint8)
            Image.fromarray(image).save(root / "benign" / "sample.png")
            first = np.zeros_like(image)
            first[1:3, 1:3] = 255
            second = np.zeros_like(image)
            second[6:8, 7:9] = 255
            Image.fromarray(first).save(root / "benign" / "sample_mask.png")
            Image.fromarray(second).save(root / "benign" / "sample_mask_1.png")

            samples = discover_busi_samples(root)
            dataset = BUSIDataset(root, samples=samples, image_size=(8, 8))
            item = dataset[0]
            self.assertEqual(item["image"].shape, (1, 8, 8))
            self.assertEqual(item["mask"].shape, (1, 8, 8))
            self.assertEqual(int(item["label"]), 1)
            self.assertGreater(float(item["mask"].sum()), 0.0)

    def test_dataloaders_have_expected_batch_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                case_id = f"case_{index}"
                np.save(root / f"{case_id}_ct.npy", np.zeros((2, 16, 16), dtype=np.float32))
                mask = np.zeros((2, 16, 16), dtype=np.uint8)
                mask[:, 3:8, 3:8] = 1
                np.save(root / f"{case_id}_mask.npy", mask)
            loaders = create_luna16_loaders(root, batch_size=2, image_size=(16, 16), seed=3)
            image, mask = next(iter(loaders.train))
            self.assertEqual(image.shape, (2, 1, 16, 16))
            self.assertEqual(mask.shape, (2, 1, 16, 16))

    def test_models_preserve_spatial_size(self):
        image = torch.randn(2, 1, 32, 32)
        unet = UNet(base_channels=4)
        self.assertEqual(unet(image).shape, (2, 1, 32, 32))
        multitask = BUSIMultiTaskUNet(base_channels=4)
        outputs = multitask(image)
        self.assertEqual(outputs["segmentation_logits"].shape, (2, 1, 32, 32))
        self.assertEqual(outputs["classification_logits"].shape, (2, 3))

    def test_classification_report_contains_confusion_matrix(self):
        logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        target = torch.tensor([0, 1, 2])
        report = classification_report(logits, target)
        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(report["macro_f1"], 1.0)

    def test_busi_inference_service_preprocesses_and_predicts(self):
        class DummyModel(torch.nn.Module):
            def forward(self, image):
                batch_size, _, height, width = image.shape
                segmentation = torch.zeros((batch_size, 1, height, width), device=image.device)
                classification = torch.tensor([[0.0, 2.0, 0.0]], device=image.device).repeat(batch_size, 1)
                return {"segmentation_logits": segmentation, "classification_logits": classification}

        buffer = BytesIO()
        Image.fromarray(np.full((20, 30), 120, dtype=np.uint8)).save(buffer, format="PNG")
        prediction = run_busi_ultrasound(
            buffer.getvalue(),
            service=BUSIModelService(model=DummyModel(), image_size=16, device="cpu"),
        )
        self.assertEqual(prediction.original_image_shape, (20, 30))
        self.assertEqual(prediction.predicted_class, "benign")
        self.assertEqual(prediction.processed_image_shape, (16, 16))

    def test_ultrasound_endpoint_returns_safe_review_result(self):
        import asyncio
        import app.main as app_main

        class DummyModel(torch.nn.Module):
            def forward(self, image):
                batch_size, _, height, width = image.shape
                segmentation = torch.zeros((batch_size, 1, height, width), device=image.device)
                classification = torch.tensor([[0.0, 2.0, 0.0]], device=image.device).repeat(batch_size, 1)
                return {"segmentation_logits": segmentation, "classification_logits": classification}

        buffer = BytesIO()
        Image.fromarray(np.full((20, 30), 120, dtype=np.uint8)).save(buffer, format="PNG")
        previous_service = app_main._busi_service
        app_main._busi_service = BUSIModelService(model=DummyModel(), image_size=16, device="cpu")
        try:
            body = asyncio.run(app_main.analyze_ultrasound(UploadFile(filename="sample.png", file=BytesIO(buffer.getvalue()))))
        finally:
            app_main._busi_service = previous_service
        parsed = UltrasoundAnalysisResponse.model_validate(body)
        self.assertEqual(parsed.prediction.predicted_class, "benign")
        self.assertEqual(parsed.analysis.modality, "ULTRASOUND")
        self.assertTrue(parsed.analysis.uncertainty.abstained)
        self.assertEqual(parsed.next_stage, "expert_review")

    def test_dataset_registry_and_universal_sample_contract(self):
        registry = get_dataset_registry()
        names = {entry.name for entry in registry.list()}
        self.assertTrue({"DeepLesion", "FLARE", "BUSI", "LUNA16"}.issubset(names))
        self.assertGreaterEqual(len(names), 4)
        for entry in registry.list():
            self.assertTrue(entry.expected_size)
            self.assertTrue(entry.license)
        universal = UniversalDataset(
            torch.utils.data.TensorDataset(torch.zeros((1, 1, 4, 4)), torch.ones((1, 1, 4, 4))),
            dataset_name="LUNA16",
            modality="CT",
            organ="lung",
        )
        sample = universal[0]
        self.assertIsInstance(sample, StandardMedicalSample)
        self.assertEqual(sample.dataset_name, "LUNA16")
        self.assertIsNotNone(sample.mask)


if __name__ == "__main__":
    unittest.main()
