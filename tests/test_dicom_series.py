from __future__ import annotations

import asyncio
from io import BytesIO
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi import UploadFile
from pydicom import dcmwrite
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from app.imaging.dicom.loader import (
    DicomSeriesError,
    load_dicom_series,
    load_dicom_series_from_bytes,
)
from app.core.specialist_input import (
    UploadedSpecialistFile,
    normalize_specialist_input,
)
from app.imaging.ct.preprocessing import preprocess_ct_volume
from app.imaging.ct.pipeline import run_ct_pipeline_from_bytes
from app.imaging.ct.luna_inference import LUNA16ModelService, run_luna_ct_segmentation
from app.main import app
from app.core.anatomy_router import route_body_region
from app.core.router import detect_input_modality
from app.core.input_inspector import inspect_input
from app.schemas.results import Modality
from app.schemas.imaging import CTSeriesAnalysisResponse, CTSpecialistAnalysisResponse
from app.schemas.clinical import MultimodalCTResponse
from scripts.inspect_ct_series import build_safe_summary


def _make_dicom_bytes(
    *,
    z_position: float,
    value: int,
    series_uid: str,
    instance_number: int,
    study_uid: str | None = None,
) -> bytes:
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(
        "slice.dcm",
        {},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False

    dataset.PatientName = "Sensitive^Patient"
    dataset.PatientID = "SECRET-ID"
    dataset.PatientBirthDate = "19700101"
    dataset.Modality = "CT"
    dataset.StudyInstanceUID = study_uid or generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPInstanceUID = sop_uid
    dataset.BodyPartExamined = "CHEST"
    dataset.Manufacturer = "Test Scanner"
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.PixelSpacing = ["0.7", "0.7"]
    dataset.SliceThickness = "1.0"
    dataset.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
    dataset.ImagePositionPatient = ["0", "0", str(z_position)]
    dataset.InstanceNumber = instance_number
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = "2"
    dataset.RescaleIntercept = "-100"
    dataset.PixelData = np.full((2, 2), value, dtype=np.int16).tobytes()

    output = BytesIO()
    dcmwrite(output, dataset, write_like_original=False)
    return output.getvalue()


def _series_files(series_uid: str | None = None):
    series_uid = series_uid or generate_uid()
    study_uid = generate_uid()
    return [
        (
            "slice_z2.dcm",
            _make_dicom_bytes(
                z_position=2,
                value=20,
                series_uid=series_uid,
                instance_number=3,
                study_uid=study_uid,
            ),
        ),
        (
            "slice_z0.dcm",
            _make_dicom_bytes(
                z_position=0,
                value=10,
                series_uid=series_uid,
                instance_number=1,
                study_uid=study_uid,
            ),
        ),
        (
            "slice_z1.dcm",
            _make_dicom_bytes(
                z_position=1,
                value=15,
                series_uid=series_uid,
                instance_number=2,
                study_uid=study_uid,
            ),
        ),
    ]


class TestDicomSeries(unittest.TestCase):
    def test_dicom_series_is_sorted_and_converted_to_hu(self):
        series = load_dicom_series_from_bytes(_series_files())

        self.assertEqual(series.modality, "CT")
        self.assertEqual(series.shape, (3, 2, 2))
        self.assertEqual(
            series.ordering_method,
            "image_position_patient_projection",
        )
        self.assertAlmostEqual(series.slice_spacing, 1.0)
        self.assertEqual(series.pixel_spacing, (0.7, 0.7))
        self.assertEqual(
            series.volume[:, 0, 0].tolist(),
            [-80.0, -70.0, -60.0],
        )
        self.assertEqual(series.safe_metadata["body_region"], "CHEST")
        self.assertFalse(
            series.safe_metadata["patient_identifiers_in_response"]
        )
        self.assertNotIn("PatientName", series.safe_metadata)
        self.assertNotIn("PatientID", series.safe_metadata)


    def test_specialist_normalizer_accepts_extensionless_dicom_and_marks_hu(self):
        files = [
            UploadedSpecialistFile(filename.removesuffix(".dcm"), data)
            for filename, data in _series_files()
        ]
        with tempfile.TemporaryDirectory() as workspace:
            normalized = normalize_specialist_input(
                model_id="ircadb01_liver_tumor_segmentation",
                files=files,
                workspace=workspace,
                modality="CT",
                organ="liver",
            )

        self.assertEqual(normalized.metadata["input_format"], "DICOM")
        self.assertEqual(normalized.metadata["intensity_units"], "HU")
        self.assertTrue(normalized.metadata["hu_conversion_applied"])
        self.assertEqual(normalized.data[:, 0, 0].tolist(), [-80.0, -70.0, -60.0])


    def test_multiple_series_are_rejected(self):
        files = _series_files()
        files.append(
            (
                "other_series.dcm",
                _make_dicom_bytes(
                    z_position=0,
                    value=1,
                    series_uid=generate_uid(),
                    instance_number=1,
                    study_uid=generate_uid(),
                ),
            )
        )

        with self.assertRaisesRegex(DicomSeriesError, "multiple DICOM series"):
            load_dicom_series_from_bytes(files)


    def test_directory_loader_ignores_sidecar_files(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            for filename, data in _series_files():
                (tmp_path / filename).write_bytes(data)
            (tmp_path / "README.txt").write_text(
                "not a DICOM file",
                encoding="utf-8",
            )

            series = load_dicom_series(tmp_path)

            self.assertEqual(series.shape, (3, 2, 2))
            self.assertEqual(series.safe_metadata["slice_count"], 3)


    def test_ct_series_endpoint_returns_safe_summary(self):
        uploads = [
            UploadFile(
                filename=filename,
                file=BytesIO(data),
            )
            for filename, data in _series_files()
        ]

        body = asyncio.run(apply_ct_endpoint(uploads))
        parsed = CTSeriesAnalysisResponse.model_validate(body)
        self.assertEqual(body["inspection"]["detected_modality"], "CT")
        self.assertEqual(body["inspection"]["volume_shape"], (3, 2, 2))
        self.assertTrue(body["inspection"]["volume_ready"])
        self.assertEqual(parsed.status, "ct_volume_preprocessed")
        self.assertEqual(parsed.next_stage, "anatomy_model_router")
        self.assertEqual(
            parsed.inspection.preprocessed_volume_shape,
            (3, 2, 2),
        )
        self.assertEqual(
            parsed.inspection.candidate_organs,
            ("lung",),
        )
        self.assertFalse(
            body["safe_metadata"]["patient_identifiers_in_response"]
        )

    def test_ct_pipeline_produces_model_safe_input(self):
        result = run_ct_pipeline_from_bytes(_series_files())
        model_input = result.model_input

        self.assertEqual(model_input["volume"].shape, (3, 2, 2))
        self.assertEqual(model_input["candidate_organs"], ("lung",))
        self.assertNotIn("PatientName", model_input["metadata"])

    def test_luna_inference_returns_geometry_aware_anatomy_summary(self):
        class DummyModel(torch.nn.Module):
            def forward(self, image):
                return torch.ones(
                    (image.shape[0], 1, image.shape[2], image.shape[3]),
                    device=image.device,
                )

        prediction = run_luna_ct_segmentation(
            np.zeros((3, 2, 2), dtype=np.float32),
            spacing_mm=(1.0, 0.7, 0.7),
            service=LUNA16ModelService(
                model=DummyModel(),
                image_size=16,
                batch_size=2,
                device="cpu",
            ),
        )

        self.assertEqual(prediction.input_volume_shape, (3, 2, 2))
        self.assertEqual(prediction.positive_slice_count, 3)
        self.assertEqual(prediction.positive_voxel_count, 12)
        self.assertAlmostEqual(prediction.segmented_volume_mm3, 5.88)
        self.assertTrue(any("not nodule" in warning for warning in prediction.warnings))

    def test_ct_specialist_endpoint_connects_luna_output(self):
        import app.main as app_main

        class DummyModel(torch.nn.Module):
            def forward(self, image):
                return torch.zeros(
                    (image.shape[0], 1, image.shape[2], image.shape[3]),
                    device=image.device,
                )

        uploads = [
            UploadFile(filename=filename, file=BytesIO(data))
            for filename, data in _series_files()
        ]
        previous_service = app_main._luna_service
        app_main._luna_service = LUNA16ModelService(
            model=DummyModel(), image_size=16, batch_size=2, device="cpu"
        )
        try:
            body = asyncio.run(app_main.analyze_ct_specialist(uploads))
        finally:
            app_main._luna_service = previous_service

        parsed = CTSpecialistAnalysisResponse.model_validate(body)
        self.assertEqual(parsed.prediction.input_volume_shape, (3, 2, 2))
        self.assertEqual(parsed.analysis.suspected_organ, "lung")
        self.assertEqual(
            parsed.analysis.final_status,
            "lung_anatomy_segmented_requires_lesion_review",
        )
        self.assertTrue(parsed.analysis.uncertainty.abstained)

    def test_multimodal_ct_endpoint_fuses_report_without_confirming_from_image(self):
        import app.main as app_main

        class DummyModel(torch.nn.Module):
            def forward(self, image):
                return torch.zeros(
                    (image.shape[0], 1, image.shape[2], image.shape[3]),
                    device=image.device,
                )

        uploads = [
            UploadFile(filename=filename, file=BytesIO(data))
            for filename, data in _series_files()
        ]
        document = UploadFile(
            filename="lung_pathology.txt",
            file=BytesIO(b"Specimen diagnosis: benign lesion."),
        )
        previous_service = app_main._luna_service
        app_main._luna_service = LUNA16ModelService(
            model=DummyModel(), image_size=16, batch_size=2, device="cpu"
        )
        try:
            body = asyncio.run(
                app_main.analyze_multimodal_ct_series(
                    files=uploads,
                    document=document,
                )
            )
        finally:
            app_main._luna_service = previous_service

        parsed = MultimodalCTResponse.model_validate(body)
        self.assertEqual(parsed.document.document_type, "PATHOLOGY")
        self.assertEqual(
            parsed.fusion.analysis.final_status,
            "pathology_benign_evidence_requires_expert_review",
        )

    def test_cli_summary_is_json_safe_and_non_diagnostic(self):
        result = run_ct_pipeline_from_bytes(_series_files())
        summary = build_safe_summary(result)

        self.assertEqual(summary["status"], "ct_volume_preprocessed")
        self.assertEqual(summary["volume_shape"], [3, 2, 2])
        self.assertEqual(summary["candidate_organs"], ["lung"])
        self.assertIsNone(summary["selected_organ"])
        self.assertFalse(summary["patient_identifiers_in_response"])
        self.assertNotIn("PatientName", summary)
        self.assertNotIn("PatientID", summary)

    def test_ct_preprocessing_resamples_and_normalizes(self):
        volume = np.array(
            [[[-200.0, 40.0], [240.0, 500.0]]],
            dtype=np.float32,
        )

        processed = preprocess_ct_volume(
            volume,
            pixel_spacing_mm=(1.0, 1.0),
            slice_spacing_mm=2.0,
            target_spacing_mm=(1.0, 0.5, 0.5),
            window_center=40.0,
            window_width=400.0,
        )

        self.assertEqual(processed.original_shape, (1, 2, 2))
        self.assertEqual(processed.output_shape, (2, 4, 4))
        self.assertEqual(processed.output_spacing_mm, (1.0, 0.5, 0.5))
        self.assertGreaterEqual(float(processed.volume.min()), 0.0)
        self.assertLessEqual(float(processed.volume.max()), 1.0)
        self.assertGreater(processed.clipped_fraction, 0.0)

    def test_anatomy_router_remains_conservative(self):
        route = route_body_region("chest")

        self.assertEqual(route.body_region, "CHEST")
        self.assertEqual(route.candidate_organs, ("lung",))
        self.assertIsNone(route.selected_organ)
        self.assertEqual(route.route_status, "requires_anatomy_model")

    def test_filename_router_does_not_guess_medical_modality(self):
        self.assertEqual(detect_input_modality("scan.dcm"), Modality.UNKNOWN)
        self.assertEqual(detect_input_modality("image.png"), Modality.UNKNOWN)

    def test_dicom_inspector_does_not_expose_identifiers(self):
        filename, data = _series_files()[0]
        inspection = inspect_input(filename, data)
        self.assertFalse(inspection["patient_identifiers_in_response"])
        self.assertTrue(inspection["has_study_instance_uid"])
        self.assertNotIn("study_instance_uid", inspection)
        self.assertNotIn("series_instance_uid", inspection)
        self.assertNotIn("sop_instance_uid", inspection)


async def apply_ct_endpoint(uploads):
    """Call the async endpoint logic without requiring optional httpx2."""

    from app.main import analyze_ct_series

    return await analyze_ct_series(uploads)
