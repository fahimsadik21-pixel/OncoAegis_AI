from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image

from app.core.specialist_input import (
    SpecialistInputError,
    UploadedSpecialistFile,
    normalize_specialist_input,
    parse_spacing_mm,
)


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.fromarray(
        np.zeros((8, 8, 3), dtype=np.uint8)
    ).save(buffer, format="PNG")
    return buffer.getvalue()


def _nifti_bytes(shape=(2, 2, 2, 4)) -> bytes:
    image = nib.Nifti1Image(
        np.zeros(shape, dtype=np.float32),
        affine=np.eye(4),
    )
    image.header.set_zooms((1.5, 1.25, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "volume.nii.gz"
        nib.save(image, str(path))
        return path.read_bytes()


class TestSpecialistInput(unittest.TestCase):
    def test_ct_numpy_input_has_safe_metadata(self):
        buffer = BytesIO()
        np.save(buffer, np.zeros((3, 4, 5), dtype=np.float32))
        with tempfile.TemporaryDirectory() as workspace:
            normalized = normalize_specialist_input(
                model_id="ircadb01_liver_tumor_segmentation",
                files=[UploadedSpecialistFile("volume.npy", buffer.getvalue())],
                workspace=workspace,
                modality="CT",
                organ="liver",
                spacing_mm=(1.2, 0.8, 0.8),
            )
        self.assertEqual(normalized.data.shape, (3, 4, 5))
        self.assertEqual(normalized.metadata["input_format"], "NumPy")
        self.assertNotIn("filename", normalized.metadata)
        self.assertEqual(normalized.metadata["spacing_mm"], (1.2, 0.8, 0.8))

    def test_raster_inputs_use_model_specific_contracts(self):
        image = UploadedSpecialistFile("thyroid.png", _png_bytes())
        with tempfile.TemporaryDirectory() as workspace:
            tn3k = normalize_specialist_input(
                model_id="tn3k_thyroid_nodule_segmentation",
                files=[image],
                workspace=workspace,
            )
            busi = normalize_specialist_input(
                model_id="busi_breast_classifier",
                files=[UploadedSpecialistFile("breast.png", image.content)],
                workspace=workspace,
            )
            self.assertIsInstance(tn3k.data, Path)
            self.assertTrue(tn3k.data.is_file())
            self.assertEqual(busi.data, image.content)

    def test_flowcap_requires_and_materializes_eight_tubes(self):
        csv_content = (
            b'"FS Lin","SS Log","FL1 Log","FL2 Log","FL3 Log",'
            b'"FL4 Log","FL5 Log"\n1,2,3,4,5,6,7\n'
        )
        files = [
            UploadedSpecialistFile(f"tube_{index}.csv", csv_content)
            for index in range(1, 9)
        ]
        with tempfile.TemporaryDirectory() as workspace:
            normalized = normalize_specialist_input(
                model_id="flowcap_aml_patient_classifier",
                files=files,
                workspace=workspace,
            )
            self.assertEqual(len(normalized.data), 8)
            self.assertEqual(normalized.metadata["tube_count"], 8)
            self.assertEqual(normalized.metadata["event_counts"], [1] * 8)
            self.assertTrue(all(path.is_file() for path in normalized.data))

    def test_mri_nifti_uses_header_spacing_and_shape_contract(self):
        with tempfile.TemporaryDirectory() as workspace:
            normalized = normalize_specialist_input(
                model_id="msd_brain_tumor_segmentation",
                files=[UploadedSpecialistFile("brain.nii.gz", _nifti_bytes())],
                workspace=workspace,
                modality="MRI",
                organ="brain",
            )
        self.assertEqual(normalized.data.shape, (2, 2, 2, 4))
        self.assertEqual(normalized.metadata["spacing_mm"], (1.5, 1.25, 1.0))
        self.assertEqual(normalized.metadata["spacing_source"], "NIfTI header")
        self.assertNotIn("filename", normalized.metadata)

    def test_ct_and_mri_reject_single_image_exports(self):
        image = UploadedSpecialistFile("scan.jpg", _png_bytes())
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(
                SpecialistInputError,
                "3D CT data",
            ):
                normalize_specialist_input(
                    model_id="luna16_lung_segmentation",
                    files=[image],
                    workspace=workspace,
                )
            with self.assertRaisesRegex(
                SpecialistInputError,
                "4-channel 3D brain MRI volume",
            ):
                normalize_specialist_input(
                    model_id="msd_brain_tumor_segmentation",
                    files=[image],
                    workspace=workspace,
                )

    def test_two_dimensional_models_reject_volume_files(self):
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(
                SpecialistInputError,
                "single grayscale thyroid ultrasound image",
            ):
                normalize_specialist_input(
                    model_id="tn3k_thyroid_nodule_segmentation",
                    files=[
                        UploadedSpecialistFile(
                            "thyroid.nii.gz",
                            _nifti_bytes(),
                        )
                    ],
                    workspace=workspace,
                )

    def test_invalid_volume_and_flow_schema_are_rejected(self):
        bad_volume = BytesIO()
        np.save(bad_volume, np.full((2, 2, 2), np.nan, dtype=np.float32))
        with self.assertRaises(SpecialistInputError):
            normalize_specialist_input(
                model_id="ircadb01_liver_tumor_segmentation",
                files=[UploadedSpecialistFile("volume.npy", bad_volume.getvalue())],
                workspace=tempfile.gettempdir(),
            )

        bad_csv = b"FS Lin,SS Log\n1,2\n"
        with self.assertRaises(SpecialistInputError):
            normalize_specialist_input(
                model_id="flowcap_aml_patient_classifier",
                files=[
                    UploadedSpecialistFile(f"tube_{index}.csv", bad_csv)
                    for index in range(1, 9)
                ],
                workspace=tempfile.gettempdir(),
            )

    def test_empty_upload_is_rejected_before_model_loading(self):
        with self.assertRaises(SpecialistInputError):
            normalize_specialist_input(
                model_id="busi_breast_classifier",
                files=[UploadedSpecialistFile("empty.png", b"")],
                workspace=tempfile.gettempdir(),
            )

    def test_metadata_and_spacing_validation_is_conservative(self):
        with self.assertRaises(SpecialistInputError):
            normalize_specialist_input(
                model_id="ircadb01_liver_tumor_segmentation",
                files=[UploadedSpecialistFile("volume.npy", b"bad")],
                workspace=tempfile.gettempdir(),
                modality="MRI",
            )
        self.assertEqual(parse_spacing_mm("1, 2, 3"), (1.0, 2.0, 3.0))
        with self.assertRaises(SpecialistInputError):
            parse_spacing_mm("1,2")


if __name__ == "__main__":
    unittest.main()
