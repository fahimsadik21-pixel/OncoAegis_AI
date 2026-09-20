from io import BytesIO
from pathlib import Path

import fitz
import pydicom
from PIL import Image

from app.schemas.results import Modality


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}

TEXT_EXTENSIONS = {
    ".txt",
}

DICOM_EXTENSIONS = {
    ".dcm",
}


def inspect_dicom(data: bytes) -> dict:
    """
    Inspect a DICOM file using metadata only.
    Pixel data is not loaded here.
    """

    ds = pydicom.dcmread(
        BytesIO(data),
        stop_before_pixels=True,
        force=True,
    )

    dicom_modality = str(
        getattr(ds, "Modality", "")
    ).strip().upper()

    modality_map = {
        "CT": Modality.CT,
        "MR": Modality.MRI,
        "US": Modality.ULTRASOUND,
    }

    detected_modality = modality_map.get(
        dicom_modality,
        Modality.UNKNOWN,
    )

    body_part = str(
        getattr(ds, "BodyPartExamined", "")
    ).strip()

    study_description = str(
        getattr(ds, "StudyDescription", "")
    ).strip()

    series_description = str(
        getattr(ds, "SeriesDescription", "")
    ).strip()

    manufacturer = str(
        getattr(ds, "Manufacturer", "")
    ).strip()

    return {
        "input_type": "dicom",
        "detected_modality": detected_modality,
        "dicom_modality": dicom_modality or None,
        "body_part": body_part or None,
        "study_description": study_description or None,
        "series_description": series_description or None,
        "manufacturer": manufacturer or None,
        # UIDs are useful internally for series validation, but exposing them
        # from the universal inspector would unnecessarily leak identifiers.
        "has_study_instance_uid": bool(getattr(ds, "StudyInstanceUID", None)),
        "has_series_instance_uid": bool(getattr(ds, "SeriesInstanceUID", None)),
        "has_sop_instance_uid": bool(getattr(ds, "SOPInstanceUID", None)),
        "patient_identifiers_in_response": False,
    }


def inspect_pdf(data: bytes) -> dict:
    """
    Inspect a PDF and estimate whether OCR is required.
    """

    document = fitz.open(
        stream=data,
        filetype="pdf",
    )

    text_parts = []

    for page in document:
        page_text = page.get_text()
        text_parts.append(page_text)

    text = "\n".join(text_parts).strip()

    page_count = document.page_count
    text_characters = len(text)

    average_text_characters_per_page = (
        text_characters / page_count
        if page_count > 0
        else 0
    )

    # Very low extracted text usually means
    # the PDF is mostly scanned/image-based.
    needs_ocr = (
        page_count > 0
        and average_text_characters_per_page < 80
    )

    document.close()

    return {
        "input_type": "pdf_document",
        "detected_modality": Modality.DOCUMENT,
        "page_count": page_count,
        "text_characters": text_characters,
        "average_text_characters_per_page": round(
            average_text_characters_per_page,
            2,
        ),
        "needs_ocr": needs_ocr,
    }


def inspect_text(data: bytes) -> dict:
    """
    Inspect plain text input.
    """

    text = data.decode(
        "utf-8",
        errors="ignore",
    ).strip()

    return {
        "input_type": "clinical_text",
        "detected_modality": Modality.TEXT,
        "text_characters": len(text),
        "is_empty": len(text) == 0,
    }


def inspect_image(data: bytes) -> dict:
    """
    Inspect a standard image file.

    Important:
    We do NOT assume every JPG/PNG is ultrasound.
    A future vision router will classify the image modality.
    """

    image = Image.open(BytesIO(data))
    image.load()

    return {
        "input_type": "medical_image_candidate",
        "detected_modality": Modality.UNKNOWN,
        "image_format": image.format,
        "width": image.width,
        "height": image.height,
        "image_mode": image.mode,
        "requires_vision_routing": True,
    }


def inspect_nifti(filename: str) -> dict:
    """
    Basic NIfTI recognition.
    Actual volume loading will be added later.
    """

    return {
        "input_type": "nifti_volume",
        "detected_modality": Modality.UNKNOWN,
        "filename": filename,
        "requires_medical_volume_routing": True,
    }


def inspect_input(
    filename: str,
    data: bytes,
) -> dict:
    """
    Main universal input inspector.

    Supported:
    - DICOM
    - PDF
    - TXT
    - PNG/JPG/BMP/WEBP
    - NIfTI
    """

    lower_name = filename.lower()
    suffix = Path(filename).suffix.lower()

    try:
        if suffix in DICOM_EXTENSIONS:
            return inspect_dicom(data)

        if suffix == ".pdf":
            return inspect_pdf(data)

        if suffix in TEXT_EXTENSIONS:
            return inspect_text(data)

        if suffix in IMAGE_EXTENSIONS:
            return inspect_image(data)

        if (
            lower_name.endswith(".nii.gz")
            or suffix == ".nii"
        ):
            return inspect_nifti(filename)

        return {
            "input_type": "unsupported",
            "detected_modality": Modality.UNKNOWN,
            "filename": filename,
            "error": "Unsupported file type",
        }

    except pydicom.errors.InvalidDicomError as exc:
        return {
            "input_type": "invalid_dicom",
            "detected_modality": Modality.UNKNOWN,
            "filename": filename,
            "error": str(exc),
        }

    except fitz.FileDataError as exc:
        return {
            "input_type": "invalid_pdf",
            "detected_modality": Modality.UNKNOWN,
            "filename": filename,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "input_type": "invalid_or_corrupted",
            "detected_modality": Modality.UNKNOWN,
            "filename": filename,
            "error": str(exc),
        }
