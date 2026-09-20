from pathlib import Path

from app.schemas.results import Modality


DICOM_EXTENSIONS = {".dcm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt"}


def detect_input_modality(filename: str) -> Modality:

    suffix = Path(filename).suffix.lower()

    if suffix in DICOM_EXTENSIONS:
        # A .dcm filename alone cannot distinguish CT, MRI, ultrasound, etc.
        # DICOM metadata inspection must make that decision.
        return Modality.UNKNOWN

    if suffix in IMAGE_EXTENSIONS:
        # A raster image may be ultrasound, pathology, screenshot, or another
        # image type. A vision router must make that decision later.
        return Modality.UNKNOWN

    if suffix == ".pdf":
        return Modality.DOCUMENT

    if suffix == ".txt":
        return Modality.TEXT

    return Modality.UNKNOWN
