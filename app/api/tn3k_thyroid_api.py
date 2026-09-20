from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.imaging.ultrasound.tn3k_analysis_service import (
    TN3KAnalysisService,
)

from app.imaging.ultrasound.tn3k_visualization import (
    create_tn3k_overlay,
    save_binary_mask,
)


router = APIRouter()

_service: TN3KAnalysisService | None = None


def get_service() -> TN3KAnalysisService:
    global _service

    if _service is None:
        _service = TN3KAnalysisService()

    return _service


@router.post("/analyze/thyroid-ultrasound")
async def analyze_thyroid_ultrasound(
    file: UploadFile = File(...),
):
    filename = file.filename or ""

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload a JPG, JPEG or PNG "
                "thyroid ultrasound image."
            ),
        )

    try:
        with tempfile.TemporaryDirectory(
            prefix="oncoaegis_tn3k_"
        ) as temp_dir:

            temp_root = Path(temp_dir)

            suffix = Path(
                filename
            ).suffix.lower()

            input_path = (
                temp_root
                / f"input{suffix}"
            )

            input_path.write_bytes(
                await file.read()
            )

            service = get_service()

            result = service.analyze(
                input_path
            )

            mask = result[
                "segmentation"
            ].pop("mask")

            output_root = (
                Path("outputs")
                / "tn3k_api"
                / uuid.uuid4().hex
            )

            output_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            mask_path = (
                output_root
                / "mask.png"
            )

            overlay_path = (
                output_root
                / "overlay.png"
            )

            save_binary_mask(
                mask,
                mask_path,
            )

            create_tn3k_overlay(
                input_path,
                mask,
                overlay_path,
            )

            result["input"][
                "uploaded_filename"
            ] = filename

            result["outputs"] = {
                "mask_path": str(
                    mask_path
                ),

                "overlay_path": str(
                    overlay_path
                ),
            }

            return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "TN3K thyroid analysis failed: "
                f"{exc}"
            ),
        ) from exc