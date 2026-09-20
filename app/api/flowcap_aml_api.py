from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.imaging.hematology.flowcap_official_service import (
    FlowCAPError,
    analyze_flowcap_patient,
)


router = APIRouter()


@router.post("/analyze/aml-flow")
async def analyze_aml_flow(
    tube1: UploadFile = File(...),
    tube2: UploadFile = File(...),
    tube3: UploadFile = File(...),
    tube4: UploadFile = File(...),
    tube5: UploadFile = File(...),
    tube6: UploadFile = File(...),
    tube7: UploadFile = File(...),
    tube8: UploadFile = File(...),
):
    uploaded_files = [
        tube1,
        tube2,
        tube3,
        tube4,
        tube5,
        tube6,
        tube7,
        tube8,
    ]

    for uploaded_file in uploaded_files:
        filename = uploaded_file.filename or ""

        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail=f"{filename} is not a CSV file.",
            )

    try:
        with tempfile.TemporaryDirectory(
            prefix="oncoaegis_upload_"
        ) as temp_directory:

            temp_root = Path(temp_directory)
            saved_paths = []

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                destination = temp_root / f"tube{index}.CSV"

                content = await uploaded_file.read()
                destination.write_bytes(content)

                saved_paths.append(destination)

            result = analyze_flowcap_patient(saved_paths)

            result["uploaded_files"] = [
                uploaded_file.filename
                for uploaded_file in uploaded_files
            ]

            return result

    except FlowCAPError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"FlowCAP AML analysis failed: {exc}",
        ) from exc