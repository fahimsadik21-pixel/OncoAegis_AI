from app.imaging.mri.mask_export import save_mask_as_nifti
from app.imaging.mri.visualization import create_mri_overlay
from fastapi import FastAPI, File, HTTPException, UploadFile
from typing import Annotated

from app.core.capability_registry import get_capabilities
from app.core.input_inspector import inspect_input
from app.core.model_router import route_specialist_model
from app.clinical.document_intelligence import (
    ClinicalDocumentError,
    ClinicalEvidence,
    analyze_clinical_document,
)
from app.clinical.fusion import ImageEvidence, fuse_evidence
from app.imaging.ct.pipeline import run_ct_pipeline_from_bytes
from app.imaging.ct.luna_inference import (
    CTInferenceError,
    LUNA16ModelService,
    LUNAModelUnavailable,
    run_luna_ct_segmentation,
)

from PIL import Image
from io import BytesIO

from app.imaging.dermatology.skin_analysis_service import (
    SkinAnalysisError,
    SkinAnalysisService,
)

from app.imaging.dermatology.skin_visualization import (
    SkinVisualizationError,
    save_skin_overlay,
)

from app.imaging.dermatology.skin_output_export import (
    SkinOutputExportError,
    export_skin_result,
)

from app.imaging.ct.colon_analysis_service import (
    ColonAnalysisError,
    ColonAnalysisService,
)

from app.imaging.ct.colon_measurement import (
    ColonMeasurementError,
    measure_colon_segmentation,
)

from app.imaging.ct.colon_visualization import (
    ColonVisualizationError,
    save_colon_overlay,
)

from app.imaging.ct.colon_output_export import (
    ColonOutputExportError,
    export_colon_result,
)

from app.imaging.ct.pancreas_analysis_service import (
    PancreasAnalysisError,
    PancreasAnalysisService,
)

from app.imaging.ct.pancreas_measurement import (
    PancreasMeasurementError,
    measure_pancreas_segmentation,
)

from app.imaging.ct.pancreas_visualization import (
    PancreasVisualizationError,
    save_pancreas_overlay,
)

from app.imaging.ct.pancreas_output_export import (
    PancreasOutputExportError,
    export_pancreas_result,
)

from pathlib import Path
import uuid

from app.imaging.ct.liver_dicom_io import (
    LiverDicomIOError,
    load_liver_ct_dicom_series,
)

from app.imaging.ct.liver_analysis_service import (
    LiverAnalysisError,
    LiverAnalysisService,
)

from app.imaging.ct.liver_visualization import (
    LiverVisualizationError,
    save_liver_overlay,
)

from app.imaging.ct.liver_output_export import (
    LiverOutputExportError,
    export_liver_ai_result,
)
from app.imaging.dicom.loader import DicomSeriesError
from app.imaging.ct.preprocessing import CTPreprocessingError
from app.imaging.ultrasound.busi_inference import (
    BUSIInferenceError,
    BUSIModelService,
    BUSIModelUnavailable,
    run_busi_ultrasound,
)
from app.imaging.mri.brain_tumor_inference import (
    BrainTumorModelService,
    BrainTumorModelUnavailable,
    BrainTumorInferenceError,
)

import torch
import os
import tempfile
import nibabel as nib
import numpy as np

from app.imaging.mri.preprocessing import (
    preprocess_mri_volume,
    load_nifti_volume,
)
from app.schemas.imaging import (
    CTSeriesAnalysisResponse,
    CTSpecialistAnalysisResponse,
    UltrasoundAnalysisResponse,
)
from app.schemas.clinical import (
    DocumentAnalysisResponse,
    FusionAnalysisResponse,
    FusionRequest,
    MultimodalCTResponse,
    MultimodalUltrasoundResponse,
)
from app.schemas.routing import ModelRouteRequest, ModelRouteResponse
from app.schemas.results import CancerAnalysisResult, Evidence, Finding, Modality, Uncertainty
from app.registry.cancer_registry import get_cancer_capabilities
from app.registry.model_registry import get_model_registry
from src.data.registry import get_dataset_registry


APP_VERSION = "0.3.0"


app = FastAPI(
    title="OncoAegis AI",
    description="Multimodal Cancer Detection and Analysis Assistant",
    version=APP_VERSION,
)


_busi_service: BUSIModelService | None = None
_luna_service: LUNA16ModelService | None = None
_brain_service: BrainTumorModelService | None = None
_liver_service: LiverAnalysisService | None = None
_pancreas_service: PancreasAnalysisService | None = None
_colon_service: ColonAnalysisService | None = None
_skin_service: SkinAnalysisService | None = None


def get_skin_service() -> SkinAnalysisService:
    global _skin_service

    if _skin_service is None:
        _skin_service = SkinAnalysisService()

    return _skin_service


def get_colon_service() -> ColonAnalysisService:
    global _colon_service

    if _colon_service is None:
        _colon_service = ColonAnalysisService()

    return _colon_service


def get_busi_service() -> BUSIModelService:
    global _busi_service
    if _busi_service is None:
        _busi_service = BUSIModelService()
    return _busi_service


def get_luna_service() -> LUNA16ModelService:
    global _luna_service
    if _luna_service is None:
        _luna_service = LUNA16ModelService()
    return _luna_service


def get_brain_service() -> BrainTumorModelService:
    global _brain_service

    if _brain_service is None:
        _brain_service = BrainTumorModelService()

    return _brain_service


def get_liver_service() -> LiverAnalysisService:
    global _liver_service

    if _liver_service is None:
        _liver_service = LiverAnalysisService()

    return _liver_service


def get_pancreas_service() -> PancreasAnalysisService:
    global _pancreas_service

    if _pancreas_service is None:
        _pancreas_service = PancreasAnalysisService()

    return _pancreas_service


@app.get("/")
def root():
    return {
        "name": "OncoAegis AI",
        "version": APP_VERSION,
        "status": "online",
        "purpose": "Multimodal cancer detection and analysis"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": APP_VERSION,
    }


@app.get("/system/status")
def system_status():
    """Return a compact readiness view for the local research platform."""

    registered_models = get_model_registry().list()
    model_status = {
        spec.model_id: (
            "available" if spec.checkpoint_available else "checkpoint_missing"
        )
        for spec in registered_models
    }
    required_baselines_ready = all(
        model_status.get(model_id) == "available"
        for model_id in (
            "luna16_lung_segmentation",
            "busi_breast_segmentation",
            "busi_breast_classifier",
        )
    )
    return {
        "name": "OncoAegis AI",
        "version": APP_VERSION,
        "status": "ready_for_registered_baselines" if required_baselines_ready else "degraded",
        "components": {
            "input_inspector": "ready",
            "dicom_series_engine": "ready",
            "ct_preprocessing": "ready",
            "specialist_router": "ready",
            "clinical_document_evidence": "rule_based_foundation",
            "multimodal_fusion": "ready",
            "expert_review_gate": "required",
        },
        "models": model_status,
        "datasets": {
            entry.name: {
                "download_status": entry.download_status,
                "preprocessing_status": entry.preprocessing_status,
            }
            for entry in get_dataset_registry().list()
        },
        "safety": {
            "imaging_alone_confirms_malignancy": False,
            "pathology_required_for_confirmed_malignancy": True,
            "clinical_validation": False,
        },
    }


@app.get("/capabilities")
def capabilities():
    return get_capabilities()


@app.get("/datasets")
def datasets():
    """Return the non-sensitive dataset registry used by the model layer."""

    return {
        "datasets": [
            entry.__dict__
            for entry in get_dataset_registry().list()
        ]
    }


@app.get("/models")
def models():
    """Return registered specialist models and runtime availability."""

    return {
        "models": [
            spec.to_dict()
            for spec in get_model_registry().list()
        ]
    }


@app.get("/cancer-capabilities")
def cancer_capabilities():
    """Return the high-level cancer-to-specialist mapping."""

    return {"capabilities": get_cancer_capabilities()}


@app.post(
    "/route/model",
    response_model=ModelRouteResponse,
)
def route_model(request: ModelRouteRequest):
    """Select a registered specialist model without running inference."""

    decision = route_specialist_model(
        modality=request.modality,
        organ=request.organ,
        task=request.task,
        cancer_type=request.cancer_type,
        require_available=request.require_available,
    )
    return decision.to_dict()


def _ct_inspection(pipeline) -> dict[str, object]:
    """Build one privacy-safe inspection payload for CT endpoints."""

    series = pipeline.series
    metadata = series.safe_metadata
    preprocessed = pipeline.preprocessed
    anatomy_route = pipeline.anatomy_route
    return {
        "input_type": "dicom_ct_series",
        "detected_modality": series.modality,
        "series_slices": len(series.slices),
        "volume_shape": series.shape,
        "body_region": metadata.get("body_region"),
        "pixel_spacing_mm": series.pixel_spacing,
        "slice_spacing_mm": series.slice_spacing,
        "slice_thickness_mm": series.slice_thickness,
        "ordering_method": series.ordering_method,
        "volume_ready": True,
        "preprocessing_applied": True,
        "preprocessed_volume_shape": preprocessed.output_shape,
        "preprocessed_spacing_mm": preprocessed.output_spacing_mm,
        "hu_window": preprocessed.hu_window,
        "clipped_fraction": preprocessed.clipped_fraction,
        "candidate_organs": anatomy_route.candidate_organs,
        "anatomy_route_status": anatomy_route.route_status,
        "patient_identifiers_in_response": False,
        "warnings": series.warnings,
    }


async def _read_ct_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    if not files:
        raise HTTPException(
            status_code=422,
            detail="At least one DICOM slice is required",
        )
    uploaded_files: list[tuple[str, bytes]] = []
    for upload in files:
        filename = upload.filename or "uploaded_dicom"
        content = await upload.read()
        if content:
            uploaded_files.append((filename, content))
    if not uploaded_files:
        raise HTTPException(
            status_code=422,
            detail="All uploaded DICOM files were empty",
        )
    return uploaded_files


def _luna_analysis(pipeline, prediction) -> CancerAnalysisResult:
    """Represent LUNA output as anatomy evidence with explicit abstention."""

    metadata = pipeline.series.safe_metadata
    return CancerAnalysisResult(
        modality=Modality.CT,
        body_region=metadata.get("body_region"),
        suspected_organ="lung",
        findings=[
            Finding(
                organ="lung",
                finding_type="lung_anatomy_segmentation",
                volume_mm3=prediction.segmented_volume_mm3,
                segmentation_available=True,
            )
        ],
        evidence=[
            Evidence(
                source="luna16_lung_segmentation",
                description=(
                    "LUNA16 research output segments lung anatomy; it does not "
                    "detect or confirm malignancy."
                ),
            )
        ],
        uncertainty=Uncertainty(
            abstained=True,
            reason=(
                "No validated lung lesion/malignancy conclusion is available; "
                "expert review is required."
            ),
        ),
        final_status="lung_anatomy_segmented_requires_lesion_review",
        model_used=prediction.model_name,
        model_version=prediction.model_version,
    )


@app.post("/analyze/input")
async def analyze_input(
    file: UploadFile = File(...)
):
    content = await file.read()
    filename = file.filename or "uploaded_input"

    inspection = inspect_input(
        filename=filename,
        data=content
    )

    return {
        "filename": filename,
        "file_size_bytes": len(content),
        "inspection": inspection,
        "status": "input_inspected",
        "next_stage": "medical_router"
    }


@app.post(
    "/analyze/document",
    response_model=DocumentAnalysisResponse,
)
async def analyze_document(
    file: UploadFile = File(..., description="One PDF or TXT clinical document"),
):
    """Extract structured, non-diagnostic evidence from a clinical document."""

    content = await file.read()
    try:
        analysis = analyze_clinical_document(
            file.filename or "uploaded_document",
            content,
        )
    except ClinicalDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **analysis.to_dict(),
        "next_stage": "evidence_fusion",
    }


@app.post(
    "/analyze/fusion",
    response_model=FusionAnalysisResponse,
)
def analyze_fusion(request: FusionRequest):
    """Combine specialist imaging evidence with extracted document evidence."""

    result = fuse_evidence(
        image_evidence=(
            ImageEvidence(
                modality=item.modality,
                finding=item.finding,
                organ=item.organ,
                risk_level=item.risk_level,
                source=item.source,
                evidence_kind=item.evidence_kind,
            )
            for item in request.image_evidence
        ),
        document_evidence=(
            ClinicalEvidence(
                source_type=item.source_type,
                evidence_type=item.evidence_type,
                statement=item.statement,
                polarity=item.polarity,
                confirmation_status=item.confirmation_status,
                organ=item.organ,
                matched_terms=tuple(item.matched_terms),
            )
            for item in request.document_evidence
        ),
    )
    return result.to_dict()


@app.post(
    "/analyze/ct-series",
    response_model=CTSeriesAnalysisResponse,
)
async def analyze_ct_series(
    files: list[UploadFile] = File(
        ...,
        description="All DICOM image slices belonging to one CT series",
    ),
):
    """Validate, preprocess and route uploaded CT slices.

    The volume is intentionally not returned through the API. It remains an
    internal object for the next anatomy/model stage, while the API exposes
    only a safe technical summary.
    """

    uploaded_files = await _read_ct_uploads(files)

    try:
        pipeline = run_ct_pipeline_from_bytes(uploaded_files)
    except (DicomSeriesError, CTPreprocessingError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    inspection = _ct_inspection(pipeline)

    return {
        "filenames_received": len(files),
        "inspection": inspection,
        "safe_metadata": pipeline.series.safe_metadata,
        "status": "ct_volume_preprocessed",
        "next_stage": "anatomy_model_router",
    }


def _run_luna_ct_specialist(
    uploaded_files: list[tuple[str, bytes]],
):
    """Prepare a chest CT with the LUNA training window and run its router."""

    # LUNA16 was normalized from [-1000, 400] HU.  Keep that model-specific
    # profile separate from the generic [-160, 240] technical CT window.
    pipeline = run_ct_pipeline_from_bytes(
        uploaded_files,
        window_center=-300.0,
        window_width=1400.0,
    )
    body_region = pipeline.series.safe_metadata.get("body_region")
    if body_region != "CHEST":
        raise CTInferenceError(
            "The LUNA16 specialist requires a CT series identified as CHEST; "
            "use the technical CT endpoint when body region is unknown or different."
        )

    route = route_specialist_model(
        modality="CT",
        organ="lung",
        task="segmentation",
        cancer_type="lung_cancer",
        require_available=True,
    )
    if route.selected_model_id != "luna16_lung_segmentation":
        raise CTInferenceError(
            f"LUNA16 specialist route unavailable: {route.route_status}. {route.reason}"
        )

    prediction = run_luna_ct_segmentation(
        pipeline.preprocessed.volume,
        spacing_mm=pipeline.preprocessed.output_spacing_mm,
        service=get_luna_service(),
    )
    return pipeline, prediction


@app.post(
    "/analyze/ct-specialist",
    response_model=CTSpecialistAnalysisResponse,
)
async def analyze_ct_specialist(
    files: list[UploadFile] = File(
        ...,
        description="All DICOM image slices belonging to one chest CT series",
    ),
):
    """Run the registered LUNA16 lung-anatomy specialist on one CT series."""

    uploaded_files = await _read_ct_uploads(files)
    try:
        pipeline, prediction = _run_luna_ct_specialist(uploaded_files)
    except LUNAModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (DicomSeriesError, CTPreprocessingError, CTInferenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "filenames_received": len(files),
        "inspection": _ct_inspection(pipeline),
        "prediction": prediction.to_dict(),
        "analysis": _luna_analysis(pipeline, prediction).model_dump(mode="json"),
        "status": "ct_specialist_prediction",
        "next_stage": "expert_review",
    }


@app.post(
    "/analyze/multimodal/ct-series",
    response_model=MultimodalCTResponse,
)
async def analyze_multimodal_ct_series(
    files: list[UploadFile] = File(
        ...,
        description="All DICOM image slices belonging to one chest CT series",
    ),
    document: UploadFile | None = File(
        None,
        description="Optional related PDF or TXT clinical report",
    ),
):
    """Run LUNA16 CT segmentation and fuse it with an optional report."""

    uploaded_files = await _read_ct_uploads(files)
    try:
        pipeline, prediction = _run_luna_ct_specialist(uploaded_files)
    except LUNAModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (DicomSeriesError, CTPreprocessingError, CTInferenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    document_result = None
    document_evidence: tuple[ClinicalEvidence, ...] = ()
    if document is not None:
        document_content = await document.read()
        try:
            document_result = analyze_clinical_document(
                document.filename or "uploaded_document",
                document_content,
            )
        except ClinicalDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        document_evidence = document_result.evidence

    fused = fuse_evidence(
        image_evidence=(
            ImageEvidence(
                modality="CT",
                organ="lung",
                finding=(
                    "LUNA16 lung anatomy segmentation completed; no nodule or "
                    "malignancy conclusion was produced."
                ),
                risk_level="technical",
                source="luna16_lung_segmentation",
                evidence_kind="technical",
            ),
        ),
        document_evidence=document_evidence,
    )
    return {
        "filenames_received": len(files),
        "inspection": _ct_inspection(pipeline),
        "prediction": prediction.to_dict(),
        "document": document_result.to_dict() if document_result else None,
        "fusion": fused.to_dict(),
        "status": "multimodal_analysis_complete",
        "next_stage": "expert_review",
    }


@app.post(
    "/analyze/ultrasound",
    response_model=UltrasoundAnalysisResponse,
)
async def analyze_ultrasound(
    file: UploadFile = File(..., description="One breast-ultrasound raster image"),
):
    """Run the BUSI research baseline and return a safe structured summary."""

    content = await file.read()
    try:
        prediction = run_busi_ultrasound(content, service=get_busi_service())
    except BUSIModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BUSIInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    analysis = CancerAnalysisResult(
        modality=Modality.ULTRASOUND,
        body_region="BREAST",
        suspected_organ="breast",
        findings=[
            Finding(
                organ="breast",
                finding_type="busi_model_prediction",
                classification=prediction.predicted_class,
                segmentation_available=True,
            )
        ],
        evidence=[
            Evidence(
                source=prediction.model_name,
                description=(
                    "Research-model output from the BUSI breast-ultrasound dataset; "
                    "not diagnostic evidence."
                ),
            )
        ],
        uncertainty=Uncertainty(
            abstained=True,
            reason="Model confidence is not clinically calibrated; expert review is required.",
        ),
        final_status="research_model_prediction_requires_expert_review",
        model_used=prediction.model_name,
        model_version=prediction.model_version,
    )
    return {
        "filename": file.filename or "uploaded_ultrasound",
        "inspection": {
            "original_image_shape": prediction.original_image_shape,
            "processed_image_shape": prediction.processed_image_shape,
            "lesion_area_fraction": prediction.lesion_area_fraction,
            "warnings": list(prediction.warnings),
        },
        "prediction": {
            "predicted_class": prediction.predicted_class,
            "class_probabilities": prediction.class_probabilities,
            "model_confidence": prediction.model_confidence,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
        },
        "analysis": analysis.model_dump(),
    }


@app.post(
    "/analyze/multimodal/ultrasound",
    response_model=MultimodalUltrasoundResponse,
)
async def analyze_multimodal_ultrasound(
    image: UploadFile = File(..., description="One breast-ultrasound image"),
    document: UploadFile | None = File(
        None,
        description="Optional related PDF or TXT clinical report",
    ),
):
    """Run BUSI inference and optionally fuse it with one clinical document."""

    image_content = await image.read()
    try:
        prediction = run_busi_ultrasound(
            image_content,
            service=get_busi_service(),
        )
    except BUSIModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BUSIInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    document_result = None
    document_evidence: tuple[ClinicalEvidence, ...] = ()
    if document is not None:
        document_content = await document.read()
        try:
            document_result = analyze_clinical_document(
                document.filename or "uploaded_document",
                document_content,
            )
        except ClinicalDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        document_evidence = document_result.evidence

    image_risk = (
        "no_suspicious_finding"
        if prediction.predicted_class == "normal"
        else "indeterminate"
    )
    fused = fuse_evidence(
        image_evidence=(
            ImageEvidence(
                modality="ULTRASOUND",
                organ="breast",
                finding=(
                    f"BUSI research model class: {prediction.predicted_class}"
                ),
                risk_level=image_risk,
                source=prediction.model_name,
            ),
        ),
        document_evidence=document_evidence,
    )
    return {
        "image_filename": image.filename or "uploaded_ultrasound",
        "prediction": {
            "predicted_class": prediction.predicted_class,
            "class_probabilities": prediction.class_probabilities,
            "model_confidence": prediction.model_confidence,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
        },
        "document": document_result.to_dict() if document_result else None,
        "fusion": fused.to_dict(),
    }

@app.post("/analyze/mri-brain")
async def analyze_mri_brain(
    file: UploadFile = File(
        ...,
        description="One MRI NIfTI (.nii/.nii.gz) brain volume"
    ),
):
    """
    Run MSD Brain Tumor MRI segmentation model.

    Research use only.
    Does not confirm malignancy.
    """

    content = await file.read()
    filename = file.filename or "uploaded_brain_mri.nii.gz"

    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=422,
            detail="Only NIfTI MRI files (.nii/.nii.gz) are supported",
        )

    temp_path = None

    try:
        # Save uploaded MRI temporarily
        suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = temp.name

        # Load MRI
        image = nib.load(temp_path)
        volume = image.get_fdata()

        # Voxel spacing
        spacing = tuple(float(x) for x in image.header.get_zooms()[:3])

        # Model prediction
        prediction = get_brain_service().predict(volume)

        # Base name for output files
        base_name = filename.replace(".nii.gz", "").replace(".nii", "")

        # Save segmentation mask
        mask_output = save_mask_as_nifti(
            prediction.segmentation_mask,
            temp_path,
            f"outputs/mri_masks/{base_name}_tumor_mask.nii.gz",
        )

        # Save visualization overlay
        overlay_output = create_mri_overlay(
            temp_path,
            mask_output,
            f"outputs/mri_visualization/{base_name}_overlay.png",
        )

        # Tumor measurements
        tumor_voxels = int(prediction.tumor_volume_voxels)

        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        tumor_volume_mm3 = float(tumor_voxels * voxel_volume_mm3)
        tumor_volume_cm3 = float(tumor_volume_mm3 / 1000.0)

    except BrainTumorModelUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except BrainTumorInferenceError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"MRI processing failed: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "filename": filename,
        "modality": "MRI",
        "organ": "brain",
        "dataset": "MSD_Brain_Tumor",
        "voxel_spacing_mm": {
            "x": float(spacing[0]),
            "y": float(spacing[1]),
            "z": float(spacing[2]),
        },
        "prediction": prediction.to_dict(),
        "tumor_measurement": {
            "tumor_volume_voxels": tumor_voxels,
            "tumor_volume_mm3": round(tumor_volume_mm3, 2),
            "tumor_volume_cm3": round(tumor_volume_cm3, 3),
        },
        "segmentation_output": {
            "mask_file": mask_output,
        },
        "visualization": {
            "overlay_image": overlay_output,
        },
        "status": "research_model_prediction",
        "safety": {
            "clinical_diagnosis": False,
            "malignancy_confirmation": False,
            "expert_review_required": True,
            "pathology_required_for_confirmation": True,
        },
    }
@app.post("/analyze/liver-ct")
async def analyze_liver_ct(
    file: UploadFile = File(
        ...,
        description="Upload one ZIP file containing the liver/abdominal CT DICOM series",
    ),
):
    """
    Analyze one abdominal/liver CT DICOM series packed inside a ZIP file.

    Research use only.
    """

    import zipfile

    filename = file.filename or "uploaded_ct.zip"

    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=422,
            detail=(
                "Please upload the CT DICOM series as one ZIP file."
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=422,
            detail="Uploaded ZIP file is empty.",
        )

    case_id = (
        "liver_"
        + uuid.uuid4().hex[:12]
    )

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_directory = Path(temp_dir)

            zip_path = (
                temp_directory
                / "uploaded_ct.zip"
            )

            zip_path.write_bytes(content)

            extract_directory = (
                temp_directory
                / "dicom_series"
            )

            extract_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -----------------------------
            # Extract ZIP safely
            # -----------------------------

            with zipfile.ZipFile(
                zip_path,
                "r",
            ) as archive:

                for member in archive.infolist():

                    member_path = (
                        extract_directory
                        / member.filename
                    ).resolve()

                    if not str(member_path).startswith(
                        str(
                            extract_directory.resolve()
                        )
                    ):
                        raise LiverAnalysisError(
                            "Unsafe ZIP structure detected."
                        )

                archive.extractall(
                    extract_directory
                )

            # -----------------------------
            # DICOM -> CT volume
            # -----------------------------

            dicom_volume = (
                load_liver_ct_dicom_series(
                    extract_directory
                )
            )

            if dicom_volume.modality.upper() != "CT":

                raise LiverAnalysisError(
                    "The uploaded study is not a CT series."
                )

            # -----------------------------
            # Liver AI analysis
            # -----------------------------

            result = (
                get_liver_service()
                .analyze(
                    dicom_volume.volume,
                    dicom_volume.spacing_mm,
                )
            )

            # -----------------------------
            # Visualization
            # -----------------------------

            visualization_path = (
                Path("outputs")
                / "liver_visualization"
                / f"{case_id}_overlay.png"
            )

            visualization = (
                save_liver_overlay(
                    dicom_volume.volume,
                    result.segmentation_mask,
                    visualization_path,
                )
            )

            # -----------------------------
            # Export
            # -----------------------------

            exported = (
                export_liver_ai_result(
                    segmentation_mask=
                        result.segmentation_mask,

                    spacing_mm=
                        dicom_volume.spacing_mm,

                    case_id=
                        case_id,

                    prediction=
                        result.to_dict(),

                    measurement=
                        result.measurement,

                    dicom_metadata={
                        "modality":
                            dicom_volume.modality,

                        "slice_count":
                            dicom_volume.slice_count,

                        "rows":
                            dicom_volume.rows,

                        "columns":
                            dicom_volume.columns,

                        "spacing_mm":
                            dicom_volume.spacing_mm,

                        "study_instance_uid":
                            dicom_volume.study_instance_uid,

                        "series_instance_uid":
                            dicom_volume.series_instance_uid,
                    },
                )
            )

    except zipfile.BadZipFile as exc:

        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a valid ZIP archive.",
        ) from exc

    except LiverDicomIOError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"DICOM processing failed: {exc}",
        ) from exc

    except LiverAnalysisError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Liver analysis failed: {exc}",
        ) from exc

    except LiverVisualizationError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Visualization failed: {exc}",
        ) from exc

    except LiverOutputExportError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Output export failed: {exc}",
        ) from exc

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail="GPU memory was insufficient for this CT study.",
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Liver CT processing failed: {exc}",
        ) from exc

    return {

        "case_id":
            case_id,

        "uploaded_file":
            filename,

        "modality":
            "CT",

        "organ":
            "liver",

        "dataset":
            "IRCADB01",

        "input": {

            "type":
                "ZIP containing DICOM CT series",

            "slice_count":
                dicom_volume.slice_count,

            "volume_shape": [
                int(x)
                for x
                in dicom_volume.volume.shape
            ],

            "voxel_spacing_mm": {

                "z":
                    float(
                        dicom_volume.spacing_mm[0]
                    ),

                "y":
                    float(
                        dicom_volume.spacing_mm[1]
                    ),

                "x":
                    float(
                        dicom_volume.spacing_mm[2]
                    ),
            },
        },

        "prediction":
            result.to_dict(),

        "visualization": {

            "overlay_image":
                visualization[
                    "output_path"
                ],

            "selected_slice":
                visualization[
                    "slice_index"
                ],
        },

        "segmentation_output": {

            "mask_file":
                exported[
                    "mask_file"
                ],

            "metadata_file":
                exported[
                    "metadata_file"
                ],
        },

        "status":
            "research_model_prediction",

        "safety": {

            "clinical_diagnosis":
                False,

            "malignancy_confirmation":
                False,

            "expert_review_required":
                True,

            "model_clinically_validated":
                False,
        },
    }
@app.post("/analyze/pancreas-ct")
async def analyze_pancreas_ct(
    file: UploadFile = File(
        ...,
        description=(
            "Upload either one NIfTI CT volume (.nii/.nii.gz) "
            "or one ZIP containing a DICOM CT series"
        ),
    ),
):
    """
    Run the MSD Task07 two-stage pancreas CT research pipeline.

    Supported inputs:
        1. NIfTI CT volume (.nii / .nii.gz)
        2. ZIP containing one DICOM CT series

    Research use only.
    This endpoint does not confirm or exclude pancreatic cancer.
    """

    import zipfile

    filename = (
        file.filename
        or
        "uploaded_pancreas_ct"
    )

    filename_lower = (
        filename.lower()
    )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty.",
        )

    case_id = (
        "pancreas_"
        +
        uuid.uuid4().hex[:12]
    )

    volume = None
    spacing_mm = None
    input_type = None
    slice_count = None

    try:

        # ==================================================
        # OPTION 1 — NIfTI input
        # ==================================================

        if (
            filename_lower.endswith(".nii")
            or
            filename_lower.endswith(".nii.gz")
        ):

            suffix = (
                ".nii.gz"
                if filename_lower.endswith(".nii.gz")
                else ".nii"
            )

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:

                temp_file.write(
                    content
                )

                temp_path = temp_file.name

            try:

                nifti_image = nib.load(
                    temp_path
                )

                volume_xyz = (
                    nifti_image.get_fdata(
                        dtype=np.float32
                    )
                )

                if volume_xyz.ndim != 3:

                    raise PancreasAnalysisError(
                        f"Expected 3D NIfTI CT volume, "
                        f"got {volume_xyz.shape}"
                    )

                # NIfTI [X,Y,Z] -> internal [Z,Y,X]

                volume = np.transpose(
                    volume_xyz,
                    (
                        2,
                        1,
                        0,
                    ),
                )

                volume = np.ascontiguousarray(
                    volume,
                    dtype=np.float32,
                )

                spacing_xyz = tuple(
                    float(x)
                    for x
                    in nifti_image.header.get_zooms()[:3]
                )

                spacing_mm = (
                    spacing_xyz[2],
                    spacing_xyz[1],
                    spacing_xyz[0],
                )

                slice_count = int(
                    volume.shape[0]
                )

                input_type = (
                    "NIfTI CT volume"
                )

            finally:

                if os.path.exists(
                    temp_path
                ):
                    os.remove(
                        temp_path
                    )

        # ==================================================
        # OPTION 2 — ZIP DICOM input
        # ==================================================

        elif filename_lower.endswith(".zip"):

            with tempfile.TemporaryDirectory() as temp_dir:

                temp_directory = Path(
                    temp_dir
                )

                zip_path = (
                    temp_directory
                    /
                    "uploaded_pancreas_ct.zip"
                )

                zip_path.write_bytes(
                    content
                )

                extract_directory = (
                    temp_directory
                    /
                    "dicom_series"
                )

                extract_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with zipfile.ZipFile(
                    zip_path,
                    "r",
                ) as archive:

                    root = (
                        extract_directory
                        .resolve()
                    )

                    for member in archive.infolist():

                        destination = (
                            extract_directory
                            /
                            member.filename
                        ).resolve()

                        try:

                            destination.relative_to(
                                root
                            )

                        except ValueError as exc:

                            raise PancreasAnalysisError(
                                "Unsafe ZIP structure detected."
                            ) from exc

                    archive.extractall(
                        extract_directory
                    )

                dicom_volume = (
                    load_liver_ct_dicom_series(
                        extract_directory
                    )
                )

                if dicom_volume.modality.upper() != "CT":

                    raise PancreasAnalysisError(
                        "The uploaded DICOM study is not CT."
                    )

                volume = (
                    dicom_volume.volume
                )

                spacing_mm = (
                    dicom_volume.spacing_mm
                )

                slice_count = (
                    dicom_volume.slice_count
                )

                input_type = (
                    "ZIP containing DICOM CT series"
                )

        else:

            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported file type. "
                    "Upload .nii, .nii.gz, or .zip."
                ),
            )

        # ==================================================
        # Pancreas AI analysis
        # ==================================================

        result = (
            get_pancreas_service()
            .analyze(
                volume
            )
        )

        # ==================================================
        # Measurement
        # ==================================================

        measurement = (
            measure_pancreas_segmentation(
                result.segmentation_mask,
                spacing_mm,
            )
        )

        # ==================================================
        # Visualization
        # ==================================================

        visualization_path = (
            Path("outputs")
            /
            "pancreas_visualization"
            /
            f"{case_id}_overlay.png"
        )

        visualization = (
            save_pancreas_overlay(
                volume,
                result.segmentation_mask,
                visualization_path,
            )
        )

        # ==================================================
        # Export
        # ==================================================

        exported = (
            export_pancreas_result(
                segmentation_mask=
                    result.segmentation_mask,

                spacing_mm=
                    spacing_mm,

                case_id=
                    case_id,

                prediction=
                    result,

                measurement=
                    measurement,
            )
        )

    except zipfile.BadZipFile as exc:

        raise HTTPException(
            status_code=422,
            detail="The uploaded ZIP file is invalid.",
        ) from exc

    except LiverDicomIOError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"DICOM processing failed: {exc}",
        ) from exc

    except PancreasAnalysisError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Pancreas analysis failed: {exc}",
        ) from exc

    except PancreasMeasurementError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Pancreas measurement failed: {exc}",
        ) from exc

    except PancreasVisualizationError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Visualization failed: {exc}",
        ) from exc

    except PancreasOutputExportError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Output export failed: {exc}",
        ) from exc

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail=(
                "GPU memory was insufficient "
                "for this pancreas CT study."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Pancreas CT processing failed: {exc}",
        ) from exc

    return {

        "case_id":
            case_id,

        "uploaded_file":
            filename,

        "modality":
            "CT",

        "organ":
            "pancreas",

        "dataset":
            "MSD_Task07_Pancreas",

        "input": {

            "type":
                input_type,

            "slice_count":
                int(
                    slice_count
                ),

            "volume_shape": [
                int(x)
                for x
                in volume.shape
            ],

            "voxel_spacing_mm": {

                "z":
                    float(
                        spacing_mm[0]
                    ),

                "y":
                    float(
                        spacing_mm[1]
                    ),

                "x":
                    float(
                        spacing_mm[2]
                    ),
            },
        },

        "prediction":
            result.to_dict(),

        "measurement":
            measurement.to_dict(),

        "visualization": {

            "overlay_image":
                visualization[
                    "output_path"
                ],

            "selected_slice":
                visualization[
                    "slice_index"
                ],
        },

        "segmentation_output": {

            "mask_file":
                exported[
                    "mask_file"
                ],

            "metadata_file":
                exported[
                    "metadata_file"
                ],
        },

        "interpretation": {

            "pancreas_region_found":
                result.pancreas_detected,

            "high_threshold_cancer_region_found":
                result.cancer_detected,

            "cancer_probability_threshold":
                result.cancer_threshold,

            "meaning": (
                "Research-model output only. "
                "A negative result does not exclude pancreatic cancer, "
                "and a positive result does not confirm malignancy."
            ),
        },

        "status":
            "research_model_prediction",

        "safety": {

            "clinical_diagnosis":
                False,

            "malignancy_confirmation":
                False,

            "cancer_exclusion":
                False,

            "expert_review_required":
                True,

            "model_clinically_validated":
                False,

            "pathology_required_for_confirmation":
                True,
        },
    }
@app.post("/analyze/colon-ct")
async def analyze_colon_ct(
    file: UploadFile = File(
        ...,
        description=(
            "Upload either one NIfTI CT volume (.nii/.nii.gz) "
            "or one ZIP containing a DICOM CT series"
        ),
    ),
):
    """
    MSD Task10 Colon tumor segmentation research pipeline.

    Supported inputs:
        - .nii
        - .nii.gz
        - .zip containing DICOM CT series

    Research use only.

    The model output does not confirm or exclude colon cancer.
    """

    import zipfile

    filename = (
        file.filename
        or
        "uploaded_colon_ct"
    )

    filename_lower = filename.lower()

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is empty.",
        )

    case_id = (
        "colon_"
        +
        uuid.uuid4().hex[:12]
    )

    volume = None
    spacing_mm = None
    input_type = None
    slice_count = None

    try:

        # ==================================================
        # NIfTI
        # ==================================================

        if (
            filename_lower.endswith(".nii")
            or
            filename_lower.endswith(".nii.gz")
        ):

            suffix = (
                ".nii.gz"
                if filename_lower.endswith(".nii.gz")
                else ".nii"
            )

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:

                temp_file.write(
                    content
                )

                temp_path = temp_file.name

            try:

                nifti_image = nib.load(
                    temp_path
                )

                volume_xyz = nifti_image.get_fdata(
                    dtype=np.float32
                )

                if volume_xyz.ndim != 3:

                    raise ColonAnalysisError(
                        f"Expected 3D NIfTI CT volume, "
                        f"got {volume_xyz.shape}"
                    )

                # NIfTI XYZ -> internal ZYX

                volume = np.transpose(
                    volume_xyz,
                    (
                        2,
                        1,
                        0,
                    ),
                )

                volume = np.ascontiguousarray(
                    volume,
                    dtype=np.float32,
                )

                spacing_xyz = tuple(
                    float(x)
                    for x
                    in nifti_image.header.get_zooms()[:3]
                )

                spacing_mm = (
                    spacing_xyz[2],
                    spacing_xyz[1],
                    spacing_xyz[0],
                )

                slice_count = int(
                    volume.shape[0]
                )

                input_type = (
                    "NIfTI CT volume"
                )

            finally:

                if os.path.exists(
                    temp_path
                ):
                    os.remove(
                        temp_path
                    )

        # ==================================================
        # DICOM ZIP
        # ==================================================

        elif filename_lower.endswith(".zip"):

            with tempfile.TemporaryDirectory() as temp_dir:

                temp_directory = Path(
                    temp_dir
                )

                zip_path = (
                    temp_directory
                    /
                    "uploaded_colon_ct.zip"
                )

                zip_path.write_bytes(
                    content
                )

                extract_directory = (
                    temp_directory
                    /
                    "dicom_series"
                )

                extract_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with zipfile.ZipFile(
                    zip_path,
                    "r",
                ) as archive:

                    root = (
                        extract_directory
                        .resolve()
                    )

                    for member in archive.infolist():

                        destination = (
                            extract_directory
                            /
                            member.filename
                        ).resolve()

                        try:

                            destination.relative_to(
                                root
                            )

                        except ValueError as exc:

                            raise ColonAnalysisError(
                                "Unsafe ZIP structure detected."
                            ) from exc

                    archive.extractall(
                        extract_directory
                    )

                dicom_volume = (
                    load_liver_ct_dicom_series(
                        extract_directory
                    )
                )

                if dicom_volume.modality.upper() != "CT":

                    raise ColonAnalysisError(
                        "The uploaded DICOM study is not CT."
                    )

                volume = (
                    dicom_volume.volume
                )

                spacing_mm = (
                    dicom_volume.spacing_mm
                )

                slice_count = (
                    dicom_volume.slice_count
                )

                input_type = (
                    "ZIP containing DICOM CT series"
                )

        else:

            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported file type. "
                    "Upload .nii, .nii.gz, or .zip."
                ),
            )

        # ==================================================
        # AI analysis
        # ==================================================

        result = (
            get_colon_service()
            .analyze(
                volume
            )
        )

        # ==================================================
        # Measurement
        # ==================================================

        measurement = (
            measure_colon_segmentation(
                result.segmentation_mask,
                spacing_mm,
            )
        )

        # ==================================================
        # Visualization
        # ==================================================

        visualization_path = (
            Path("outputs")
            /
            "colon_visualization"
            /
            f"{case_id}_overlay.png"
        )

        visualization = (
            save_colon_overlay(
                volume,
                result.segmentation_mask,
                visualization_path,
            )
        )

        # ==================================================
        # Export
        # ==================================================

        exported = (
            export_colon_result(
                segmentation_mask=
                    result.segmentation_mask,

                spacing_mm=
                    spacing_mm,

                case_id=
                    case_id,

                prediction=
                    result,

                measurement=
                    measurement,
            )
        )

    except zipfile.BadZipFile as exc:

        raise HTTPException(
            status_code=422,
            detail="The uploaded ZIP file is invalid.",
        ) from exc

    except LiverDicomIOError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"DICOM processing failed: {exc}",
        ) from exc

    except ColonAnalysisError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Colon analysis failed: {exc}",
        ) from exc

    except ColonMeasurementError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Colon measurement failed: {exc}",
        ) from exc

    except ColonVisualizationError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Visualization failed: {exc}",
        ) from exc

    except ColonOutputExportError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Output export failed: {exc}",
        ) from exc

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail=(
                "GPU memory was insufficient "
                "for this colon CT study."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Colon CT processing failed: {exc}",
        ) from exc

    return {

        "case_id":
            case_id,

        "uploaded_file":
            filename,

        "modality":
            "CT",

        "organ":
            "colon",

        "dataset":
            "MSD_Task10_Colon",

        "input": {

            "type":
                input_type,

            "slice_count":
                int(
                    slice_count
                ),

            "volume_shape": [
                int(x)
                for x
                in volume.shape
            ],

            "voxel_spacing_mm": {

                "z":
                    float(
                        spacing_mm[0]
                    ),

                "y":
                    float(
                        spacing_mm[1]
                    ),

                "x":
                    float(
                        spacing_mm[2]
                    ),
            },
        },

        "prediction":
            result.to_dict(),

        "measurement":
            measurement.to_dict(),

        "visualization": {

            "overlay_image":
                visualization[
                    "output_path"
                ],

            "selected_slice":
                visualization[
                    "slice_index"
                ],
        },

        "segmentation_output": {

            "mask_file":
                exported[
                    "mask_file"
                ],

            "metadata_file":
                exported[
                    "metadata_file"
                ],
        },

        "interpretation": {

            "tumor_model_region_detected":
                result.tumor_detected,

            "tumor_probability_threshold":
                result.tumor_threshold,

            "meaning": (
                "This is an experimental research-model segmentation "
                "of a colon tumor-like region. "
                "A positive result does not confirm colon cancer "
                "and a negative result does not exclude it."
            ),
        },

        "status":
            "research_model_prediction",

        "safety": {

            "clinical_diagnosis":
                False,

            "malignancy_confirmation":
                False,

            "cancer_exclusion":
                False,

            "expert_review_required":
                True,

            "model_clinically_validated":
                False,

            "pathology_required_for_confirmation":
                True,
        },
    }
@app.post("/analyze/skin-lesion")
async def analyze_skin_lesion(
    file: UploadFile = File(
        ...,
        description=(
            "Upload a dermoscopy or skin lesion image "
            "(.jpg, .jpeg, .png)"
        ),
    ),
):
    """
    ISIC2016 skin lesion segmentation research pipeline.

    Supported:
        .jpg
        .jpeg
        .png

    This model segments the visible lesion region.
    It does NOT classify melanoma or confirm skin cancer.
    """

    filename = (
        file.filename
        or
        "uploaded_skin_image"
    )

    filename_lower = filename.lower()

    allowed_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
    )

    if not filename_lower.endswith(
        allowed_extensions
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported image type. "
                "Upload .jpg, .jpeg, or .png."
            ),
        )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=422,
            detail="Uploaded image is empty.",
        )

    case_id = (
        "skin_"
        +
        uuid.uuid4().hex[:12]
    )

    try:

        # ---------------------------------
        # Decode RGB image
        # ---------------------------------

        try:

            pil_image = Image.open(
                BytesIO(
                    content
                )
            ).convert(
                "RGB"
            )

        except Exception as exc:

            raise SkinAnalysisError(
                "Unable to decode uploaded image."
            ) from exc

        image_np = np.asarray(
            pil_image,
            dtype=np.uint8,
        ).copy()

        # ---------------------------------
        # AI segmentation
        # ---------------------------------

        result = (
            get_skin_service()
            .analyze(
                image_np
            )
        )

        # ---------------------------------
        # Visualization
        # ---------------------------------

        visualization_path = (
            Path("outputs")
            /
            "skin_visualization"
            /
            f"{case_id}_overlay.png"
        )

        visualization = (
            save_skin_overlay(
                image_np,
                result.segmentation_mask,
                visualization_path,
            )
        )

        # ---------------------------------
        # Export
        # ---------------------------------

        exported = (
            export_skin_result(
                segmentation_mask=
                    result.segmentation_mask,

                case_id=
                    case_id,

                prediction=
                    result,
            )
        )

    except SkinAnalysisError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Skin analysis failed: {exc}",
        ) from exc

    except SkinVisualizationError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Visualization failed: {exc}",
        ) from exc

    except SkinOutputExportError as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Output export failed: {exc}",
        ) from exc

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail=(
                "GPU memory was insufficient "
                "for skin lesion analysis."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=f"Skin image processing failed: {exc}",
        ) from exc

    return {

        "case_id":
            case_id,

        "uploaded_file":
            filename,

        "modality":
            "DERMOSCOPY",

        "organ":
            "skin",

        "dataset":
            "ISIC2016",

        "task":
            "lesion_segmentation",

        "input": {

            "type":
                "RGB skin lesion image",

            "height":
                int(
                    image_np.shape[0]
                ),

            "width":
                int(
                    image_np.shape[1]
                ),

            "channels":
                int(
                    image_np.shape[2]
                ),
        },

        "prediction":
            result.to_dict(),

        "visualization": {

            "overlay_image":
                visualization[
                    "output_path"
                ],
        },

        "segmentation_output": {

            "mask_file":
                exported[
                    "mask_file"
                ],

            "metadata_file":
                exported[
                    "metadata_file"
                ],
        },

        "interpretation": {

            "lesion_region_found":
                result.lesion_detected,

            "lesion_image_percentage":
                result.lesion_percentage,

            "segmentation_threshold":
                result.threshold,

            "meaning": (
                "This model segments the visible lesion region "
                "in a skin image. "
                "It does not determine whether the lesion is benign "
                "or malignant and does not diagnose melanoma."
            ),
        },

        "status":
            "research_model_prediction",

        "safety": {

            "clinical_diagnosis":
                False,

            "melanoma_classification":
                False,

            "malignancy_confirmation":
                False,

            "cancer_exclusion":
                False,

            "expert_review_required":
                True,

            "model_clinically_validated":
                False,

            "biopsy_or_pathology_required_for_confirmation":
                True,
        },
    }