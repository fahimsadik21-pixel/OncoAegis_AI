"""Resource-safe Railway API profile for the deployed Onco Aegis product.

The full local ``app.main`` keeps legacy research endpoints for development.
This managed profile exposes the current product contract while deferring every
PyTorch specialist import until a user explicitly starts an image analysis.
"""

from __future__ import annotations

from pathlib import Path
import hmac
import os
import tempfile

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth_api import router as auth_router
from app.api.cancer_information_api import router as cancer_information_router
from app.api.chat_history_api import router as chat_history_router
from app.clinical.document_intelligence import ClinicalDocumentError, analyze_clinical_document
from app.core.analysis_orchestrator import AnalysisOrchestrator
from app.core.capability_registry import get_capabilities
from app.core.model_router import route_specialist_model
from app.core.specialist_input import (
    MAX_SPECIALIST_FILE_BYTES,
    MAX_SPECIALIST_FILES,
    MAX_SPECIALIST_TOTAL_BYTES,
    SpecialistInputError,
    UploadedSpecialistFile,
    normalize_specialist_input,
    parse_spacing_mm,
)
from app.deployment.checkpoint_bootstrap import (
    CheckpointUnavailableError,
    bootstrap_checkpoints,
    checkpoint_prefetch_enabled,
    ensure_model_checkpoint,
)
from app.deployment.resource_policy import hosted_memory_error
from app.registry.cancer_registry import get_cancer_capabilities
from app.registry.model_registry import get_model_registry
from app.schemas.analysis_result import StandardAnalysisResult
from app.schemas.clinical import DocumentAnalysisResponse
from app.schemas.routing import ModelRouteRequest, ModelRouteResponse
from app.security.dependencies import get_current_user, require_permission
from app.security.permissions import Permission
from app.storage.case_store import (
    CaseDeletedError,
    CaseNotFoundError,
    InvalidCaseIDError,
    StorageConflictError,
    StorageError,
    get_case_store,
    normalize_actor_id,
)


APP_VERSION = "0.8.1"
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title="Onco Aegis AI",
    description="Resource-safe managed API for cancer information and research imaging analysis",
    version=APP_VERSION,
)
app.include_router(auth_router)
app.include_router(cancer_information_router)
app.include_router(chat_history_router)


@app.on_event("startup")
def download_deployment_checkpoints() -> None:
    if checkpoint_prefetch_enabled():
        bootstrap_checkpoints()


if WEB_ROOT.is_dir():
    app.mount("/app/assets", StaticFiles(directory=str(WEB_ROOT)), name="web-assets")
    app.mount("/assets", StaticFiles(directory=str(WEB_ROOT / "assets")), name="frontend-assets")

    @app.get("/app", include_in_schema=False)
    @app.get("/app/", include_in_schema=False)
    def web_app() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/styles.css", include_in_schema=False)
    def web_styles() -> FileResponse:
        return FileResponse(WEB_ROOT / "styles.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    def web_script() -> FileResponse:
        return FileResponse(WEB_ROOT / "app.js", media_type="application/javascript")


async def _read_specialist_uploads(files: list[UploadFile]) -> list[UploadedSpecialistFile]:
    if len(files) > MAX_SPECIALIST_FILES:
        raise HTTPException(status_code=400, detail="Too many uploaded files")
    uploads: list[UploadedSpecialistFile] = []
    total_bytes = 0
    for file in files:
        content = await file.read()
        if len(content) > MAX_SPECIALIST_FILE_BYTES:
            raise HTTPException(status_code=413, detail="An uploaded file is too large")
        total_bytes += len(content)
        if total_bytes > MAX_SPECIALIST_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Total upload size is too large")
        uploads.append(UploadedSpecialistFile(name=file.filename or "upload", content=content))
    return uploads


def _storage_actor(actor_id: object | None) -> str:
    return normalize_actor_id(actor_id if isinstance(actor_id, str) else None)


def _require_storage_admin(admin_token: str | None) -> None:
    expected = os.getenv("ONCOAEGIS_STORAGE_ADMIN_TOKEN", "").strip()
    if not expected or not admin_token or not hmac.compare_digest(expected, admin_token):
        raise HTTPException(status_code=403, detail="Storage administration is not authorized")


def _safe_specialist_error(exc: Exception, model_name: str) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "cannot allocate" in message:
        return (
            f"{model_name} exceeded the currently available analysis memory. "
            "Increase Railway memory to at least 2 GB before retrying this original 3D scan."
        )
    if "no foreground region" in message:
        return (
            "The selected research model could not localise the target anatomy in this volume. "
            "Confirm this is the original CT acquisition for the selected organ, not a screenshot, mask, or unrelated scan."
        )
    return (
        f"{model_name} could not complete this research analysis. The service is still available. "
        "Confirm the selected route and original imaging acquisition, then retry."
    )


@app.get("/")
def root() -> dict[str, object]:
    return {"name": "Onco Aegis AI", "version": APP_VERSION, "status": "ready"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "version": APP_VERSION}


@app.get("/system/status")
def system_status() -> dict[str, object]:
    return {"status": "healthy", "profile": "railway-resource-safe", "version": APP_VERSION}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    return get_capabilities()


@app.get("/models")
def models() -> dict[str, object]:
    return {"models": [spec.to_dict() for spec in get_model_registry().list()]}


@app.get("/cancer-capabilities")
def cancer_capabilities() -> dict[str, object]:
    return get_cancer_capabilities()


@app.post("/route/model", response_model=ModelRouteResponse)
def route_model(request: ModelRouteRequest) -> ModelRouteResponse:
    decision = route_specialist_model(
        modality=request.modality,
        organ=request.organ,
        task=request.task,
        cancer_type=request.cancer_type,
        require_available=request.require_available,
    )
    return decision.to_dict()


@app.post("/analyze/specialist", response_model=StandardAnalysisResult)
async def analyze_specialist(
    model_id: str = Form(...),
    files: list[UploadFile] = File(...),
    modality: str | None = Form(None),
    organ: str | None = Form(None),
    task: str | None = Form(None),
    cancer_type: str | None = Form(None),
    case_id: str | None = Form(None),
    spacing_mm: str | None = Form(None),
    actor_id: str | None = Header(default=None, alias="X-Actor-ID"),
    current_user=Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one input file is required")
    uploaded = await _read_specialist_uploads(files)
    model_spec = get_model_registry().get_spec(model_id)
    if model_spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown specialist model: {model_id}")
    memory_error = hosted_memory_error(model_spec)
    if memory_error:
        raise HTTPException(status_code=503, detail=memory_error)

    try:
        owner_user_id = getattr(current_user, "user_id", None)
        if task or cancer_type:
            decision = route_specialist_model(
                modality=modality or model_spec.modality,
                organ=organ or model_spec.organ,
                task=task,
                cancer_type=cancer_type,
                require_available=True,
            )
            if decision.selected_model_id != model_id:
                raise SpecialistInputError(
                    "Requested model does not match the supplied route: "
                    f"{decision.reason}"
                )
        parsed_spacing = parse_spacing_mm(spacing_mm)
        with tempfile.TemporaryDirectory(prefix="oncoaegis_specialist_") as workspace:
            normalized = normalize_specialist_input(
                model_id=model_id,
                files=uploaded,
                workspace=workspace,
                modality=modality,
                organ=organ,
                spacing_mm=parsed_spacing,
            )
            ensure_model_checkpoint(model_id)
            result = AnalysisOrchestrator().execute_analysis(
                model_id=model_id,
                input_data=normalized.data,
                input_metadata=normalized.metadata,
            )
    except CheckpointUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{model_spec.name} cannot run in this deployment because its trained "
                f"checkpoint could not be retrieved. Required model input: {model_spec.input_type}."
            ),
        ) from exc
    except SpecialistInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=_safe_specialist_error(exc, model_spec.name)) from exc

    requested_case_id = case_id if isinstance(case_id, str) else None
    if requested_case_id:
        result.case_id = requested_case_id
    result.provenance["api"] = {
        "endpoint": "/analyze/specialist",
        "request_model_id": model_id,
        "file_count": len(files),
        "task": task,
        "cancer_type": cancer_type,
    }
    try:
        get_case_store().store_result(
            result,
            files=uploaded,
            actor_id=_storage_actor(actor_id),
            case_id=requested_case_id,
            owner_user_id=owner_user_id or actor_id,
        )
    except InvalidCaseIDError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CaseDeletedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StorageConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Analysis completed but result storage was unavailable") from exc
    return result


@app.post("/analyze/document", response_model=DocumentAnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
):
    content = await file.read()
    try:
        analysis = analyze_clinical_document(file.filename or "uploaded_document", content)
    except ClinicalDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**analysis.to_dict(), "next_stage": "evidence_fusion"}


@app.get("/storage/policy")
def storage_policy(current_user=Depends(require_permission(Permission.VIEW_RESULT))):
    try:
        return get_case_store().policy()
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Storage is unavailable") from exc


@app.get("/cases")
def list_cases(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_deleted: bool = Query(False),
    current_user=Depends(require_permission(Permission.VIEW_RESULT)),
):
    try:
        return get_case_store().list_cases(
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            actor_id=current_user.user_id,
            owner_user_id=current_user.user_id,
        )
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Case history is unavailable") from exc


@app.get("/cases/{case_id}/results/{analysis_id}")
def get_case_result(
    case_id: str,
    analysis_id: str,
    current_user=Depends(require_permission(Permission.VIEW_RESULT)),
):
    try:
        return get_case_store().get_result(
            case_id,
            analysis_id,
            actor_id=current_user.user_id,
            owner_user_id=current_user.user_id,
        )
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Case result is unavailable") from exc


@app.get("/cases/{case_id}/results")
def list_case_results(
    case_id: str,
    current_user=Depends(require_permission(Permission.VIEW_RESULT)),
):
    try:
        return get_case_store().list_results(
            case_id,
            actor_id=current_user.user_id,
            owner_user_id=current_user.user_id,
        )
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Case results are unavailable") from exc


@app.get("/cases/{case_id}/audit")
def list_case_audit(
    case_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user=Depends(require_permission(Permission.VIEW_AUDIT)),
):
    try:
        return get_case_store().list_audit(
            case_id,
            limit=limit,
            actor_id=current_user.user_id,
            owner_user_id=current_user.user_id,
        )
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Case audit is unavailable") from exc


@app.delete("/cases/{case_id}")
def delete_case(
    case_id: str,
    reason: str | None = Query(None, max_length=500),
    admin_token: str | None = Header(default=None, alias="X-Storage-Admin-Token"),
    current_user=Depends(require_permission(Permission.MANAGE_USERS)),
):
    _require_storage_admin(admin_token)
    try:
        deleted = get_case_store().delete_case(
            case_id,
            actor_id=current_user.user_id,
            owner_user_id=current_user.user_id,
            reason=(reason or "user_requested_deletion"),
        )
        return {"status": "deleted", "case": deleted}
    except CaseNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Case deletion is unavailable") from exc


@app.post("/admin/storage/retention/purge")
def purge_expired_cases(
    admin_token: str | None = Header(default=None, alias="X-Storage-Admin-Token"),
    current_user=Depends(require_permission(Permission.MANAGE_USERS)),
):
    _require_storage_admin(admin_token)
    try:
        return get_case_store().purge_expired_cases(actor_id=current_user.user_id)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Retention purge is unavailable") from exc
