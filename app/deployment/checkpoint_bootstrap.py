"""Fetch deployable research checkpoints from the public model repository."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOSITORY = "skfahim21/oncoaegis-checkpoints"

# Only trained research checkpoints are fetched. Dataset files, user uploads,
# and generated artifacts are deliberately never downloaded by a deployment.
CHECKPOINTS = (
    "busi/busi_multitask_unet_best.pt",
    "cnmc2019/cnmc2019_classifier_best.pt",
    "isic2016_segmentation/isic2016_skin_segmentation_best.pt",
    "liver_locator/liver_locator_best.pt",
    "liver_tumor/liver_tumor_best.pt",
    "liver_tumor_stage2/liver_tumor_stage2_best.pt",
    "luna16/luna16_unet_best.pt",
    "luna16_nodule/luna16_nodule_classifier_best.pt",
    "msd_brain_tumor/msd_brain_tumor_best.pt",
    "msd_colon_tumor/colon_tumor_best.pt",
    "msd_pancreas_tumor/pancreas_tumor_best.pt",
    "msd_pancreas_tumor_fast/pancreas_tumor_fast_best.pt",
    "msd_pancreas_tumor_v2/pancreas_tumor_v2_best.pt",
    "tn3k_segmentation/tn3k_unet_best.pt",
)

# The generic specialist API needs only the checkpoint(s) for the selected
# model. Keeping this mapping lets constrained deployments fetch a model on
# demand instead of downloading every research checkpoint during startup.
MODEL_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "busi_breast_segmentation": (
        "busi/busi_multitask_unet_best.pt",
    ),
    "busi_breast_classifier": (
        "busi/busi_multitask_unet_best.pt",
    ),
    "tn3k_thyroid_nodule_segmentation": (
        "tn3k_segmentation/tn3k_unet_best.pt",
    ),
    "isic2016_skin_lesion_segmentation": (
        "isic2016_segmentation/isic2016_skin_segmentation_best.pt",
    ),
    "msd_brain_tumor_segmentation": (
        "msd_brain_tumor/msd_brain_tumor_best.pt",
    ),
    "luna16_lung_segmentation": (
        "luna16/luna16_unet_best.pt",
    ),
    "luna16_nodule_detector": (
        "luna16_nodule/luna16_nodule_classifier_best.pt",
    ),
    "ircadb01_liver_tumor_segmentation": (
        "liver_locator/liver_locator_best.pt",
        "liver_tumor_stage2/liver_tumor_stage2_best.pt",
    ),
    "msd_pancreas_segmentation": (
        "msd_pancreas_tumor/pancreas_tumor_best.pt",
        "msd_pancreas_tumor_fast/pancreas_tumor_fast_best.pt",
    ),
    "msd_pancreas_tumor_segmentation": (
        "msd_pancreas_tumor/pancreas_tumor_best.pt",
        "msd_pancreas_tumor_fast/pancreas_tumor_fast_best.pt",
    ),
    "msd_colon_tumor_segmentation": (
        "msd_colon_tumor/colon_tumor_best.pt",
    ),
    "cnmc2019_all_cell_classifier": (
        "cnmc2019/cnmc2019_classifier_best.pt",
    ),
}


class CheckpointUnavailableError(RuntimeError):
    """Raised when a requested research checkpoint cannot be made available."""


def checkpoint_download_enabled() -> bool:
    configured = os.getenv("ONCOAEGIS_DOWNLOAD_CHECKPOINTS", "").strip().lower()
    if configured:
        return configured in {
        "1", "true", "yes", "on",
        }
    # Managed deployments need real weights by default. Local development
    # remains opt-in, so a developer never unexpectedly downloads models.
    return any(
        os.getenv(marker, "").strip()
        for marker in ("RENDER", "RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID")
    )


def checkpoint_prefetch_enabled() -> bool:
    """Whether all checkpoints should be fetched during application startup.

    Render keeps its historic eager behaviour. Railway uses on-demand fetching
    by default, which lowers boot time and avoids filling a small ephemeral
    filesystem before a user has requested an image analysis.
    """

    configured = os.getenv("ONCOAEGIS_PREFETCH_CHECKPOINTS", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes", "on"}
    return os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"}


def checkpoint_is_ready(path: Path) -> bool:
    """Return true only for a usable model file, never a Git-LFS pointer."""

    try:
        if not path.is_file() or path.stat().st_size <= 1024:
            return False
        with path.open("rb") as handle:
            return not handle.read(96).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def bootstrap_checkpoints(
    relative_paths: Iterable[str] | None = None,
) -> dict[str, int | bool]:
    """Download missing public model weights without blocking local development."""

    if not checkpoint_download_enabled():
        return {"enabled": False, "downloaded": 0, "available": 0, "failed": 0}

    repository = os.getenv("ONCOAEGIS_CHECKPOINT_REPOSITORY", DEFAULT_REPOSITORY).strip()
    targets = tuple(relative_paths) if relative_paths is not None else CHECKPOINTS
    downloaded = available = failed = 0
    for relative_path in targets:
        destination = PROJECT_ROOT / "checkpoints" / relative_path
        if checkpoint_is_ready(destination):
            available += 1
            continue
        candidates = [relative_path]
        if relative_path.endswith("_best.pt"):
            candidates.append(relative_path.replace("_best.pt", "_last.pt"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        success = False
        for candidate in candidates:
            temporary = destination.with_suffix(destination.suffix + ".part")
            url = f"https://huggingface.co/{repository}/resolve/main/{candidate}?download=true"
            try:
                request = Request(url, headers={"User-Agent": "OncoAegisAI/1.0"})
                with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if not checkpoint_is_ready(temporary):
                    raise OSError("downloaded file is missing or a Git-LFS pointer")
                temporary.replace(destination)
                downloaded += 1
                available += 1
                success = True
                LOGGER.info("Downloaded checkpoint: %s (source: %s)", relative_path, candidate)
                break
            except OSError as error:
                temporary.unlink(missing_ok=True)
                LOGGER.warning("Checkpoint download failed for %s: %s", candidate, error)
        if not success:
            failed += 1
    return {"enabled": True, "downloaded": downloaded, "available": available, "failed": failed}


def ensure_model_checkpoint(model_id: str) -> None:
    """Ensure a selected specialist can access its own checkpoint(s).

    It intentionally does nothing for model types that have no PyTorch
    checkpoint in this repository (for example the FlowCAP workflow).
    """

    required = MODEL_CHECKPOINTS.get(model_id, ())
    if not required:
        return

    missing = tuple(
        relative_path
        for relative_path in required
        if not checkpoint_is_ready(PROJECT_ROOT / "checkpoints" / relative_path)
    )
    if not missing:
        return

    if not checkpoint_download_enabled():
        raise CheckpointUnavailableError(
            "The selected research model is not installed in this environment. "
            "Checkpoint download is disabled."
        )

    result = bootstrap_checkpoints(missing)
    unresolved = [
        relative_path
        for relative_path in missing
        if not checkpoint_is_ready(PROJECT_ROOT / "checkpoints" / relative_path)
    ]
    if unresolved or result["failed"]:
        raise CheckpointUnavailableError(
            "The selected research model is temporarily unavailable because "
            "its trained checkpoint could not be retrieved."
        )
