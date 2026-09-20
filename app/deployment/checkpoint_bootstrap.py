"""Fetch deployable research checkpoints from the public model repository."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOSITORY = "skfahim21/oncoaegis-checkpoints"

# Only the validated best checkpoints are required for inference.  Dataset files,
# last-epoch checkpoints, and user uploads are deliberately never downloaded.
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


def checkpoint_download_enabled() -> bool:
    return os.getenv("ONCOAEGIS_DOWNLOAD_CHECKPOINTS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def bootstrap_checkpoints() -> dict[str, int | bool]:
    """Download missing public model weights without blocking local development."""

    if not checkpoint_download_enabled():
        return {"enabled": False, "downloaded": 0, "available": 0, "failed": 0}

    repository = os.getenv("ONCOAEGIS_CHECKPOINT_REPOSITORY", DEFAULT_REPOSITORY).strip()
    downloaded = available = failed = 0
    for relative_path in CHECKPOINTS:
        destination = PROJECT_ROOT / "checkpoints" / relative_path
        if destination.is_file() and destination.stat().st_size > 1024:
            available += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        url = f"https://huggingface.co/{repository}/resolve/main/{relative_path}?download=true"
        try:
            request = Request(url, headers={"User-Agent": "OncoAegisAI/1.0"})
            with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size <= 1024:
                raise OSError("downloaded checkpoint is unexpectedly small")
            temporary.replace(destination)
            downloaded += 1
            available += 1
            LOGGER.info("Downloaded checkpoint: %s", relative_path)
        except OSError as error:
            failed += 1
            temporary.unlink(missing_ok=True)
            LOGGER.warning("Checkpoint download failed for %s: %s", relative_path, error)
    return {"enabled": True, "downloaded": downloaded, "available": available, "failed": failed}
