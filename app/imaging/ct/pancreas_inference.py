"""
MSD Task07 Pancreas CT inference.

Classes:
    0 = background
    1 = pancreas
    2 = cancer

Uses:
    checkpoints/msd_pancreas_tumor_fast/pancreas_tumor_fast_best.pt

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.models.ct.pancreas_tumor_unet import PancreasTumor3DUNet


DEFAULT_CHECKPOINT = Path(
    "checkpoints/msd_pancreas_tumor_fast/"
    "pancreas_tumor_fast_best.pt"
)

TARGET_SIZE = (
    96,
    96,
    96,
)

HU_MIN = -125.0
HU_MAX = 275.0


class PancreasModelUnavailable(Exception):
    pass


class PancreasInferenceError(Exception):
    pass


@dataclass
class PancreasPrediction:

    segmentation_mask: np.ndarray

    pancreas_detected: bool

    cancer_detected: bool

    pancreas_voxels: int

    cancer_voxels: int

    original_shape: tuple[int, int, int]

    model_name: str

    model_version: str

    status: str

    def to_dict(self):

        return {
            "pancreas_detected":
                self.pancreas_detected,

            "cancer_detected":
                self.cancer_detected,

            "pancreas_voxels":
                self.pancreas_voxels,

            "cancer_voxels":
                self.cancer_voxels,

            "original_shape":
                self.original_shape,

            "model_name":
                self.model_name,

            "model_version":
                self.model_version,

            "status":
                self.status,
        }


def normalize_ct(
    volume: np.ndarray,
) -> np.ndarray:

    volume = volume.astype(
        np.float32
    )

    volume = np.clip(
        volume,
        HU_MIN,
        HU_MAX,
    )

    volume = (
        volume
        -
        HU_MIN
    ) / (
        HU_MAX
        -
        HU_MIN
    )

    return volume.astype(
        np.float32
    )


def resize_volume(
    volume: np.ndarray,
    size=TARGET_SIZE,
) -> torch.Tensor:

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            volume
        )
    ).float()

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    tensor = F.interpolate(
        tensor,
        size=size,
        mode="trilinear",
        align_corners=False,
    )

    return tensor


def restore_mask(
    mask: torch.Tensor,
    original_shape: tuple[int, int, int],
) -> np.ndarray:

    mask = (
        mask
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    mask = F.interpolate(
        mask,
        size=original_shape,
        mode="nearest",
    )

    mask = (
        mask
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return mask


class PancreasTumorInferenceService:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():

            raise PancreasModelUnavailable(
                f"Pancreas checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else
                "cpu"
            )

        self.device = torch.device(
            device
        )

        self.model = (
            PancreasTumor3DUNet()
            .to(
                self.device
            )
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:

            raise PancreasModelUnavailable(
                "Invalid pancreas checkpoint."
            )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.model_name = checkpoint.get(
            "model_name",
            "MSD Task07 Pancreas Tumor Model",
        )

        self.model_version = (
            f"research-baseline-epoch-"
            f"{checkpoint.get('epoch', 'unknown')}"
        )

    @torch.no_grad()
    def predict(
        self,
        volume: np.ndarray,
    ) -> PancreasPrediction:

        try:

            if volume.ndim != 3:

                raise PancreasInferenceError(
                    f"Expected 3D CT volume, "
                    f"got {volume.shape}"
                )

            original_shape = tuple(
                int(x)
                for x
                in volume.shape
            )

            normalized = normalize_ct(
                volume
            )

            input_tensor = resize_volume(
                normalized,
                TARGET_SIZE,
            ).to(
                self.device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(
                    self.device.type
                    ==
                    "cuda"
                ),
            ):

                logits = self.model(
                    input_tensor
                )

            prediction_small = torch.argmax(
                logits,
                dim=1,
            )[0]

            segmentation_mask = restore_mask(
                prediction_small,
                original_shape,
            )

            pancreas_voxels = int(
                np.count_nonzero(
                    segmentation_mask
                    ==
                    1
                )
            )

            cancer_voxels = int(
                np.count_nonzero(
                    segmentation_mask
                    ==
                    2
                )
            )

            return PancreasPrediction(

                segmentation_mask=
                    segmentation_mask,

                pancreas_detected=
                    pancreas_voxels
                    >
                    0,

                cancer_detected=
                    cancer_voxels
                    >
                    0,

                pancreas_voxels=
                    pancreas_voxels,

                cancer_voxels=
                    cancer_voxels,

                original_shape=
                    original_shape,

                model_name=
                    self.model_name,

                model_version=
                    self.model_version,

                status=
                    "research_model_prediction",
            )

        except PancreasInferenceError:
            raise

        except RuntimeError as exc:

            if (
                "out of memory"
                in
                str(exc).lower()
            ):

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise PancreasInferenceError(
                    "GPU memory was insufficient "
                    "for pancreas inference."
                ) from exc

            raise PancreasInferenceError(
                str(exc)
            ) from exc

        except Exception as exc:

            raise PancreasInferenceError(
                str(exc)
            ) from exc


def run_pancreas_inference(
    volume: np.ndarray,
    service: PancreasTumorInferenceService | None = None,
):

    if service is None:

        service = (
            PancreasTumorInferenceService()
        )

    return service.predict(
        volume
    )


if __name__ == "__main__":

    print(
        "Pancreas inference module ready."
    )