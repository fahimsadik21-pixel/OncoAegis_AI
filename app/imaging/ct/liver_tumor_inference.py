"""
IRCADb01 Liver Tumor Segmentation Inference

Research use only.

Classes:
    0 = background
    1 = liver
    2 = tumor
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.models.ct.liver_tumor_unet import LiverTumor3DUNet
from app.imaging.ct.liver_preprocessing import (
    TARGET_SIZE,
    normalize_ct,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT_PATH = (
    _PROJECT_ROOT
    / "checkpoints"
    / "liver_tumor"
    / "liver_tumor_best.pt"
)


class LiverTumorInferenceError(Exception):
    pass


@dataclass
class LiverTumorPrediction:

    segmentation_mask: np.ndarray

    liver_voxels: int

    tumor_voxels: int

    tumor_detected: bool

    model_name: str

    model_version: str

    processed_shape: tuple[int, int, int]

    original_shape: tuple[int, int, int]

    def to_dict(self):

        return {
            "liver_voxels": self.liver_voxels,
            "tumor_voxels": self.tumor_voxels,
            "tumor_detected": self.tumor_detected,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "processed_shape": self.processed_shape,
            "original_shape": self.original_shape,
        }


class LiverTumorInferenceService:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str | None = None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise LiverTumorInferenceError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.model = LiverTumor3DUNet(
            in_channels=1,
            out_channels=3,
        )

        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def _prepare_volume(
        self,
        volume: np.ndarray,
    ):

        if volume.ndim != 3:
            raise LiverTumorInferenceError(
                f"Expected 3D CT volume, got shape {volume.shape}"
            )

        original_shape = tuple(
            int(x)
            for x in volume.shape
        )

        volume = normalize_ct(
            volume
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                volume
            )
        ).float()

        tensor = tensor.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        tensor = F.interpolate(
            tensor,
            size=TARGET_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        return (
            tensor,
            original_shape,
        )

    def _restore_mask(
        self,
        mask: torch.Tensor,
        original_shape: tuple[int, int, int],
    ):

        mask = mask.float()

        mask = mask.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        mask = F.interpolate(
            mask,
            size=original_shape,
            mode="nearest",
        )

        mask = mask.squeeze(
            0
        ).squeeze(
            0
        )

        return mask.to(
            torch.uint8
        )

    def predict(
        self,
        volume: np.ndarray,
    ) -> LiverTumorPrediction:

        input_tensor, original_shape = (
            self._prepare_volume(
                volume
            )
        )

        input_tensor = input_tensor.to(
            self.device
        )

        use_amp = (
            self.device.type == "cuda"
        )

        try:

            with torch.no_grad():

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                ):

                    logits = self.model(
                        input_tensor
                    )

                predicted_mask = torch.argmax(
                    logits,
                    dim=1,
                )[0]

        except torch.cuda.OutOfMemoryError as exc:

            if self.device.type == "cuda":

                torch.cuda.empty_cache()

            raise LiverTumorInferenceError(
                "CUDA out of memory during liver CT inference."
            ) from exc

        restored_mask = self._restore_mask(
            predicted_mask.cpu(),
            original_shape,
        )

        mask_np = restored_mask.numpy()

        liver_voxels = int(
            np.count_nonzero(
                mask_np == 1
            )
        )

        tumor_voxels = int(
            np.count_nonzero(
                mask_np == 2
            )
        )

        tumor_detected = (
            tumor_voxels > 0
        )

        return LiverTumorPrediction(
            segmentation_mask=mask_np,
            liver_voxels=liver_voxels,
            tumor_voxels=tumor_voxels,
            tumor_detected=tumor_detected,
            model_name="IRCADb01 Liver Tumor 3D U-Net",
            model_version="research-baseline-v1",
            processed_shape=tuple(
                TARGET_SIZE
            ),
            original_shape=original_shape,
        )


def load_liver_tumor_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str | None = None,
):

    return LiverTumorInferenceService(
        checkpoint_path=checkpoint_path,
        device=device,
    )