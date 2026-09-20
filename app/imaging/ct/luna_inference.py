"""Inference adapter for the existing LUNA16 lung-segmentation baseline.

The LUNA16 checkpoint segments lung anatomy.  It is deliberately exposed as
an anatomy/segmentation result, not as a lung-cancer or malignancy classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet import UNet


class CTInferenceError(ValueError):
    """Raised when a CT volume cannot be passed safely to the model."""


class LUNAModelUnavailable(CTInferenceError):
    """Raised when the registered LUNA16 checkpoint cannot be loaded."""


def default_luna_checkpoint() -> Path:
    configured = os.getenv("ONCOAEGIS_LUNA_CHECKPOINT")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "luna16"
        / "luna16_unet_best.pt"
    )


@dataclass(frozen=True)
class LUNACTPrediction:
    """Safe summary of a 3D volume segmentation prediction."""

    input_volume_shape: tuple[int, int, int]
    output_mask_shape: tuple[int, int, int]
    model_input_size: tuple[int, int]
    positive_slice_count: int
    positive_voxel_count: int
    segmented_fraction: float
    segmented_volume_mm3: float
    model_name: str
    model_version: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_volume_shape": self.input_volume_shape,
            "output_mask_shape": self.output_mask_shape,
            "model_input_size": self.model_input_size,
            "positive_slice_count": self.positive_slice_count,
            "positive_voxel_count": self.positive_voxel_count,
            "segmented_fraction": self.segmented_fraction,
            "segmented_volume_mm3": self.segmented_volume_mm3,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "warnings": list(self.warnings),
        }


class LUNA16ModelService:
    """Lazy-loading, batched service for the local LUNA16 U-Net."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        image_size: int = 64,
        batch_size: int = 16,
        device: str | torch.device = "auto",
        model: torch.nn.Module | None = None,
    ) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else default_luna_checkpoint()
        )
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self._model = model
        self._model_version = "injected-model"
        if self._model is not None:
            self._model.to(self.device).eval()

    def load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.is_file():
            raise LUNAModelUnavailable(
                f"LUNA16 checkpoint not found: {self.checkpoint_path}"
            )
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
            config = checkpoint.get("model_config", {})
            self._model = UNet(
                in_channels=int(config.get("in_channels", 1)),
                out_channels=int(config.get("out_channels", 1)),
                base_channels=int(config.get("base_channels", 16)),
            )
            self._model.load_state_dict(checkpoint["model_state"])
            self._model.to(self.device).eval()
            metadata = checkpoint.get("metadata", {})
            epoch = checkpoint.get("epoch")
            self._model_version = str(
                metadata.get("model_version")
                or (f"research-baseline-epoch-{epoch}" if epoch else "research-baseline")
            )
        except LUNAModelUnavailable:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise LUNAModelUnavailable(
                "LUNA16 checkpoint has an invalid or unreadable model state"
            ) from exc
        return self._model

    @staticmethod
    def _validate_volume(volume: np.ndarray) -> np.ndarray:
        array = np.asarray(volume, dtype=np.float32)
        if array.ndim != 3 or any(size <= 0 for size in array.shape):
            raise CTInferenceError("CT model input must be a non-empty 3D volume")
        if not np.isfinite(array).all():
            raise CTInferenceError("CT model input contains non-finite values")
        return np.clip(array, 0.0, 1.0)

    @staticmethod
    def _validate_spacing(spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(spacing_mm) != 3:
            raise CTInferenceError("spacing_mm must contain three values")
        values = tuple(float(value) for value in spacing_mm)
        if any(value <= 0 or not np.isfinite(value) for value in values):
            raise CTInferenceError("spacing_mm must contain positive finite values")
        return values

    def predict(
        self,
        volume: np.ndarray,
        *,
        spacing_mm: tuple[float, float, float],
    ) -> LUNACTPrediction:
        """Segment all slices and return a geometry-aware safe summary."""

        array = self._validate_volume(volume)
        spacing = self._validate_spacing(spacing_mm)
        model = self.load()
        depth, height, width = (int(value) for value in array.shape)
        source = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(1)
        predicted_chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, depth, self.batch_size):
                batch = source[start : start + self.batch_size].to(self.device)
                if tuple(batch.shape[-2:]) != (self.image_size, self.image_size):
                    batch = F.interpolate(
                        batch,
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                outputs = model(batch)
                if isinstance(outputs, dict):
                    outputs = outputs.get("segmentation_logits")
                if not isinstance(outputs, torch.Tensor):
                    raise CTInferenceError(
                        "LUNA16 model did not return segmentation logits"
                    )
                masks = (torch.sigmoid(outputs) >= 0.5).float()
                if tuple(masks.shape[-2:]) != (height, width):
                    masks = F.interpolate(
                        masks,
                        size=(height, width),
                        mode="nearest",
                    )
                predicted_chunks.append(
                    masks[:, 0].to(device="cpu", dtype=torch.uint8).numpy()
                )

        predicted_mask = np.concatenate(predicted_chunks, axis=0)
        positive_voxels = int(predicted_mask.sum())
        positive_slices = int(np.any(predicted_mask > 0, axis=(1, 2)).sum())
        total_voxels = int(predicted_mask.size)
        warnings = (
            "LUNA16 output is lung anatomy segmentation, not nodule or cancer confirmation.",
            "This research baseline is not clinically calibrated; expert review is required.",
        )
        if positive_voxels == 0:
            warnings += (
                "The model produced an empty lung mask; input quality and model suitability must be reviewed.",
            )

        return LUNACTPrediction(
            input_volume_shape=(depth, height, width),
            output_mask_shape=tuple(int(value) for value in predicted_mask.shape),
            model_input_size=(self.image_size, self.image_size),
            positive_slice_count=positive_slices,
            positive_voxel_count=positive_voxels,
            segmented_fraction=(positive_voxels / total_voxels if total_voxels else 0.0),
            segmented_volume_mm3=float(positive_voxels * np.prod(spacing)),
            model_name="LUNA16 lung segmentation U-Net",
            model_version=self._model_version,
            warnings=warnings,
        )


def run_luna_ct_segmentation(
    volume: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    service: LUNA16ModelService | None = None,
) -> LUNACTPrediction:
    """Run the registered LUNA16 lung-segmentation specialist."""

    return (service or LUNA16ModelService()).predict(
        volume,
        spacing_mm=spacing_mm,
    )
