"""Inference adapter for the trained BUSI research baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from src.data.busi_dataset import BUSI_CLASSES
from src.models.busi_models import BUSIMultiTaskUNet


class BUSIInferenceError(ValueError):
    """Raised when an uploaded ultrasound cannot be processed."""


class BUSIModelUnavailable(BUSIInferenceError):
    """Raised when the trained BUSI checkpoint is not available."""


@dataclass(frozen=True)
class BusiPrediction:
    original_image_shape: tuple[int, int]
    processed_image_shape: tuple[int, int]
    predicted_class: str
    class_probabilities: dict[str, float]
    model_confidence: float
    lesion_area_fraction: float
    model_name: str
    model_version: str
    warnings: tuple[str, ...]


def default_busi_checkpoint() -> Path:
    configured = os.getenv("ONCOAEGIS_BUSI_CHECKPOINT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "checkpoints" / "busi" / "busi_multitask_unet_best.pt"


class BUSIModelService:
    """Lazy-loading service that keeps raw uploads out of model responses."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        image_size: int = 256,
        device: str | torch.device = "auto",
        model: torch.nn.Module | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_busi_checkpoint()
        self.image_size = image_size
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self._model = model
        if self._model is not None:
            self._model.to(self.device).eval()

    def load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.is_file():
            raise BUSIModelUnavailable(f"BUSI checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get("model_config", {})
        try:
            self._model = BUSIMultiTaskUNet(
                num_classes=int(config.get("num_classes", len(BUSI_CLASSES))),
                base_channels=int(config.get("base_channels", 16)),
            )
        except (TypeError, ValueError) as exc:
            raise BUSIModelUnavailable("BUSI checkpoint has an invalid model configuration") from exc
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(self.device).eval()
        return self._model

    def _preprocess(self, content: bytes) -> tuple[torch.Tensor, tuple[int, int]]:
        if not content:
            raise BUSIInferenceError("Uploaded ultrasound image is empty")
        try:
            with Image.open(BytesIO(content)) as image:
                grayscale = image.convert("L")
                array = np.asarray(grayscale, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError) as exc:
            raise BUSIInferenceError("Uploaded file is not a readable raster image") from exc
        if array.ndim != 2 or min(array.shape) <= 0:
            raise BUSIInferenceError("Ultrasound image has an invalid shape")
        original_shape = (int(array.shape[0]), int(array.shape[1]))
        tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return tensor.to(self.device), original_shape

    def predict(self, content: bytes) -> BusiPrediction:
        tensor, original_shape = self._preprocess(content)
        model = self.load()
        with torch.inference_mode():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs["classification_logits"], dim=1)[0].detach().cpu()
            predicted_mask = torch.sigmoid(outputs["segmentation_logits"]) >= 0.5
        predicted_index = int(probabilities.argmax().item())
        class_probabilities = {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(BUSI_CLASSES)
        }
        return BusiPrediction(
            original_image_shape=original_shape,
            processed_image_shape=(self.image_size, self.image_size),
            predicted_class=BUSI_CLASSES[predicted_index],
            class_probabilities=class_probabilities,
            model_confidence=float(probabilities[predicted_index].item()),
            lesion_area_fraction=float(predicted_mask.float().mean().item()),
            model_name="BUSI multi-task U-Net",
            model_version="research-baseline",
            warnings=(
                "BUSI labels are dataset classes, not a confirmed cancer diagnosis.",
                "Model confidence is uncalibrated; expert review is required.",
            ),
        )


def run_busi_ultrasound(content: bytes, *, service: BUSIModelService | None = None) -> BusiPrediction:
    """Run BUSI preprocessing, segmentation and classification on one image."""

    return (service or BUSIModelService()).predict(content)
