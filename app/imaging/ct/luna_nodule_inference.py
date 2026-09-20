"""LUNA16 annotated-nodule patch classification service.

The classifier identifies an annotated-nodule-like patch from the LUNA16
research task. It is not a benign/malignant classifier and cannot confirm
lung cancer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.luna_models import LUNANoduleClassifier


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT
    / "checkpoints"
    / "luna16_nodule"
    / "luna16_nodule_classifier_best.pt"
)


class LUNANoduleInferenceError(ValueError):
    """Raised when a nodule patch cannot be analyzed safely."""


@dataclass(frozen=True)
class LUNANodulePrediction:
    predicted_class: str
    nodule_detected: bool
    class_probabilities: dict[str, float]
    confidence: float
    original_shape: tuple[int, int]
    model_name: str
    model_version: str
    warnings: tuple[str, ...]


class LUNANoduleModelService:
    """Lazy-loading service for the LUNA16 patch classifier."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        image_size: int = 128,
        device: str | torch.device = "auto",
    ) -> None:
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = int(image_size)
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self._model: torch.nn.Module | None = None
        self._model_version = "research-baseline"

    def load(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.is_file():
            raise LUNANoduleInferenceError(
                f"LUNA16 nodule checkpoint not found: {self.checkpoint_path}"
            )
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )
            config = checkpoint.get("model_config", {})
            model = LUNANoduleClassifier(
                num_classes=int(config.get("num_classes", 2)),
                base_channels=int(config.get("base_channels", 16)),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            metadata = checkpoint.get("metadata", {})
            epoch = checkpoint.get("epoch")
            self._model_version = str(
                metadata.get("model_version")
                or (
                    f"research-baseline-epoch-{epoch}"
                    if epoch
                    else "research-baseline"
                )
            )
            self._model = model
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise LUNANoduleInferenceError(
                "LUNA16 nodule checkpoint is invalid or unreadable"
            ) from exc
        return self._model

    def predict(self, patch: np.ndarray) -> LUNANodulePrediction:
        array = np.asarray(patch, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or any(size <= 0 for size in array.shape):
            raise LUNANoduleInferenceError(
                "LUNA16 nodule input must be a non-empty 2D patch"
            )
        if not np.isfinite(array).all():
            raise LUNANoduleInferenceError(
                "LUNA16 nodule input contains non-finite values"
            )
        original_shape = (int(array.shape[0]), int(array.shape[1]))
        array = np.clip(array, 0.0, 1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(self.device)
        model = self.load()
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
        negative_probability = float(probabilities[0].item())
        nodule_probability = float(probabilities[1].item())
        detected = nodule_probability >= negative_probability
        return LUNANodulePrediction(
            predicted_class=(
                "annotated-nodule-like"
                if detected
                else "no-annotated-nodule-like"
            ),
            nodule_detected=detected,
            class_probabilities={
                "no_annotated_nodule": negative_probability,
                "annotated_nodule": nodule_probability,
            },
            confidence=max(negative_probability, nodule_probability),
            original_shape=original_shape,
            model_name="LUNA16 annotated-nodule patch classifier",
            model_version=self._model_version,
            warnings=(
                "LUNA16 annotations are research nodule findings, not pathology-confirmed cancer.",
                "The classifier is not a benign/malignant model; expert review is required.",
            ),
        )
