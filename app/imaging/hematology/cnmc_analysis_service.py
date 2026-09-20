"""
C-NMC 2019 single-cell ALL vs HEM classification service.

Class 0 = HEM-like
Class 1 = ALL-like

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.imaging.hematology.cnmc_preprocessing import (
    preprocess_cnmc_image,
)

from app.models.hematology.cnmc_classifier import (
    CNMCClassifier,
)


CHECKPOINT_PATH = Path(
    "checkpoints/cnmc2019/cnmc2019_classifier_best.pt"
)


class CNMCAnalysisError(Exception):
    pass


@dataclass
class CNMCAnalysisResult:

    predicted_class: str

    predicted_index: int

    hem_probability: float

    all_probability: float

    confidence: float

    original_shape: tuple[int, int, int]

    model_name: str

    model_version: str

    status: str

    def to_dict(self):

        return {
            "predicted_class":
                self.predicted_class,

            "predicted_index":
                self.predicted_index,

            "hem_probability":
                round(
                    self.hem_probability,
                    6,
                ),

            "all_probability":
                round(
                    self.all_probability,
                    6,
                ),

            "confidence":
                round(
                    self.confidence,
                    6,
                ),

            "original_shape":
                self.original_shape,

            "model_name":
                self.model_name,

            "model_version":
                self.model_version,

            "status":
                self.status,
        }


class CNMCAnalysisService:

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        device: str | torch.device | None = None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():

            raise CNMCAnalysisError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else
                "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model = CNMCClassifier()

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.model_name = checkpoint.get(
            "model_name",
            "C-NMC 2019 ALL Cell Classifier",
        )

        self.model_version = checkpoint.get(
            "model_version",
            "research-baseline",
        )

    def analyze(
        self,
        image,
    ) -> CNMCAnalysisResult:

        if isinstance(
            image,
            Image.Image,
        ):

            image = image.convert(
                "RGB"
            )

            image_np = np.asarray(
                image,
                dtype=np.uint8,
            ).copy()

        else:

            image_np = np.asarray(
                image
            ).copy()

        if image_np.ndim != 3:

            raise CNMCAnalysisError(
                f"Expected RGB image, got "
                f"{image_np.shape}"
            )

        if image_np.shape[2] != 3:

            raise CNMCAnalysisError(
                f"Expected 3 channels, got "
                f"{image_np.shape}"
            )

        original_shape = tuple(
            int(v)
            for v
            in image_np.shape
        )

        tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    image_np
                ).copy()
            )
            .permute(
                2,
                0,
                1,
            )
            .float()
        )

        processed = preprocess_cnmc_image(
            tensor
        )

        processed = (
            processed
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        with torch.no_grad():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(
                    self.device.type
                    ==
                    "cuda"
                ),
            ):

                logits = self.model(
                    processed
                )

                probabilities = torch.softmax(
                    logits.float(),
                    dim=1,
                )[0]

        hem_probability = float(
            probabilities[0].cpu()
        )

        all_probability = float(
            probabilities[1].cpu()
        )

        predicted_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        if predicted_index == 1:

            predicted_class = (
                "ALL-like"
            )

            confidence = (
                all_probability
            )

        else:

            predicted_class = (
                "HEM-like"
            )

            confidence = (
                hem_probability
            )

        return CNMCAnalysisResult(

            predicted_class=
                predicted_class,

            predicted_index=
                predicted_index,

            hem_probability=
                hem_probability,

            all_probability=
                all_probability,

            confidence=
                confidence,

            original_shape=
                original_shape,

            model_name=
                self.model_name,

            model_version=
                self.model_version,

            status=
                "research_model_prediction",
        )