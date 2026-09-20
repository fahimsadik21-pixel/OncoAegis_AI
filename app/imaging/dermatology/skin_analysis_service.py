"""
ISIC2016 skin lesion segmentation analysis service.

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from app.imaging.dermatology.skin_preprocessing import (
    preprocess_skin_image,
)

from app.models.dermatology.skin_lesion_unet import (
    SkinLesionUNet2D,
)


CHECKPOINT_PATH = Path(
    "checkpoints/isic2016_segmentation/"
    "isic2016_skin_segmentation_best.pt"
)

LESION_THRESHOLD = 0.50


class SkinAnalysisError(Exception):
    pass


@dataclass
class SkinAnalysisResult:

    lesion_detected: bool

    lesion_pixels: int

    lesion_fraction: float

    lesion_percentage: float

    original_shape: tuple[int, int]

    threshold: float

    model_name: str

    model_version: str

    status: str

    segmentation_mask: np.ndarray

    def to_dict(self):

        return {
            "lesion_detected":
                self.lesion_detected,

            "lesion_pixels":
                self.lesion_pixels,

            "lesion_fraction":
                round(
                    self.lesion_fraction,
                    6,
                ),

            "lesion_percentage":
                round(
                    self.lesion_percentage,
                    3,
                ),

            "original_shape":
                self.original_shape,

            "threshold":
                self.threshold,

            "model_name":
                self.model_name,

            "model_version":
                self.model_version,

            "status":
                self.status,
        }


class SkinAnalysisService:

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        device: str | torch.device | None = None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():

            raise SkinAnalysisError(
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

        self.model = SkinLesionUNet2D()

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
            "ISIC2016 Skin Lesion 2D U-Net",
        )

        self.model_version = checkpoint.get(
            "model_version",
            "research-baseline",
        )

    def analyze(
        self,
        image,
    ) -> SkinAnalysisResult:

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
            )

        else:

            image_np = np.asarray(
                image
            )

        if image_np.ndim != 3:

            raise SkinAnalysisError(
                f"Expected RGB image, got "
                f"{image_np.shape}"
            )

        if image_np.shape[2] != 3:

            raise SkinAnalysisError(
                f"Expected 3 RGB channels, got "
                f"{image_np.shape}"
            )

        original_height = int(
            image_np.shape[0]
        )

        original_width = int(
            image_np.shape[1]
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

        processed = preprocess_skin_image(
            tensor
        )

        processed = (
            processed
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        try:

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
                        logits,
                        dim=1,
                    )

                    lesion_probability = probabilities[
                        :,
                        1:2,
                    ]

            restored_probability = F.interpolate(
                lesion_probability.float(),
                size=(
                    original_height,
                    original_width,
                ),
                mode="bilinear",
                align_corners=False,
            )

            restored_probability = (
                restored_probability
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )

            segmentation_mask = (
                restored_probability
                >=
                LESION_THRESHOLD
            ).astype(
                np.uint8
            )

        except torch.cuda.OutOfMemoryError as exc:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise SkinAnalysisError(
                "GPU memory was insufficient "
                "for skin lesion analysis."
            ) from exc

        lesion_pixels = int(
            np.count_nonzero(
                segmentation_mask
                ==
                1
            )
        )

        total_pixels = int(
            segmentation_mask.size
        )

        lesion_fraction = (
            lesion_pixels
            /
            total_pixels
            if total_pixels > 0
            else 0.0
        )

        lesion_percentage = (
            lesion_fraction
            *
            100.0
        )

        return SkinAnalysisResult(

            lesion_detected=(
                lesion_pixels
                >
                0
            ),

            lesion_pixels=
                lesion_pixels,

            lesion_fraction=
                lesion_fraction,

            lesion_percentage=
                lesion_percentage,

            original_shape=(
                original_height,
                original_width,
            ),

            threshold=
                LESION_THRESHOLD,

            model_name=
                self.model_name,

            model_version=
                self.model_version,

            status=
                "research_model_prediction",

            segmentation_mask=
                segmentation_mask,
        )