"""
MSD Task10 Colon CT analysis service.

Binary tumor segmentation:
    0 = background
    1 = colon cancer primary model region

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.imaging.ct.colon_preprocessing import (
    preprocess_colon_volume,
)

from app.models.ct.colon_tumor_unet import (
    ColonTumorUNet3D,
)


CHECKPOINT_PATH = Path(
    "checkpoints/msd_colon_tumor/colon_tumor_best.pt"
)

TUMOR_THRESHOLD = 0.50


class ColonAnalysisError(Exception):
    pass


@dataclass
class ColonAnalysisResult:

    tumor_detected: bool

    tumor_voxels: int

    original_shape: tuple[int, int, int]

    tumor_threshold: float

    model_name: str

    model_version: str

    status: str

    segmentation_mask: np.ndarray

    def to_dict(self):

        return {
            "tumor_detected":
                self.tumor_detected,

            "tumor_voxels":
                self.tumor_voxels,

            "original_shape":
                self.original_shape,

            "tumor_threshold":
                self.tumor_threshold,

            "model_name":
                self.model_name,

            "model_version":
                self.model_version,

            "status":
                self.status,
        }


class ColonAnalysisService:

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        device: str | torch.device | None = None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():

            raise ColonAnalysisError(
                f"Checkpoint not found: {self.checkpoint_path}"
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

        self.model = ColonTumorUNet3D()

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
            "MSD Task10 Colon Tumor 3D U-Net",
        )

        self.model_version = checkpoint.get(
            "model_version",
            "research-baseline",
        )

    def analyze(
        self,
        volume: np.ndarray,
    ) -> ColonAnalysisResult:

        if volume.ndim != 3:

            raise ColonAnalysisError(
                f"Expected 3D CT volume, got {volume.shape}"
            )

        original_shape = tuple(
            int(x)
            for x
            in volume.shape
        )

        processed = preprocess_colon_volume(
            volume,
            label=None,
        )

        image = (
            processed.image
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
                        image
                    )

                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                    tumor_probability = probabilities[
                        :,
                        1:2,
                    ]

            restored_probability = F.interpolate(
                tumor_probability.float(),
                size=original_shape,
                mode="trilinear",
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
                TUMOR_THRESHOLD
            ).astype(
                np.uint8
            )

        except torch.cuda.OutOfMemoryError as exc:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise ColonAnalysisError(
                "GPU memory was insufficient for colon CT analysis."
            ) from exc

        tumor_voxels = int(
            np.count_nonzero(
                segmentation_mask
                ==
                1
            )
        )

        return ColonAnalysisResult(

            tumor_detected=(
                tumor_voxels
                >
                0
            ),

            tumor_voxels=
                tumor_voxels,

            original_shape=
                original_shape,

            tumor_threshold=
                TUMOR_THRESHOLD,

            model_name=
                self.model_name,

            model_version=
                self.model_version,

            status=
                "research_model_prediction",

            segmentation_mask=
                segmentation_mask,
        )