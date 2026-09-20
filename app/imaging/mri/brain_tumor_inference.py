"""
MSD Brain Tumor MRI Inference Service

Medical Segmentation Decathlon
Task01 Brain Tumour

Research use only.
Does not provide clinical diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.models.mri.brain_tumor_unet import BrainTumor3DUNet
from app.imaging.mri.preprocessing import preprocess_mri_volume


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT
    / "checkpoints"
    / "msd_brain_tumor"
    / "msd_brain_tumor_best.pt"
)


class BrainTumorModelUnavailable(Exception):
    pass


class BrainTumorInferenceError(Exception):
    pass


@dataclass
class BrainTumorPrediction:

    tumor_detected: bool
    tumor_volume_voxels: int
    class_distribution: dict[str, int]
    segmentation_mask: np.ndarray
    model_name: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:

        return {
            "tumor_detected": self.tumor_detected,
            "tumor_volume_voxels": self.tumor_volume_voxels,
            "class_distribution": self.class_distribution,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


class BrainTumorModelService:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    ):

        self.checkpoint_path = Path(checkpoint_path)

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = None

        self.model_name = (
            "MSD Brain Tumor MRI Segmentation"
        )

        self.model_version = (
            "research-baseline-v1"
        )


    @property
    def checkpoint_available(self):

        return self.checkpoint_path.is_file()


    def load_model(self):

        if not self.checkpoint_available:

            raise BrainTumorModelUnavailable(
                f"Checkpoint missing: {self.checkpoint_path}"
            )

        try:

            model = BrainTumor3DUNet(
                in_channels=4,
                out_channels=4,
            )

            state = torch.load(
                self.checkpoint_path,
                map_location="cpu",
            )

            model.load_state_dict(
                state
            )

            model.to(
                self.device
            )

            model.eval()

            self.model = model


        except Exception as exc:

            raise BrainTumorModelUnavailable(
                str(exc)
            ) from exc


    def _prepare_input(
        self,
        volume: np.ndarray,
    ):

        processed = preprocess_mri_volume(
            volume
        )

        volume = processed.volume


        if (
            volume.ndim == 4
            and volume.shape[-1] == 4
        ):

            volume = np.moveaxis(
                volume,
                -1,
                0,
            )

        else:

            raise BrainTumorInferenceError(
                f"Unexpected MRI shape: {volume.shape}"
            )


        tensor = torch.tensor(
            volume,
            dtype=torch.float32,
        )


        tensor = tensor.unsqueeze(
            0
        )


        return tensor


    def predict(
        self,
        volume: np.ndarray,
    ) -> BrainTumorPrediction:


        if self.model is None:

            self.load_model()


        try:

            image = self._prepare_input(
                volume
            )


            image = image.to(
                self.device
            )


            with torch.no_grad():

                if self.device.type == "cuda":

                    with torch.cuda.amp.autocast():

                        output = self.model(
                            image
                        )

                else:

                    output = self.model(
                        image
                    )


            prediction = torch.argmax(
                output,
                dim=1,
            )


            mask = (
                prediction
                .squeeze(0)
                .cpu()
                .numpy()
            )


            labels, counts = np.unique(
                mask,
                return_counts=True,
            )


            distribution = {
                str(int(label)): int(count)
                for label, count in zip(
                    labels,
                    counts,
                )
            }


            tumor_voxels = int(
                (mask > 0).sum()
            )


            return BrainTumorPrediction(

                tumor_detected=(
                    tumor_voxels > 0
                ),

                tumor_volume_voxels=tumor_voxels,

                class_distribution=distribution,

                segmentation_mask=mask,

                model_name=self.model_name,

                model_version=self.model_version,
            )


        except BrainTumorModelUnavailable:

            raise


        except Exception as exc:

            raise BrainTumorInferenceError(
                str(exc)
            ) from exc