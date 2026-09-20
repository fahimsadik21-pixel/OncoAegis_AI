"""
Liver Locator Inference

Stage 1 of the liver CT pipeline.

Improvements:
- Uses liver-class probability instead of plain argmax
- Applies confidence threshold
- Keeps largest 3D connected component
- Produces a robust liver ROI

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from app.models.ct.liver_locator_unet import LiverLocator3DUNet
from app.imaging.ct.liver_preprocessing import normalize_ct


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT_PATH = (
    _PROJECT_ROOT
    / "checkpoints"
    / "liver_locator"
    / "liver_locator_best.pt"
)

TARGET_SIZE = (
    128,
    128,
    128,
)

ROI_MARGIN = (
    6,
    20,
    20,
)

DEFAULT_LIVER_THRESHOLD = 0.80


class LiverLocatorInferenceError(Exception):
    pass


@dataclass
class LiverLocatorPrediction:

    liver_mask: np.ndarray
    liver_probability: np.ndarray

    liver_voxels: int
    liver_detected: bool

    crop_bounds: tuple | None

    original_shape: tuple[int, int, int]
    processed_shape: tuple[int, int, int]

    threshold: float

    model_name: str
    model_version: str

    def to_dict(self):

        return {
            "liver_voxels":
                self.liver_voxels,

            "liver_detected":
                self.liver_detected,

            "crop_bounds":
                self.crop_bounds,

            "original_shape":
                self.original_shape,

            "processed_shape":
                self.processed_shape,

            "threshold":
                self.threshold,

            "model_name":
                self.model_name,

            "model_version":
                self.model_version,
        }


def _largest_connected_component(
    mask: np.ndarray,
):

    binary = (
        mask > 0
    )

    if not np.any(binary):

        return np.zeros_like(
            mask,
            dtype=np.uint8,
        )

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    labeled, component_count = ndimage.label(
        binary,
        structure=structure,
    )

    if component_count == 0:

        return np.zeros_like(
            mask,
            dtype=np.uint8,
        )

    component_sizes = np.bincount(
        labeled.ravel()
    )

    component_sizes[0] = 0

    largest_label = int(
        np.argmax(
            component_sizes
        )
    )

    cleaned = (
        labeled
        ==
        largest_label
    )

    cleaned = ndimage.binary_opening(
        cleaned,
        structure=structure,
        iterations=1,
    )

    cleaned = ndimage.binary_closing(
        cleaned,
        structure=structure,
        iterations=2,
    )

    cleaned = ndimage.binary_fill_holes(
        cleaned
    )

    return cleaned.astype(
        np.uint8
    )


def _robust_bounds(
    mask: np.ndarray,
):

    coords = np.argwhere(
        mask > 0
    )

    if coords.size == 0:

        return None

    # Ignore extreme tail voxels.
    # This prevents a few edge false positives from
    # expanding the ROI to the entire CT volume.

    lower = np.percentile(
        coords,
        1.0,
        axis=0,
    )

    upper = np.percentile(
        coords,
        99.0,
        axis=0,
    )

    lower = np.floor(
        lower
    ).astype(int)

    upper = np.ceil(
        upper
    ).astype(int)

    depth, height, width = mask.shape

    mz, my, mx = ROI_MARGIN

    z_min = max(
        0,
        int(lower[0]) - mz,
    )

    z_max = min(
        depth - 1,
        int(upper[0]) + mz,
    )

    y_min = max(
        0,
        int(lower[1]) - my,
    )

    y_max = min(
        height - 1,
        int(upper[1]) + my,
    )

    x_min = max(
        0,
        int(lower[2]) - mx,
    )

    x_max = min(
        width - 1,
        int(upper[2]) + mx,
    )

    return (
        (z_min, z_max),
        (y_min, y_max),
        (x_min, x_max),
    )


class LiverLocatorInferenceService:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str | None = None,
        liver_threshold: float = DEFAULT_LIVER_THRESHOLD,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():

            raise LiverLocatorInferenceError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        if not (
            0.0 < liver_threshold < 1.0
        ):

            raise LiverLocatorInferenceError(
                "liver_threshold must be between 0 and 1."
            )

        self.liver_threshold = float(
            liver_threshold
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

        self.model = LiverLocator3DUNet(
            in_channels=1,
            out_channels=2,
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

            raise LiverLocatorInferenceError(
                f"Expected 3D CT volume, got {volume.shape}"
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

    def _restore_probability(
        self,
        probability: torch.Tensor,
        original_shape,
    ):

        probability = probability.unsqueeze(
            0
        ).unsqueeze(
            0
        )

        probability = F.interpolate(
            probability,
            size=original_shape,
            mode="trilinear",
            align_corners=False,
        )

        return (
            probability
            .squeeze(0)
            .squeeze(0)
        )

    def predict(
        self,
        volume: np.ndarray,
    ) -> LiverLocatorPrediction:

        input_tensor, original_shape = (
            self._prepare_volume(
                volume
            )
        )

        input_tensor = input_tensor.to(
            self.device
        )

        use_amp = (
            self.device.type
            ==
            "cuda"
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

                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                    liver_probability = (
                        probabilities[
                            0,
                            1
                        ]
                    )

        except torch.cuda.OutOfMemoryError as exc:

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            raise LiverLocatorInferenceError(
                "CUDA out of memory during liver locator inference."
            ) from exc

        liver_probability = (
            self._restore_probability(
                liver_probability.cpu(),
                original_shape,
            )
        )

        probability_np = (
            liver_probability.numpy()
            .astype(np.float32)
        )

        threshold_mask = (
            probability_np
            >=
            self.liver_threshold
        ).astype(
            np.uint8
        )

        cleaned_mask = (
            _largest_connected_component(
                threshold_mask
            )
        )

        liver_voxels = int(
            np.count_nonzero(
                cleaned_mask
            )
        )

        liver_detected = (
            liver_voxels > 0
        )

        crop_bounds = _robust_bounds(
            cleaned_mask
        )

        return LiverLocatorPrediction(

            liver_mask=
                cleaned_mask,

            liver_probability=
                probability_np,

            liver_voxels=
                liver_voxels,

            liver_detected=
                liver_detected,

            crop_bounds=
                crop_bounds,

            original_shape=
                original_shape,

            processed_shape=
                TARGET_SIZE,

            threshold=
                self.liver_threshold,

            model_name=
                "IRCADb01 Liver Locator 3D U-Net",

            model_version=
                "research-baseline-v1",
        )


def load_liver_locator(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str | None = None,
    liver_threshold: float = DEFAULT_LIVER_THRESHOLD,
):

    return LiverLocatorInferenceService(
        checkpoint_path=checkpoint_path,
        device=device,
        liver_threshold=liver_threshold,
    )