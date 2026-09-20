"""
Combined Liver CT Analysis Service

Pipeline:
    1. Liver localization
    2. Liver ROI extraction
    3. Stage-2 tumor segmentation
    4. Combine Stage-1 liver mask + Stage-2 tumor mask
    5. Physical measurement

Final labels:
    0 = background
    1 = liver
    2 = tumor

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.models.ct.liver_tumor_unet import (
    LiverTumor3DUNet,
)

from app.imaging.ct.liver_locator_inference import (
    LiverLocatorInferenceService,
)

from app.imaging.ct.liver_preprocessing import (
    normalize_ct,
)

from app.imaging.ct.liver_measurement import (
    measure_liver_segmentation,
)


_PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


STAGE2_CHECKPOINT = (
    _PROJECT_ROOT
    / "checkpoints"
    / "liver_tumor_stage2"
    / "liver_tumor_stage2_best.pt"
)


TARGET_SIZE = (
    128,
    128,
    128,
)


LOCATOR_THRESHOLDS = (
    0.80,
    0.70,
    0.60,
)


class LiverAnalysisError(Exception):
    pass


@dataclass
class LiverAnalysisResult:

    segmentation_mask: np.ndarray

    liver_detected: bool
    tumor_detected: bool

    liver_voxels: int
    tumor_voxels: int

    locator_threshold: float | None
    locator_fallback_used: bool

    crop_bounds: tuple | None

    original_shape: tuple[int, int, int]

    measurement: dict | None

    model_name: str
    model_version: str

    def to_dict(self):

        return {
            "liver_detected":
                self.liver_detected,

            "tumor_detected":
                self.tumor_detected,

            "liver_voxels":
                self.liver_voxels,

            "tumor_voxels":
                self.tumor_voxels,

            "locator_threshold":
                self.locator_threshold,

            "locator_fallback_used":
                self.locator_fallback_used,

            "crop_bounds":
                self.crop_bounds,

            "original_shape":
                self.original_shape,

            "measurement":
                self.measurement,

            "model_name":
                self.model_name,

            "model_version":
                self.model_version,

            "status":
                "research_model_prediction",
        }


class LiverAnalysisService:

    def __init__(
        self,
        device: str | None = None,
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        if not STAGE2_CHECKPOINT.exists():

            raise LiverAnalysisError(
                f"Stage-2 checkpoint not found: "
                f"{STAGE2_CHECKPOINT}"
            )

        self.locator = (
            LiverLocatorInferenceService(
                device=str(
                    self.device
                ),
                liver_threshold=0.80,
            )
        )

        self.tumor_model = (
            LiverTumor3DUNet(
                in_channels=1,
                out_channels=3,
            )
        )

        state_dict = torch.load(
            STAGE2_CHECKPOINT,
            map_location=self.device,
        )

        self.tumor_model.load_state_dict(
            state_dict
        )

        self.tumor_model.to(
            self.device
        )

        self.tumor_model.eval()

    def _locate_liver(
        self,
        volume: np.ndarray,
    ):

        for attempt_index, threshold in enumerate(
            LOCATOR_THRESHOLDS
        ):

            self.locator.liver_threshold = (
                threshold
            )

            result = self.locator.predict(
                volume
            )

            if (
                result.liver_detected
                and
                result.crop_bounds is not None
            ):

                return (
                    result,
                    threshold,
                    attempt_index > 0,
                )

        return (
            None,
            None,
            True,
        )

    @staticmethod
    def _crop_volume(
        volume: np.ndarray,
        bounds,
    ):

        (
            (z_min, z_max),
            (y_min, y_max),
            (x_min, x_max),
        ) = bounds

        crop = volume[
            z_min:z_max + 1,
            y_min:y_max + 1,
            x_min:x_max + 1,
        ]

        if crop.size == 0:

            raise LiverAnalysisError(
                "Generated liver ROI is empty."
            )

        return crop

    def _prepare_stage2(
        self,
        crop: np.ndarray,
    ):

        crop = normalize_ct(
            crop
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                crop
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

        return tensor

    def _run_stage2(
        self,
        tensor: torch.Tensor,
    ):

        tensor = tensor.to(
            self.device
        )

        use_amp = (
            self.device.type
            ==
            "cuda"
        )

        with torch.no_grad():

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):

                logits = self.tumor_model(
                    tensor
                )

            prediction = torch.argmax(
                logits,
                dim=1,
            )[0]

        return prediction.cpu()

    @staticmethod
    def _restore_roi_mask(
        roi_prediction: torch.Tensor,
        crop_shape,
    ):

        mask = (
            roi_prediction
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        mask = F.interpolate(
            mask,
            size=crop_shape,
            mode="nearest",
        )

        return (
            mask
            .squeeze(0)
            .squeeze(0)
            .to(torch.uint8)
            .numpy()
        )

    @staticmethod
    def _paste_tumor_into_original(
        roi_prediction: np.ndarray,
        original_shape,
        bounds,
    ):

        full_tumor_mask = np.zeros(
            original_shape,
            dtype=np.uint8,
        )

        (
            (z_min, z_max),
            (y_min, y_max),
            (x_min, x_max),
        ) = bounds

        expected_shape = (
            z_max - z_min + 1,
            y_max - y_min + 1,
            x_max - x_min + 1,
        )

        if roi_prediction.shape != expected_shape:

            raise LiverAnalysisError(
                "ROI mask shape mismatch: "
                f"{roi_prediction.shape} "
                f"!= {expected_shape}"
            )

        tumor_only = (
            roi_prediction == 2
        ).astype(
            np.uint8
        )

        full_tumor_mask[
            z_min:z_max + 1,
            y_min:y_max + 1,
            x_min:x_max + 1,
        ] = tumor_only

        return full_tumor_mask

    def analyze(
        self,
        volume: np.ndarray,
        spacing_mm: tuple[
            float,
            float,
            float,
        ] | None = None,
    ) -> LiverAnalysisResult:

        if volume.ndim != 3:

            raise LiverAnalysisError(
                "Expected a 3D CT volume, "
                f"got {volume.shape}"
            )

        original_shape = tuple(
            int(x)
            for x in volume.shape
        )

        (
            locator_result,
            used_threshold,
            fallback_used,
        ) = self._locate_liver(
            volume
        )

        if locator_result is None:

            empty_mask = np.zeros(
                original_shape,
                dtype=np.uint8,
            )

            return LiverAnalysisResult(

                segmentation_mask=
                    empty_mask,

                liver_detected=
                    False,

                tumor_detected=
                    False,

                liver_voxels=
                    0,

                tumor_voxels=
                    0,

                locator_threshold=
                    None,

                locator_fallback_used=
                    True,

                crop_bounds=
                    None,

                original_shape=
                    original_shape,

                measurement=
                    None,

                model_name=
                    "IRCADb01 Two-Stage Liver Analysis",

                model_version=
                    "research-baseline-v1",
            )

        bounds = (
            locator_result.crop_bounds
        )

        crop = self._crop_volume(
            volume,
            bounds,
        )

        crop_shape = tuple(
            int(x)
            for x in crop.shape
        )

        stage2_input = (
            self._prepare_stage2(
                crop
            )
        )

        roi_prediction = (
            self._run_stage2(
                stage2_input
            )
        )

        roi_prediction = (
            self._restore_roi_mask(
                roi_prediction,
                crop_shape,
            )
        )

        full_tumor_mask = (
            self._paste_tumor_into_original(
                roi_prediction,
                original_shape,
                bounds,
            )
        )

        # Stage-1 owns the liver segmentation.
        full_mask = np.zeros(
            original_shape,
            dtype=np.uint8,
        )

        full_mask[
            locator_result.liver_mask > 0
        ] = 1

        # Stage-2 only contributes tumor predictions.
        #
        # Keep tumors inside the locator ROI.
        full_mask[
            full_tumor_mask > 0
        ] = 2

        liver_voxels = int(
            np.count_nonzero(
                full_mask == 1
            )
        )

        tumor_voxels = int(
            np.count_nonzero(
                full_mask == 2
            )
        )

        liver_detected = (
            liver_voxels
            +
            tumor_voxels
        ) > 0

        tumor_detected = (
            tumor_voxels > 0
        )

        measurement = None

        if spacing_mm is not None:

            measurement_result = (
                measure_liver_segmentation(
                    full_mask,
                    spacing_mm,
                )
            )

            measurement = (
                measurement_result
                .to_dict()
            )

        return LiverAnalysisResult(

            segmentation_mask=
                full_mask,

            liver_detected=
                liver_detected,

            tumor_detected=
                tumor_detected,

            liver_voxels=
                liver_voxels,

            tumor_voxels=
                tumor_voxels,

            locator_threshold=
                used_threshold,

            locator_fallback_used=
                fallback_used,

            crop_bounds=
                bounds,

            original_shape=
                original_shape,

            measurement=
                measurement,

            model_name=
                "IRCADb01 Two-Stage Liver Analysis",

            model_version=
                "research-baseline-v1",
        )


def load_liver_analysis_service(
    device: str | None = None,
):

    return LiverAnalysisService(
        device=device
    )