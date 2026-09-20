"""
Two-stage pancreas CT analysis.

Stage 1:
    Full-volume pancreas localization.

Stage 2:
    ROI pancreas/cancer segmentation.

Labels:
    0 = background
    1 = pancreas
    2 = cancer

Research use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.models.ct.pancreas_tumor_unet import PancreasTumor3DUNet



BASE_DIR = Path(__file__).resolve().parents[3]


LOCATOR_CHECKPOINT = (
    BASE_DIR
    /
    "checkpoints"
    /
    "msd_pancreas_tumor"
    /
    "pancreas_tumor_best.pt"
)


STAGE2_CHECKPOINT = (
    BASE_DIR
    /
    "checkpoints"
    /
    "msd_pancreas_tumor_fast"
    /
    "pancreas_tumor_fast_best.pt"
)



LOCATOR_SIZE = (
    128,
    128,
    128,
)


STAGE2_SIZE = (
    96,
    96,
    96,
)


ROI_MARGIN = (
    8,
    24,
    24,
)


HU_MIN = -125.0
HU_MAX = 275.0


CANCER_THRESHOLD = 0.90


LOCATOR_FOREGROUND_THRESHOLD = 0.15




class PancreasAnalysisError(Exception):
    """
    Raised when pancreas analysis cannot complete.
    """

    pass





@dataclass
class PancreasAnalysisResult:


    segmentation_mask: np.ndarray


    pancreas_detected: bool


    cancer_detected: bool


    pancreas_voxels: int


    cancer_voxels: int


    crop_bounds: tuple


    original_shape: tuple[int, int, int]


    cancer_threshold: float


    model_name: str


    model_version: str


    status: str



    def to_dict(self):

        return {

            "pancreas_detected":
                self.pancreas_detected,


            "cancer_detected":
                self.cancer_detected,


            "pancreas_voxels":
                self.pancreas_voxels,


            "cancer_voxels":
                self.cancer_voxels,


            "crop_bounds":
                self.crop_bounds,


            "original_shape":
                self.original_shape,


            "cancer_threshold":
                self.cancer_threshold,


            "model_name":
                self.model_name,


            "model_version":
                self.model_version,


            "status":
                self.status,
        }





def normalize_ct(
    volume: np.ndarray,
) -> np.ndarray:


    volume = volume.astype(
        np.float32
    )


    volume = np.clip(
        volume,
        HU_MIN,
        HU_MAX,
    )


    volume = (
        volume - HU_MIN
    ) / (
        HU_MAX - HU_MIN
    )


    return volume.astype(
        np.float32
    )





def resize_image(
    volume: np.ndarray,
    size: tuple[int, int, int],
) -> torch.Tensor:


    tensor = torch.from_numpy(
        np.ascontiguousarray(
            volume
        )
    ).float()


    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )


    tensor = F.interpolate(
        tensor,
        size=size,
        mode="trilinear",
        align_corners=False,
    )


    return tensor





def restore_mask(
    mask: torch.Tensor,
    shape: tuple[int, int, int],
) -> np.ndarray:


    mask = (
        mask
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )


    mask = F.interpolate(
        mask,
        size=shape,
        mode="nearest",
    )


    return (
        mask
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )





def restore_probability(
    probability: torch.Tensor,
    shape: tuple[int, int, int],
) -> np.ndarray:


    probability = (
        probability
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )


    probability = F.interpolate(
        probability,
        size=shape,
        mode="trilinear",
        align_corners=False,
    )


    return (
        probability
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )





def find_bounds(
    mask: np.ndarray,
):


    coordinates = np.argwhere(
        mask > 0
    )


    if coordinates.size == 0:

        return None


    minimum = coordinates.min(
        axis=0
    )


    maximum = (
        coordinates.max(
            axis=0
        )
        +
        1
    )


    bounds = []


    for axis in range(3):


        start = max(
            0,
            int(
                minimum[axis]
                -
                ROI_MARGIN[axis]
            ),
        )


        end = min(
            mask.shape[axis],
            int(
                maximum[axis]
                +
                ROI_MARGIN[axis]
            ),
        )


        bounds.append(
            (
                start,
                end,
            )
        )


    return tuple(bounds)
class PancreasAnalysisService:


    def __init__(
        self,
        device: str | None = None,
    ):


        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else
                "cpu"
            )


        self.device = torch.device(
            device
        )



        if not LOCATOR_CHECKPOINT.exists():

            raise PancreasAnalysisError(
                f"Locator checkpoint missing: {LOCATOR_CHECKPOINT}"
            )



        if not STAGE2_CHECKPOINT.exists():

            raise PancreasAnalysisError(
                f"Stage-2 checkpoint missing: {STAGE2_CHECKPOINT}"
            )



        # ==========================
        # Stage 1 Locator Model
        # ==========================


        self.locator = (
            PancreasTumor3DUNet()
            .to(
                self.device
            )
        )


        locator_checkpoint = torch.load(
            LOCATOR_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )


        self.locator.load_state_dict(
            locator_checkpoint[
                "model_state_dict"
            ]
        )


        self.locator.eval()



        # ==========================
        # Stage 2 Cancer Model
        # ==========================


        self.stage2 = (
            PancreasTumor3DUNet()
            .to(
                self.device
            )
        )


        stage2_checkpoint = torch.load(
            STAGE2_CHECKPOINT,
            map_location=self.device,
            weights_only=False,
        )


        self.stage2.load_state_dict(
            stage2_checkpoint[
                "model_state_dict"
            ]
        )


        self.stage2.eval()



        self.stage2_epoch = (
            stage2_checkpoint.get(
                "epoch",
                "unknown",
            )
        )




    def _forward_model(
        self,
        model,
        tensor,
    ):


        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=(
                self.device.type
                ==
                "cuda"
            ),
        ):

            output = model(
                tensor
            )


        return output
    @torch.no_grad()
    def analyze(
        self,
        volume: np.ndarray,
    ) -> PancreasAnalysisResult:


        try:


            if volume.ndim != 3:

                raise PancreasAnalysisError(
                    f"Expected 3D CT volume, got {volume.shape}"
                )



            original_shape = tuple(
                int(x)
                for x in volume.shape
            )



            normalized = normalize_ct(
                volume
            )



            # ==========================================
            # STAGE 1
            # Pancreas localization
            # ==========================================


            locator_input = resize_image(
                normalized,
                LOCATOR_SIZE,
            ).to(
                self.device,
                non_blocking=True,
            )



            locator_logits = self._forward_model(
                self.locator,
                locator_input,
            )



            locator_probability = torch.softmax(
                locator_logits,
                dim=1,
            )



            # Class 1 + Class 2 = pancreas foreground

            foreground_probability = (
                locator_probability[0, 1]
                +
                locator_probability[0, 2]
            )



            foreground_probability = restore_probability(
                foreground_probability,
                original_shape,
            )



            locator_mask = (
                foreground_probability
                >
                LOCATOR_FOREGROUND_THRESHOLD
            ).astype(
                np.uint8
            )



            bounds = find_bounds(
                locator_mask
            )



            if bounds is None:


                raise PancreasAnalysisError(
                    "Pancreas locator produced no foreground region."
                )



            (
                (z0, z1),
                (y0, y1),
                (x0, x1),
            ) = bounds



            cropped = normalized[
                z0:z1,
                y0:y1,
                x0:x1
            ]



            crop_shape = tuple(
                int(x)
                for x in cropped.shape
            )



            # ==========================================
            # STAGE 2
            # Pancreas + Cancer segmentation
            # ==========================================


            stage2_input = resize_image(
                cropped,
                STAGE2_SIZE,
            ).to(
                self.device,
                non_blocking=True,
            )



            stage2_logits = self._forward_model(
                self.stage2,
                stage2_input,
            )



            stage2_probability = torch.softmax(
                stage2_logits,
                dim=1,
            )



            cancer_probability_small = (
                stage2_probability[0, 2]
            )



            cancer_probability = restore_probability(
                cancer_probability_small,
                crop_shape,
            )



            # ==========================================
            # Final mask
            # ==========================================


            final_mask = np.zeros(
                original_shape,
                dtype=np.uint8,
            )



            # pancreas region

            final_mask[
                locator_mask > 0
            ] = 1



            locator_crop = locator_mask[
                z0:z1,
                y0:y1,
                x0:x1
            ]



            cancer_region = (
                cancer_probability
                >=
                CANCER_THRESHOLD
            )



            cancer_region = (
                cancer_region
                &
                (
                    locator_crop > 0
                )
            )



            final_crop = final_mask[
                z0:z1,
                y0:y1,
                x0:x1
            ]



            final_crop[
                cancer_region
            ] = 2



            final_mask[
                z0:z1,
                y0:y1,
                x0:x1
            ] = final_crop



            pancreas_voxels = int(
                np.count_nonzero(
                    final_mask == 1
                )
            )



            cancer_voxels = int(
                np.count_nonzero(
                    final_mask == 2
                )
            )



            return PancreasAnalysisResult(


                segmentation_mask=final_mask,


                pancreas_detected=(
                    pancreas_voxels
                    +
                    cancer_voxels
                )
                >
                0,


                cancer_detected=(
                    cancer_voxels
                    >
                    0
                ),



                pancreas_voxels=pancreas_voxels,


                cancer_voxels=cancer_voxels,


                crop_bounds=bounds,


                original_shape=original_shape,


                cancer_threshold=CANCER_THRESHOLD,


                model_name=(
                    "MSD Task07 Two-Stage "
                    "Pancreas Analysis"
                ),


                model_version=(
                    "research-baseline-"
                    f"stage2-epoch-{self.stage2_epoch}"
                ),


                status=(
                    "research_model_prediction"
                ),
            )



        except PancreasAnalysisError:

            raise



        except RuntimeError as exc:


            if "out of memory" in str(exc).lower():


                if torch.cuda.is_available():

                    torch.cuda.empty_cache()



                raise PancreasAnalysisError(
                    "GPU memory insufficient during pancreas analysis."
                ) from exc



            raise PancreasAnalysisError(
                str(exc)
            ) from exc



        except Exception as exc:


            raise PancreasAnalysisError(
                str(exc)
            ) from exc