from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.imaging.ultrasound.tn3k_measurement import (
    measure_binary_mask,
)

from app.imaging.ultrasound.tn3k_preprocessing import (
    preprocess_image,
)

from app.models.ultrasound.tn3k_unet import (
    TN3KUNet,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "tn3k_segmentation"
    / "tn3k_unet_best.pt"
)


class TN3KAnalysisService:

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
        threshold: float = 0.5,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"TN3K checkpoint not found: "
                f"{self.checkpoint_path}"
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

        self.threshold = float(
            threshold
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.image_size = int(
            checkpoint.get(
                "image_size",
                256,
            )
        )

        self.fold = int(
            checkpoint.get(
                "fold",
                0,
            )
        )

        self.best_val_dice = float(
            checkpoint.get(
                "best_val_dice",
                0.0,
            )
        )

        self.model = TN3KUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()


    def analyze(
        self,
        image_path: str | Path,
    ) -> dict:

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Ultrasound image not found: "
                f"{image_path}"
            )

        original = Image.open(
            image_path
        ).convert("L")

        original_width, original_height = (
            original.size
        )

        tensor = preprocess_image(
            original,
            image_size=self.image_size,
        )

        tensor = tensor.unsqueeze(
            0
        ).to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            )

        probability_np = (
            probability[
                0,
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        model_mask = (
            probability_np
            >= self.threshold
        ).astype(
            np.uint8
        )

        # Restore predicted mask to original image dimensions.
        resized_mask = Image.fromarray(
            model_mask * 255
        ).resize(
            (
                original_width,
                original_height,
            ),
            Image.Resampling.NEAREST,
        )

        original_mask = (
            np.asarray(
                resized_mask
            ) > 0
        ).astype(
            np.uint8
        )

        measurements = (
            measure_binary_mask(
                original_mask
            )
        )

        positive_probabilities = (
            probability_np[
                model_mask > 0
            ]
        )

        if (
            positive_probabilities.size
            > 0
        ):
            mean_region_probability = float(
                positive_probabilities.mean()
            )
        else:
            mean_region_probability = 0.0

        return {
            "model": {
                "id": (
                    "tn3k_thyroid_nodule_segmentation"
                ),

                "name": (
                    "TN3K Thyroid Nodule "
                    "Ultrasound Segmentation"
                ),

                "model_type": "2D U-Net",

                "dataset": "TN3K",

                "checkpoint": str(
                    self.checkpoint_path
                ),

                "fold": self.fold,

                "validation_dice": (
                    self.best_val_dice
                ),

                "test_dice": 0.722753,

                "test_iou": 0.608823,

                "clinical_validation": False,
            },

            "input": {
                "filename": (
                    image_path.name
                ),

                "modality": "ULTRASOUND",

                "organ": "thyroid",

                "original_width_pixels": (
                    original_width
                ),

                "original_height_pixels": (
                    original_height
                ),

                "model_input_size": (
                    self.image_size
                ),
            },

            "segmentation": {
                "threshold": (
                    self.threshold
                ),

                "detected_region": (
                    measurements[
                        "detected_region"
                    ]
                ),

                "mean_probability_inside_predicted_region": (
                    mean_region_probability
                ),

                "mask": original_mask,
            },

            "measurements": (
                measurements
            ),

            "safety": {
                "research_only": True,

                "clinical_diagnosis": False,

                "thyroid_cancer_diagnosis": False,

                "malignancy_classification": False,

                "clinical_validation": False,

                "message": (
                    "Research-only thyroid nodule segmentation. "
                    "A predicted region does not establish thyroid "
                    "cancer or malignancy. Clinical interpretation "
                    "requires ultrasound assessment and, when "
                    "appropriate, cytology/pathology and other "
                    "clinical information."
                ),
            },
        }