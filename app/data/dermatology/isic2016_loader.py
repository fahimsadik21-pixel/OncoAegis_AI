"""
ISIC 2016 skin lesion segmentation dataset loader.

Image:
    ISIC_xxxxxxx.jpg

Mask:
    ISIC_xxxxxxx_Segmentation.png

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


_PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


DEFAULT_DATASET_ROOT = (
    _PROJECT_ROOT
    / "datasets"
    / "dermatology"
    / "isic2016"
)


class ISIC2016LoaderError(Exception):
    pass


class ISIC2016SegmentationDataset(Dataset):

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_DATASET_ROOT,
    ):

        self.root_dir = Path(
            root_dir
        )

        if not self.root_dir.exists():

            raise ISIC2016LoaderError(
                f"Dataset not found: {self.root_dir}"
            )

        image_files = sorted(
            [
                path
                for path
                in self.root_dir.glob("ISIC_*.jpg")
                if "_Segmentation" not in path.name
            ]
        )

        self.samples = []

        missing_masks = []

        for image_path in image_files:

            case_id = image_path.stem

            mask_path = (
                self.root_dir
                /
                f"{case_id}_Segmentation.png"
            )

            if not mask_path.exists():

                missing_masks.append(
                    case_id
                )

                continue

            self.samples.append(
                (
                    image_path,
                    mask_path,
                    case_id,
                )
            )

        if missing_masks:

            raise ISIC2016LoaderError(
                "Missing masks for: "
                +
                ", ".join(
                    missing_masks[:10]
                )
            )

        if not self.samples:

            raise ISIC2016LoaderError(
                "No valid ISIC image-mask pairs found."
            )

    def __len__(
        self,
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        (
            image_path,
            mask_path,
            case_id,
        ) = self.samples[
            index
        ]

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        mask = Image.open(
            mask_path
        ).convert(
            "L"
        )

        image_np = np.asarray(
            image,
            dtype=np.uint8,
        )

        mask_np = np.asarray(
            mask,
            dtype=np.uint8,
        )

        if image_np.shape[:2] != mask_np.shape:

            raise ISIC2016LoaderError(
                f"Shape mismatch for {case_id}: "
                f"{image_np.shape[:2]} != {mask_np.shape}"
            )

        mask_np = (
            mask_np > 127
        ).astype(
            np.uint8
        )

        image_tensor = (
            torch.from_numpy(
                image_np.copy()
            )
            .permute(
                2,
                0,
                1,
            )
            .float()
        )

        mask_tensor = (
            torch.from_numpy(
                mask_np.copy()
            )
            .long()
        )

        return {

            "image":
                image_tensor,

            "mask":
                mask_tensor,

            "case":
                case_id,

            "image_path":
                str(
                    image_path
                ),

            "mask_path":
                str(
                    mask_path
                ),

            "original_shape":
                tuple(
                    int(x)
                    for x
                    in image_np.shape
                ),
        }


def inspect_isic2016_dataset():

    dataset = (
        ISIC2016SegmentationDataset()
    )

    sample = dataset[
        0
    ]

    return {

        "cases":
            len(
                dataset
            ),

        "first_case":
            sample[
                "case"
            ],

        "first_image_shape":
            tuple(
                sample[
                    "image"
                ].shape
            ),

        "first_mask_shape":
            tuple(
                sample[
                    "mask"
                ].shape
            ),
    }


if __name__ == "__main__":

    print(
        inspect_isic2016_dataset()
    )