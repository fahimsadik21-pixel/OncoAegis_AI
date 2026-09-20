"""
MSD Brain Tumor Dataset Loader

Medical Segmentation Decathlon
Task01 Brain Tumour

Research use only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import nibabel as nib

from torch.utils.data import Dataset


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_DATASET_PATH = (
    _PROJECT_ROOT
    / "datasets"
    / "mri"
    / "msd_task01_brain_tumour"
    / "Task01_BrainTumour"
)


class MSDBrainDataset(Dataset):

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        transform=None,
        target_size: int = 128,
    ):

        self.dataset_path = Path(dataset_path)

        self.images_dir = (
            self.dataset_path / "imagesTr"
        )

        self.labels_dir = (
            self.dataset_path / "labelsTr"
        )

        self.transform = transform

        self.target_size = target_size


        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images folder missing: {self.images_dir}"
            )


        self.samples = sorted(
            [
                p
                for p in self.images_dir.glob("*.nii.gz")
                if not p.name.startswith("._")
            ]
        )


        if not self.samples:
            raise RuntimeError(
                "No MRI samples found"
            )


    def __len__(self):

        return len(self.samples)



    def _case_id(self, filename):

        if filename.endswith("_0000.nii.gz"):

            return filename.replace(
                "_0000.nii.gz",
                ""
            )

        return filename.replace(
            ".nii.gz",
            ""
        )



    def _normalize(self, volume):

        mean = volume.mean()

        std = volume.std()


        if std == 0:
            std = 1.0


        return (
            volume - mean
        ) / std



    def _resize_volume(
        self,
        volume
    ):

        tensor = torch.tensor(
            volume,
            dtype=torch.float32
        )


        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(
                self.target_size,
                self.target_size,
                self.target_size,
            ),
            mode="trilinear",
            align_corners=False,
        )


        return tensor.squeeze(0).numpy()



    def _resize_mask(
        self,
        mask
    ):

        tensor = torch.tensor(
            mask,
            dtype=torch.float32
        )


        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=(
                self.target_size,
                self.target_size,
                self.target_size,
            ),
            mode="nearest",
        )


        return tensor.squeeze().numpy().astype(
            np.int64
        )



    def __getitem__(
        self,
        index
    ) -> dict[str,Any]:


        image_path = self.samples[index]


        case_id = self._case_id(
            image_path.name
        )


        label_path = (
            self.labels_dir
            /
            f"{case_id}.nii.gz"
        )


        if not label_path.exists():

            raise FileNotFoundError(
                f"Mask missing: {label_path}"
            )


        image = nib.load(
            image_path
        ).get_fdata()


        mask = nib.load(
            label_path
        ).get_fdata()



        image = image.astype(
            np.float32
        )


        mask = mask.astype(
            np.int64
        )



        # MSD MRI format:
        # H,W,D,4

        if image.ndim != 4:

            raise ValueError(
                f"Invalid MRI shape: {image.shape}"
            )


        # Convert:
        # H,W,D,4
        #
        # to:
        # 4,H,W,D

        image = np.moveaxis(
            image,
            -1,
            0
        )


        for c in range(
            image.shape[0]
        ):

            image[c] = self._normalize(
                image[c]
            )


        image = self._resize_volume(
            image
        )


        mask = self._resize_mask(
            mask
        )


        image = torch.tensor(
            image,
            dtype=torch.float32
        )


        mask = torch.tensor(
            mask,
            dtype=torch.long
        )



        if self.transform:

            image, mask = self.transform(
                image,
                mask
            )


        return {

            "image": image,

            "mask": mask,

            "label": mask,

            "case": case_id,
        }



def inspect_msd_brain_dataset():

    dataset = MSDBrainDataset()


    return {

        "dataset":
            "MSD Brain Tumor",

        "samples":
            len(dataset),

        "path":
            str(dataset.dataset_path),

        "status":
            "ready",
    }