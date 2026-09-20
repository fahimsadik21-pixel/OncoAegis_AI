"""
MSD Task10 Colon CT Dataset Loader

Labels:
    0 = background
    1 = colon cancer primary lesion

Source:
    NIfTI (.nii.gz)

Internal tensor convention:
    image: [1, D, H, W]
    label: [D, H, W]

Research use only.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset


_PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


DEFAULT_DATASET_ROOT = (
    _PROJECT_ROOT
    / "datasets"
    / "universal_ct"
    / "msd_task10_colon"
    / "Task10_Colon"
)


class MSDColonLoaderError(Exception):
    pass


def _is_valid_nifti(
    path: Path,
) -> bool:

    name = path.name

    if name.startswith("."):
        return False

    if name.startswith("._"):
        return False

    return (
        name.endswith(".nii")
        or
        name.endswith(".nii.gz")
    )


def _case_number(
    path: Path,
) -> int:

    name = path.name

    if name.endswith(".nii.gz"):
        name = name[:-7]

    elif name.endswith(".nii"):
        name = name[:-4]

    try:
        return int(
            name.split("_")[-1]
        )

    except Exception:
        return 0


class MSDColonDataset(Dataset):

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_DATASET_ROOT,
    ):

        self.root_dir = Path(
            root_dir
        )

        self.images_dir = (
            self.root_dir
            / "imagesTr"
        )

        self.labels_dir = (
            self.root_dir
            / "labelsTr"
        )

        if not self.images_dir.exists():

            raise MSDColonLoaderError(
                f"imagesTr not found: "
                f"{self.images_dir}"
            )

        if not self.labels_dir.exists():

            raise MSDColonLoaderError(
                f"labelsTr not found: "
                f"{self.labels_dir}"
            )

        image_files = sorted(
            [
                path
                for path
                in self.images_dir.iterdir()

                if (
                    path.is_file()
                    and
                    _is_valid_nifti(path)
                )
            ],
            key=_case_number,
        )

        label_files = {
            path.name:
                path

            for path
            in self.labels_dir.iterdir()

            if (
                path.is_file()
                and
                _is_valid_nifti(path)
            )
        }

        self.samples = []

        missing_labels = []

        for image_path in image_files:

            label_path = label_files.get(
                image_path.name
            )

            if label_path is None:

                missing_labels.append(
                    image_path.name
                )

                continue

            self.samples.append(
                (
                    image_path,
                    label_path,
                )
            )

        if missing_labels:

            raise MSDColonLoaderError(
                "Missing labels for: "
                +
                ", ".join(
                    missing_labels[:10]
                )
            )

        if not self.samples:

            raise MSDColonLoaderError(
                "No valid colon training samples found."
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
            label_path,
        ) = self.samples[
            index
        ]

        image_nifti = nib.load(
            str(
                image_path
            )
        )

        label_nifti = nib.load(
            str(
                label_path
            )
        )

        image_xyz = (
            image_nifti.get_fdata(
                dtype=np.float32
            )
        )

        label_xyz = (
            label_nifti.get_fdata()
        )

        if image_xyz.ndim != 3:

            raise MSDColonLoaderError(
                f"Expected 3D image, "
                f"got {image_xyz.shape} "
                f"for {image_path.name}"
            )

        if label_xyz.ndim != 3:

            raise MSDColonLoaderError(
                f"Expected 3D label, "
                f"got {label_xyz.shape} "
                f"for {label_path.name}"
            )

        if image_xyz.shape != label_xyz.shape:

            raise MSDColonLoaderError(
                f"Image/label shape mismatch "
                f"for {image_path.name}: "
                f"{image_xyz.shape} "
                f"!= {label_xyz.shape}"
            )

        label_xyz = np.rint(
            label_xyz
        ).astype(
            np.uint8
        )

        unique_labels = set(
            int(x)
            for x
            in np.unique(
                label_xyz
            )
        )

        unexpected = (
            unique_labels
            -
            {0, 1}
        )

        if unexpected:

            raise MSDColonLoaderError(
                f"Unexpected labels "
                f"{sorted(unexpected)} "
                f"in {label_path.name}"
            )

        # NIfTI [X,Y,Z]
        # -> PyTorch [D,H,W] = [Z,Y,X]

        image_zyx = np.transpose(
            image_xyz,
            (
                2,
                1,
                0,
            ),
        )

        label_zyx = np.transpose(
            label_xyz,
            (
                2,
                1,
                0,
            ),
        )

        image_zyx = np.ascontiguousarray(
            image_zyx,
            dtype=np.float32,
        )

        label_zyx = np.ascontiguousarray(
            label_zyx,
            dtype=np.uint8,
        )

        spacing_xyz = tuple(
            float(x)
            for x
            in image_nifti.header.get_zooms()[:3]
        )

        spacing_zyx = (
            spacing_xyz[2],
            spacing_xyz[1],
            spacing_xyz[0],
        )

        case_name = image_path.name

        if case_name.endswith(".nii.gz"):
            case_name = case_name[:-7]

        elif case_name.endswith(".nii"):
            case_name = case_name[:-4]

        return {

            "image":
                torch.from_numpy(
                    image_zyx
                )
                .float()
                .unsqueeze(0),

            "label":
                torch.from_numpy(
                    label_zyx
                )
                .long(),

            "case":
                case_name,

            "image_path":
                str(
                    image_path
                ),

            "label_path":
                str(
                    label_path
                ),

            "original_shape_xyz":
                tuple(
                    int(x)
                    for x
                    in image_xyz.shape
                ),

            "tensor_shape_zyx":
                tuple(
                    int(x)
                    for x
                    in image_zyx.shape
                ),

            "spacing_xyz_mm":
                spacing_xyz,

            "spacing_zyx_mm":
                spacing_zyx,

            "affine":
                image_nifti.affine.copy(),
        }


def inspect_msd_colon_dataset():

    dataset = MSDColonDataset()

    return {

        "training_cases":
            len(
                dataset
            ),

        "first_case":
            dataset.samples[
                0
            ][0].name,

        "last_case":
            dataset.samples[
                -1
            ][0].name,

        "dataset_root":
            str(
                dataset.root_dir
            ),
    }


if __name__ == "__main__":

    dataset = MSDColonDataset()

    print(
        "Training cases:",
        len(
            dataset
        )
    )

    sample = dataset[
        0
    ]

    print(
        "Case:",
        sample["case"]
    )

    print(
        "Image:",
        sample["image"].shape
    )

    print(
        "Label:",
        sample["label"].shape
    )

    print(
        "Labels:",
        sample["label"].unique()
    )

    print(
        "Spacing ZYX:",
        sample["spacing_zyx_mm"]
    )