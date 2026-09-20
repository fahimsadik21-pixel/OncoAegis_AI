"""
IRCADb01 Liver CT Dataset Loader

Research use only.

Labels:
    0 = background
    1 = liver
    2 = liver tumor
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET_PATH = (
    _PROJECT_ROOT
    / "datasets"
    / "universal_ct"
    / "ircadb01"
    / "3Dircadb1"
)


class IRCADLoaderError(Exception):
    pass


def _numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)

    if match:
        return int(match.group(1))

    return 0


def _case_number(path: Path) -> int:
    match = re.search(r"\.(\d+)$", path.name)

    if match:
        return int(match.group(1))

    return 0


class IRCADLiverDataset(Dataset):

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_DATASET_PATH,
    ):

        self.root_dir = Path(root_dir)

        if not self.root_dir.exists():
            raise IRCADLoaderError(
                f"Dataset not found: {self.root_dir}"
            )

        self.cases = sorted(
            [
                p
                for p in self.root_dir.iterdir()
                if p.is_dir()
                and p.name.startswith("3Dircadb1.")
            ],
            key=_case_number,
        )

        if not self.cases:
            raise IRCADLoaderError(
                "No IRCADb01 cases found."
            )

    def __len__(self):
        return len(self.cases)

    def _list_dicom_files(
        self,
        folder: Path,
    ):

        if not folder.exists():
            return []

        files = [
            p
            for p in folder.iterdir()
            if p.is_file()
        ]

        return sorted(
            files,
            key=_numeric_suffix,
        )

    def _load_ct_stack(
        self,
        folder: Path,
    ):

        files = self._list_dicom_files(
            folder
        )

        if not files:
            raise IRCADLoaderError(
                f"No CT DICOM slices found: {folder}"
            )

        slices = []

        for file in files:

            ds = pydicom.dcmread(
                str(file),
                force=True,
            )

            pixel = ds.pixel_array.astype(
                np.float32
            )

            slope = float(
                getattr(
                    ds,
                    "RescaleSlope",
                    1.0,
                )
            )

            intercept = float(
                getattr(
                    ds,
                    "RescaleIntercept",
                    0.0,
                )
            )

            pixel = (
                pixel * slope
                +
                intercept
            )

            slices.append(
                pixel
            )

        return np.stack(
            slices,
            axis=0,
        ).astype(
            np.float32
        )

    def _load_mask_stack(
        self,
        folder: Path,
        expected_depth: int,
        expected_shape: tuple[int, int],
    ):

        files = self._list_dicom_files(
            folder
        )

        if not files:
            return np.zeros(
                (
                    expected_depth,
                    expected_shape[0],
                    expected_shape[1],
                ),
                dtype=np.uint8,
            )

        file_map = {
            _numeric_suffix(file): file
            for file in files
        }

        mask_volume = np.zeros(
            (
                expected_depth,
                expected_shape[0],
                expected_shape[1],
            ),
            dtype=np.uint8,
        )

        for slice_index in range(
            expected_depth
        ):

            file = file_map.get(
                slice_index
            )

            if file is None:
                continue

            ds = pydicom.dcmread(
                str(file),
                force=True,
            )

            pixel = ds.pixel_array

            if pixel.shape != expected_shape:
                raise IRCADLoaderError(
                    f"Mask shape mismatch in {file}: "
                    f"{pixel.shape} != {expected_shape}"
                )

            mask_volume[
                slice_index
            ] = (
                pixel > 0
            ).astype(
                np.uint8
            )

        return mask_volume

    def _find_tumor_folders(
        self,
        masks_root: Path,
    ):

        tumor_folders = []

        if not masks_root.exists():
            return tumor_folders

        for folder in masks_root.iterdir():

            if not folder.is_dir():
                continue

            name = folder.name.lower()

            if (
                name.startswith("livertumor")
                or name == "tumor"
            ):
                tumor_folders.append(
                    folder
                )

        return sorted(
            tumor_folders,
            key=lambda p: p.name.lower(),
        )

    def _build_label(
        self,
        case: Path,
        image_shape: tuple[int, int, int],
    ):

        depth, height, width = image_shape

        masks_root = (
            case
            / "MASKS_DICOM"
            / "MASKS_DICOM"
        )

        liver_folder = (
            masks_root
            / "liver"
        )

        if not liver_folder.exists():
            raise IRCADLoaderError(
                f"Liver mask missing for {case.name}"
            )

        liver = self._load_mask_stack(
            liver_folder,
            expected_depth=depth,
            expected_shape=(
                height,
                width,
            ),
        )

        label = np.zeros(
            image_shape,
            dtype=np.uint8,
        )

        label[
            liver > 0
        ] = 1

        tumor_folders = self._find_tumor_folders(
            masks_root
        )

        for tumor_folder in tumor_folders:

            tumor = self._load_mask_stack(
                tumor_folder,
                expected_depth=depth,
                expected_shape=(
                    height,
                    width,
                ),
            )

            label[
                tumor > 0
            ] = 2

        return (
            label,
            [
                folder.name
                for folder in tumor_folders
            ],
        )

    def _normalize(
        self,
        volume: np.ndarray,
    ):

        mean = float(
            volume.mean()
        )

        std = float(
            volume.std()
        )

        if std > 0:
            volume = (
                volume - mean
            ) / std

        return volume.astype(
            np.float32
        )

    def __getitem__(
        self,
        index,
    ):

        case = self.cases[index]

        image_folder = (
            case
            / "PATIENT_DICOM"
            / "PATIENT_DICOM"
        )

        image = self._load_ct_stack(
            image_folder
        )

        label, tumor_folders = (
            self._build_label(
                case,
                image.shape,
            )
        )

        image = self._normalize(
            image
        )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                image
            )
        ).float().unsqueeze(
            0
        )

        label_tensor = torch.from_numpy(
            np.ascontiguousarray(
                label
            )
        ).long()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "case": case.name,
            "tumor_mask_folders": tumor_folders,
        }


def inspect_ircadb01_liver_dataset():

    dataset = IRCADLiverDataset()

    report = []

    for index in range(
        len(dataset)
    ):

        case = dataset.cases[
            index
        ]

        masks_root = (
            case
            / "MASKS_DICOM"
            / "MASKS_DICOM"
        )

        tumor_folders = (
            dataset._find_tumor_folders(
                masks_root
            )
        )

        report.append(
            {
                "case": case.name,
                "tumor_folders": [
                    folder.name
                    for folder
                    in tumor_folders
                ],
            }
        )

    return report


if __name__ == "__main__":

    dataset = IRCADLiverDataset()

    print(
        "Cases:",
        len(dataset)
    )

    for item in inspect_ircadb01_liver_dataset():
        print(
            item
        )