"""
MRI preprocessing utilities for MSD Brain Tumor dataset.

Research use only.
Prepares NIfTI MRI volumes for 3D segmentation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class MRIProcessingError(Exception):
    pass


@dataclass
class MRIProcessedVolume:
    volume: np.ndarray
    original_shape: tuple[int, ...]
    processed_shape: tuple[int, ...]
    modality: str
    normalization: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_shape": self.original_shape,
            "processed_shape": self.processed_shape,
            "modality": self.modality,
            "normalization": self.normalization,
        }


def normalize_mri(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalization for MRI volume.
    """

    volume = volume.astype(np.float32)

    mean = np.mean(volume)
    std = np.std(volume)

    if std == 0:
        return volume

    normalized = (volume - mean) / std

    return normalized.astype(np.float32)



def add_channel_dimension(
    volume: np.ndarray,
) -> np.ndarray:
    """
    Convert:
    D,H,W

    into:

    C,D,H,W
    """

    if volume.ndim == 3:
        return np.expand_dims(volume, axis=0)

    return volume



def preprocess_mri_volume(
    volume: np.ndarray,
) -> MRIProcessedVolume:
    """
    Main MRI preprocessing pipeline.
    """

    try:

        original_shape = tuple(volume.shape)

        volume = normalize_mri(volume)

        volume = add_channel_dimension(volume)


        return MRIProcessedVolume(
            volume=volume,
            original_shape=original_shape,
            processed_shape=tuple(volume.shape),
            modality="MRI",
            normalization="z-score",
        )


    except Exception as exc:

        raise MRIProcessingError(
            str(exc)
        ) from exc



def load_nifti_volume(
    path: str | Path,
) -> np.ndarray:
    """
    Load NIfTI MRI file.
    """

    try:

        import nibabel as nib

        image = nib.load(str(path))

        volume = image.get_fdata()

        return np.asarray(volume)


    except Exception as exc:

        raise MRIProcessingError(
            f"Failed loading MRI volume: {exc}"
        ) from exc