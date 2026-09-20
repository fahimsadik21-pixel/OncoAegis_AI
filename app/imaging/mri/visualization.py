"""
MRI Tumor Segmentation Visualization

Research use only.
Creates MRI slice overlay images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt



class MRIVisualizationError(Exception):
    pass



def create_mri_overlay(
    mri_path: str,
    mask_path: str,
    output_path: str,
    slice_index: int | None = None,
):

    try:

        mri_img = nib.load(
            mri_path
        )

        mask_img = nib.load(
            mask_path
        )


        volume = mri_img.get_fdata()

        mask = mask_img.get_fdata()



        if volume.ndim == 4:

            # Use first MRI modality
            volume = volume[:, :, :, 0]



        if slice_index is None:

            slice_index = volume.shape[2] // 2



        mri_slice = volume[:, :, slice_index]

        mask_slice = mask[:, :, slice_index]



        # normalize MRI display

        mri_slice = (
            mri_slice - np.min(mri_slice)
        ) / (
            np.max(mri_slice)
            -
            np.min(mri_slice)
            +
            1e-8
        )



        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )



        plt.figure(
            figsize=(8,8)
        )


        plt.imshow(
            mri_slice,
            cmap="gray"
        )


        tumor = np.ma.masked_where(
            mask_slice == 0,
            mask_slice
        )


        plt.imshow(
            tumor,
            cmap="autumn",
            alpha=0.45
        )


        plt.axis(
            "off"
        )


        plt.title(
            "MRI Brain Tumor Segmentation Overlay"
        )


        plt.savefig(
            output,
            bbox_inches="tight",
            dpi=300
        )


        plt.close()



        return str(output)



    except Exception as exc:

        raise MRIVisualizationError(
            str(exc)
        ) from exc