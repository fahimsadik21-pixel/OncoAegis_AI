"""
MRI Tumor Mask Export Utility

Research use only.
"""

from pathlib import Path

import numpy as np
import nibabel as nib



def save_mask_as_nifti(
    mask: np.ndarray,
    reference_path: str,
    output_path: str,
):

    reference = nib.load(
        reference_path
    )


    mask_image = nib.Nifti1Image(
        mask.astype(np.uint8),
        reference.affine,
        reference.header,
    )


    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    nib.save(
        mask_image,
        str(output),
    )


    return str(output)