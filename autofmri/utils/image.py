"""Image processing helpers for NIfTI data."""

import numpy as np
from nilearn.image import get_data, new_img_like


def union_images(images):
    """Compute the voxel-wise union of a list of NIfTI images.

    Parameters
    ----------
    images : list of Nifti1Image
        Binary mask images to union.

    Returns
    -------
    Nifti1Image
        Union mask where a voxel is 1 if any input image has a non-zero value.
    """
    union = get_data(images[0]).astype(bool)
    for img in images[1:]:
        union = np.logical_or(union, get_data(img))
    return new_img_like(images[0], union.astype(np.float64))
