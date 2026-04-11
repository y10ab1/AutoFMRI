"""Atlas loading and brain patchification utilities."""

from __future__ import annotations

from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import get_data, new_img_like, resample_img


def load_atlas(
    atlas_name: str,
    reference_img=None,
    atlas_path: str | None = None,
) -> Tuple[np.ndarray, list]:
    """Load a brain atlas and return 4-D label data + region labels.

    Parameters
    ----------
    atlas_name : str
        One of ``"harvard_oxford"``, ``"juelich"``, ``"yeo"``,
        ``"yeo_thick17"``, ``"yeo400"``, ``"hcp_mmp_sym"``, or ``"custom"``.
    reference_img : Nifti1Image, optional
        If provided the atlas is resampled to this space.
    atlas_path : str, optional
        Path to a local NIfTI atlas file (required when *atlas_name* is
        ``"custom"``; also used to override built-in paths for ``"yeo400"``
        and ``"hcp_mmp_sym"``).

    Returns
    -------
    atlas_data : ndarray, shape (..., n_regions)
        4-D array where each volume is a binary region mask.
    labels : list
        Region labels (integers or strings).
    atlas_maps_img : Nifti1Image
        The atlas maps image (before 4-D expansion), used for resampling.
    """
    if atlas_name == "harvard_oxford":
        atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        atlas_data = _3d_to_4d(get_data(atlas.maps))
        maps_img = atlas.maps
        labels = list(np.unique(get_data(atlas.maps)))

    elif atlas_name == "juelich":
        atlas = datasets.fetch_atlas_juelich("prob-2mm")
        atlas_data = get_data(atlas.maps)
        maps_img = atlas.maps
        labels = list(np.unique(get_data(atlas.maps)))

    elif atlas_name in ("yeo", "yeo_thick17"):
        atlas = datasets.fetch_atlas_yeo_2011()
        maps_img_raw = atlas["thick_17"]
        raw_data = get_data(maps_img_raw)
        if raw_data.ndim == 4:
            raw_data = raw_data[..., 0]
        atlas_maps = []
        for label in np.unique(raw_data)[1:]:
            atlas_maps.append(raw_data == label)
        atlas_data = np.stack(atlas_maps, axis=-1)
        maps_img = maps_img_raw
        labels = list(np.unique(raw_data))

    elif atlas_name == "yeo400":
        path = atlas_path or "atlas/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii"
        maps_img = nib.load(path)
        atlas_data = _3d_to_4d(get_data(maps_img))
        labels = list(np.unique(get_data(maps_img)))

    elif atlas_name == "hcp_mmp_sym":
        path = atlas_path or "atlas/MMP_in_MNI_symmetrical_1.nii"
        maps_img = nib.load(path)
        atlas_data = _3d_to_4d(get_data(maps_img))
        labels = list(np.unique(get_data(maps_img)))

    elif atlas_name == "custom":
        if atlas_path is None:
            raise ValueError("atlas_path is required when atlas_name='custom'.")
        maps_img = nib.load(atlas_path)
        raw = get_data(maps_img)
        if raw.ndim == 4:
            atlas_data = raw
        else:
            atlas_data = _3d_to_4d(raw)
        labels = list(np.unique(raw))

    else:
        raise ValueError(
            f"Unknown atlas_name '{atlas_name}'. Supported: "
            "harvard_oxford, juelich, yeo, yeo400, hcp_mmp_sym, custom."
        )

    if reference_img is not None:
        atlas_resampled = resample_img(
            new_img_like(maps_img, atlas_data),
            target_affine=reference_img.affine,
            target_shape=reference_img.shape,
            interpolation="nearest",
        )
        atlas_data = get_data(atlas_resampled)

    return atlas_data, labels, maps_img


def _3d_to_4d(data: np.ndarray) -> np.ndarray:
    """Convert a 3-D label volume to a 4-D binary atlas."""
    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels != 0]
    return np.stack([data == label for label in unique_labels], axis=-1)


def patchify_by_atlas(
    X: np.ndarray,
    reference_img,
    atlas_name: str = "yeo",
    threshold: float = 0.1,
    atlas_path: str | None = None,
    verbose: bool = False,
) -> list:
    """Partition brain voxels into patches using an atlas.

    Parameters
    ----------
    X : ndarray, shape (n_samples, nx, ny, nz)
        4-D fMRI data.
    reference_img : Nifti1Image
        Reference image for resampling.
    atlas_name : str
        Atlas identifier.
    threshold : float
        Minimum fraction of non-zero voxels in a region to keep the patch.
    atlas_path : str, optional
        Path to a local atlas NIfTI file.
    verbose : bool

    Returns
    -------
    patch_masks : list of tuples
        Each element is ``(x_indices, y_indices, z_indices)``.
    """
    atlas_data, labels, _ = load_atlas(atlas_name, reference_img, atlas_path)

    print(f"Atlas '{atlas_name}': {len(labels)} labels, data shape {atlas_data.shape}")

    patch_masks = []
    ref_data = reference_img.get_fdata()

    for i in range(atlas_data.shape[-1]):
        mask = atlas_data[..., i].astype(bool)

        if verbose:
            region_label = labels[i + 1] if i + 1 < len(labels) else i
            print(f"Region {region_label}: {np.sum(mask)} voxels")

        if np.count_nonzero(ref_data[mask]) <= threshold * np.sum(mask):
            continue

        indices = np.where(mask)
        if len(indices[0]) == 0:
            continue
        patch_masks.append(indices)

    print(f"Number of candidate patches: {len(patch_masks)}")
    return patch_masks


def patchify_by_cube(
    X: np.ndarray,
    cube_size: Tuple[int, int, int] = (20, 20, 20),
    stride: Tuple[int, int, int] = (20, 20, 20),
) -> list:
    """Partition brain voxels into cubic patches.

    Parameters
    ----------
    X : ndarray, shape (n_samples, nx, ny, nz)
    cube_size : tuple of int
    stride : tuple of int

    Returns
    -------
    patch_masks : list of tuples
    """
    ref = X[0]
    patch_masks = []
    for x in range(0, ref.shape[0], stride[0]):
        for y in range(0, ref.shape[1], stride[1]):
            for z in range(0, ref.shape[2], stride[2]):
                cube = ref[
                    x : x + cube_size[0],
                    y : y + cube_size[1],
                    z : z + cube_size[2],
                ]
                if np.count_nonzero(cube) < 0.1 * np.prod(cube_size):
                    continue
                mask = np.zeros(ref.shape, dtype=int)
                mask[
                    x : x + cube_size[0],
                    y : y + cube_size[1],
                    z : z + cube_size[2],
                ] = 1
                patch_masks.append(np.where(mask == 1))

    print(f"Number of patches: {len(patch_masks)}")
    return patch_masks


def filter_clusters_by_atlas(
    X: np.ndarray,
    reference_img,
    atlas_name: str = "yeo",
    threshold: float = 0.05,
    atlas_path: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Zero out voxels that don't belong to active atlas regions."""
    atlas_data, labels, _ = load_atlas(atlas_name, reference_img, atlas_path)

    filtered_X = np.zeros_like(X)
    for i in range(atlas_data.shape[-1]):
        mask = atlas_data[..., i].astype(bool)
        if np.count_nonzero(X[..., 0][mask]) <= threshold * np.sum(mask):
            continue
        filtered_X[..., 0][mask] = X[..., 0][mask]

    return filtered_X


def get_top_k_patches(
    patch_scores: np.ndarray,
    patch_masks: list,
    X_train: np.ndarray,
    topk_patches: int,
    ref_niimg=None,
    output_filename: Optional[str] = None,
) -> Tuple[List, np.ndarray]:
    """Select the top-k scoring patches."""
    topk_patches = int(topk_patches)

    selected_idx = np.argsort(patch_scores)[-topk_patches:]
    selected_masks = np.array(patch_masks, dtype=object)[selected_idx].tolist()
    hp_mask = np.zeros(X_train[0].shape)

    if output_filename and ref_niimg is not None:
        _save_patches_3d(
            selected_masks, patch_scores, selected_idx, ref_niimg, hp_mask, output_filename
        )
    hp_mask = np.zeros(X_train[0].shape)

    return selected_masks, hp_mask


def _save_patches_3d(
    selected_masks, patch_scores, selected_idx, ref_niimg, hp_mask, output_filename
):
    for i, pm in enumerate(selected_masks):
        hp_mask[pm[0], pm[1], pm[2]] = patch_scores[selected_idx[i]]
    new_img_like(ref_niimg=ref_niimg, data=hp_mask).to_filename(output_filename)
