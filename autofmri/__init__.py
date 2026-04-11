"""AutoFMRI: Automatic two-stage thresholding for fMRI decoding."""

from autofmri.estimator import AutoEstimator
from autofmri.patchify import (
    filter_clusters_by_atlas,
    get_top_k_patches,
    load_atlas,
    patchify_by_atlas,
    patchify_by_cube,
)
from autofmri.models.factory import get_model
from autofmri.data import (
    HaxbyDataLoader,
    HCPDataLoader,
    EmotionDataLoader,
    GamblingDataLoader,
    LanguageDataLoader,
    VKMCDataLoader,
    GTZANDataLoader,
)
from autofmri.utils.seed import set_seed, get_tsnr

__version__ = "0.1.0"

__all__ = [
    "AutoEstimator",
    "patchify_by_atlas",
    "patchify_by_cube",
    "filter_clusters_by_atlas",
    "get_top_k_patches",
    "load_atlas",
    "get_model",
    "set_seed",
    "get_tsnr",
    "HaxbyDataLoader",
    "HCPDataLoader",
    "EmotionDataLoader",
    "GamblingDataLoader",
    "LanguageDataLoader",
    "VKMCDataLoader",
    "GTZANDataLoader",
]
