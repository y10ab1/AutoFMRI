import numpy as np
import shap

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from copy import deepcopy

from nilearn.image import new_img_like
from scipy.stats import uniform

from autofmri.patchify import patchify_by_atlas


class AutoEstimator(BaseEstimator, ClassifierMixin):
    """Two-stage automatic voxel selection estimator for fMRI decoding.

    Stage 1: Partition the brain into atlas-based patches, score each patch
    via cross-validated accuracy, then compute SHAP values for the top patches.

    Stage 2: Select the top-k patches and top-k% SHAP voxels, merge them into
    a compact feature set, and train a final classifier. Hyperparameters
    (topk_patches, topk_percent_shap) are tuned via RandomizedSearchCV.
    """

    def __init__(
        self,
        stage1_model=None,
        stage2_model=None,
        stage1_model_type=None,
        stage2_model_type=None,
        kfold=4,
        atlas_name=None,
        topk_patches=0.2,
        topk_percent_shap=0.01,
        max_patches_ratio=0.5,
        n_jobs=1,
        result_dir="results",
        ref_img=None,
        random_search=True,
        sample_to_calculate_shap=1.0,
    ):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage1_model_type = stage1_model_type
        self.stage2_model_type = stage2_model_type
        self.kfold = kfold
        self.atlas_name = atlas_name
        self.topk_patches = topk_patches
        self.topk_percent_shap = topk_percent_shap
        self.n_jobs = n_jobs
        self.result_dir = result_dir
        self.max_patches_ratio = max_patches_ratio
        self.ref_img = ref_img
        self.random_search = random_search
        self.sample_to_calculate_shap = sample_to_calculate_shap

        if self.ref_img is None:
            raise ValueError("ref_img must be provided (a Nifti1Image reference).")

    def fit(self, X, y):
        from autofmri.utils.shap import calculate_shap_values

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)

        self.patch_masks_ = patchify_by_atlas(
            X, self.ref_img, atlas_name=self.atlas_name
        )

        patch_scores = [
            cross_val_score(
                estimator=deepcopy(self.stage1_model),
                X=X[:, pm[0], pm[1], pm[2]],
                y=y_enc,
                cv=self.kfold,
                n_jobs=self.n_jobs,
            ).mean()
            for pm in self.patch_masks_
        ]

        # break ties
        epsilon = 1e-10
        patch_scores = [s + i * epsilon for i, s in enumerate(patch_scores)]

        patch_scores, self.patch_masks_ = zip(
            *sorted(zip(patch_scores, self.patch_masks_), reverse=True)
        )

        self.patch_masks_ = self.patch_masks_[
            : int(self.max_patches_ratio * len(self.patch_masks_))
        ]

        patch_abs_shap_values = []
        for patch_mask in tqdm(
            self.patch_masks_, desc="Computing SHAP values per patch"
        ):
            X_patch = X[:, patch_mask[0], patch_mask[1], patch_mask[2]]
            model = deepcopy(self.stage1_model)
            model.fit(X_patch, y_enc)

            n_sample = int(self.sample_to_calculate_shap * len(X_patch))
            idx = np.random.choice(len(X_patch), size=n_sample, replace=False)
            shap_values = calculate_shap_values(
                model, X_patch[idx], self.stage1_model_type
            )
            patch_abs_shap_values.append(np.mean(np.abs(shap_values), axis=(0, 1)))

            del X_patch, model

        stage2_est = _Stage2Estimator(
            patch_masks=self.patch_masks_,
            patch_shap_values=patch_abs_shap_values,
            topk_patches=self.topk_patches,
            topk_percent_shap=self.topk_percent_shap,
            stage2_model=self.stage2_model,
        )

        if self.random_search:
            params = {
                "topk_patches": list(range(1, len(self.patch_masks_))),
                "topk_percent_shap": uniform(0.01, 0.5),
            }
            ra_search = RandomizedSearchCV(
                estimator=stage2_est,
                param_distributions=params,
                scoring=make_scorer(accuracy_score),
                cv=StratifiedKFold(n_splits=self.kfold, shuffle=True),
                n_jobs=self.n_jobs,
                n_iter=2 * len(self.patch_masks_) + 10,
                verbose=1,
            )
            ra_search.fit(X, y_enc)
            self.best_model_ = ra_search.best_estimator_
        else:
            stage2_est.fit(X, y_enc)
            self.best_model_ = stage2_est

        self.high_performance_voxels_mask_ = self.best_model_.high_performance_voxels_mask_

        mask_3d = np.zeros_like(self.ref_img.get_fdata())
        mask_3d[
            self.high_performance_voxels_mask_[0],
            self.high_performance_voxels_mask_[1],
            self.high_performance_voxels_mask_[2],
        ] = 1
        self.img_high_performance_voxels_mask_ = new_img_like(
            ref_niimg=self.ref_img, data=mask_3d
        )

        return self

    def predict(self, X):
        y_pred = self.best_model_.predict(X)
        return self.le_.inverse_transform(y_pred)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class _Stage2Estimator(BaseEstimator, ClassifierMixin):
    """Inner estimator that selects top SHAP voxels from top patches."""

    def __init__(
        self,
        patch_masks=None,
        patch_shap_values=None,
        topk_patches=1,
        topk_percent_shap=0.1,
        stage2_model=None,
    ):
        self.patch_masks = patch_masks
        self.patch_shap_values = patch_shap_values
        self.topk_patches = topk_patches
        self.topk_percent_shap = topk_percent_shap
        self.stage2_model = stage2_model

    def fit(self, X, y=None):
        k = (
            self.topk_patches
            if self.topk_patches >= 1
            else int(self.topk_patches * len(self.patch_masks))
        )

        x_idx, y_idx, z_idx = [], [], []
        for i, pm in enumerate(self.patch_masks[:k]):
            n_select = max(1, int(self.topk_percent_shap * len(self.patch_shap_values[i])))
            top_idx = np.argsort(-self.patch_shap_values[i])[:n_select]
            top_idx = np.unravel_index(top_idx, pm[0].shape)

            x_idx.append(pm[0][top_idx])
            y_idx.append(pm[1][top_idx])
            z_idx.append(pm[2][top_idx])

        x_idx = np.concatenate(x_idx)
        y_idx = np.concatenate(y_idx)
        z_idx = np.concatenate(z_idx)

        self.high_performance_voxels_mask_ = np.array([x_idx, y_idx, z_idx])

        model = deepcopy(self.stage2_model)
        model.fit(X[:, x_idx, y_idx, z_idx], y)
        self.model_ = model

        return self

    def predict(self, X):
        xi, yi, zi = self.high_performance_voxels_mask_
        return self.model_.predict(X[:, xi, yi, zi])
