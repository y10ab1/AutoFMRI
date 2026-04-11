"""Tests for AutoEstimator with synthetic data."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from sklearn.ensemble import RandomForestClassifier


class TestStage2Estimator:
    """Test the inner Stage2Estimator logic."""

    def test_fit_predict_basic(self):
        from autofmri.estimator import _Stage2Estimator

        rng = np.random.RandomState(42)
        n_samples, nx, ny, nz = 20, 5, 5, 5
        X = rng.randn(n_samples, nx, ny, nz).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)

        patch_masks = [
            (np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 1, 2])),
            (np.array([3, 4]), np.array([3, 4]), np.array([3, 4])),
        ]
        shap_vals = [
            rng.rand(3),
            rng.rand(2),
        ]

        est = _Stage2Estimator(
            patch_masks=patch_masks,
            patch_shap_values=shap_vals,
            topk_patches=2,
            topk_percent_shap=0.5,
            stage2_model=RandomForestClassifier(n_estimators=10, random_state=42),
        )

        est.fit(X, y)
        preds = est.predict(X)
        assert preds.shape == (n_samples,)
        assert hasattr(est, "high_performance_voxels_mask_")
        assert est.high_performance_voxels_mask_.shape[0] == 3

    def test_topk_percent_selects_at_least_one(self):
        from autofmri.estimator import _Stage2Estimator

        rng = np.random.RandomState(0)
        X = rng.randn(10, 4, 4, 4).astype(np.float32)
        y = np.array([0] * 5 + [1] * 5)

        patch_masks = [
            (np.array([0, 1]), np.array([0, 1]), np.array([0, 1])),
        ]
        shap_vals = [rng.rand(2)]

        est = _Stage2Estimator(
            patch_masks=patch_masks,
            patch_shap_values=shap_vals,
            topk_patches=1,
            topk_percent_shap=0.001,  # very small, should still select >= 1
            stage2_model=RandomForestClassifier(n_estimators=5, random_state=42),
        )
        est.fit(X, y)
        assert est.high_performance_voxels_mask_.shape[1] >= 1
