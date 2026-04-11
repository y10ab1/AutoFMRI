"""Tests for patchify utilities with synthetic data."""

import numpy as np
import pytest


class TestPatchifyByCube:
    def test_basic_cubing(self):
        from autofmri.patchify import patchify_by_cube

        rng = np.random.RandomState(42)
        X = rng.randn(5, 40, 40, 40).astype(np.float32)
        # Make sure there's signal everywhere
        X[:, :20, :20, :20] += 10

        patches = patchify_by_cube(X, cube_size=(20, 20, 20), stride=(20, 20, 20))
        assert len(patches) > 0

        for p in patches:
            assert len(p) == 3  # (x_idx, y_idx, z_idx)
            assert p[0].shape == p[1].shape == p[2].shape

    def test_empty_brain_returns_no_patches(self):
        from autofmri.patchify import patchify_by_cube

        X = np.zeros((5, 20, 20, 20), dtype=np.float32)
        patches = patchify_by_cube(X, cube_size=(10, 10, 10), stride=(10, 10, 10))
        assert len(patches) == 0


class TestLoadAtlas:
    def test_3d_to_4d(self):
        from autofmri.patchify import _3d_to_4d

        data = np.array([[[0, 1, 2], [0, 1, 2]]])
        result = _3d_to_4d(data)
        assert result.shape[-1] == 2  # labels 1 and 2
        assert result[..., 0].sum() == 2  # two voxels with label 1
        assert result[..., 1].sum() == 2  # two voxels with label 2
