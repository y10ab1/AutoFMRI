"""Tests for model factory."""

import pytest


class TestGetModel:
    def test_rf(self):
        from autofmri.models.factory import get_model

        model = get_model("rf", n_jobs=1)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_svm(self):
        from autofmri.models.factory import get_model

        model = get_model("svm")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_rfr(self):
        from autofmri.models.factory import get_model

        model = get_model("rfr")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_svr(self):
        from autofmri.models.factory import get_model

        model = get_model("svr")
        assert hasattr(model, "fit")

    def test_unknown_raises(self):
        from autofmri.models.factory import get_model

        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent")

    def test_xgb_import_error(self):
        from autofmri.models.factory import get_model

        try:
            model = get_model("xgb")
            assert hasattr(model, "fit")
        except ImportError:
            pass  # xgboost not installed, expected

    def test_cnn_import_error(self):
        from autofmri.models.factory import get_model

        try:
            model = get_model("cnn", num_classes=2)
            assert hasattr(model, "fit")
        except ImportError:
            pass  # torch/skorch not installed, expected


class TestSeedUtils:
    def test_set_seed_reproducible(self):
        import numpy as np
        from autofmri.utils.seed import set_seed

        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_get_tsnr(self):
        import numpy as np
        from autofmri.utils.seed import get_tsnr

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10) + 5  # positive mean, some variance
        tsnr = get_tsnr(X)
        assert isinstance(tsnr, float)
        assert tsnr > 0
