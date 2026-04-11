"""Model factory — create sklearn-compatible estimators by name."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


def get_model(model_name: str, *, n_jobs: int = -1, **kwargs):
    """Return a scikit-learn compatible estimator.

    Parameters
    ----------
    model_name : str
        One of ``"rf"``, ``"rfr"``, ``"svm"``, ``"svr"``, ``"xgb"``,
        ``"xgbr"``, ``"cnn"``.
    n_jobs : int
        Parallelism level for tree-based models.
    **kwargs
        Extra keyword arguments forwarded to CNN / XGBoost constructors.
        For ``"cnn"``: ``num_classes``, ``lr``, ``num_epochs``,
        ``batch_size``, ``verbose``.

    Returns
    -------
    estimator
        A scikit-learn compatible estimator instance.
    """
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=n_jobs)
    if model_name == "rfr":
        return RandomForestRegressor(n_estimators=100, n_jobs=n_jobs)
    if model_name == "svm":
        return SVC()
    if model_name == "svr":
        return SVR()

    if model_name in ("xgb", "xgbr"):
        return _make_xgboost(model_name, n_jobs=n_jobs)

    if model_name == "cnn":
        return _make_cnn(**kwargs)

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Supported: rf, rfr, svm, svr, xgb, xgbr, cnn."
    )


def _make_xgboost(name: str, n_jobs: int = -1):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for model='xgb'. "
            "Install with: pip install autofmri[xgboost]"
        ) from exc

    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=100, n_jobs=n_jobs)
    return xgb.XGBRegressor(n_estimators=100, n_jobs=n_jobs)


def _make_cnn(
    num_classes: int = 2,
    lr: float = 0.001,
    num_epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 0,
    **_extra,
):
    try:
        import torch
        from skorch import NeuralNetClassifier
        from skorch.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError(
            "PyTorch and skorch are required for model='cnn'. "
            "Install with: pip install autofmri[cnn]"
        ) from exc

    from autofmri.models.cnn import CNN

    return NeuralNetClassifier(
        module=CNN,
        module__task="classification",
        module__num_classes=num_classes,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=lr,
        max_epochs=num_epochs,
        batch_size=batch_size,
        train_split=None,
        verbose=verbose,
        callbacks=[EarlyStopping(patience=20, monitor="train_loss")],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
