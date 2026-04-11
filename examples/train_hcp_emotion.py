"""Example: Train AutoFMRI on HCP Emotion Recognition data.

Usage::

    python examples/train_hcp_emotion.py \
        --data-dir data/fmri-dataset-for-emotion-recognition \
        --subjects 1 2 3
"""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from autofmri import AutoEstimator, set_seed
from autofmri.data import EmotionDataLoader
from autofmri.models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--atlas", default="yeo400")
    parser.add_argument("--kfold", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)

    loader = EmotionDataLoader(data_dir=args.data_dir, subjects=args.subjects)
    X, y = loader.load_volume_data()
    print(f"Loaded: X={X.shape}, labels={np.unique(y, return_counts=True)}")

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf = AutoEstimator(
            stage1_model=get_model("rf"),
            stage2_model=get_model("rf"),
            stage1_model_type="rf",
            stage2_model_type="rf",
            kfold=args.kfold,
            atlas_name=args.atlas,
            ref_img=loader.reference_img,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        print(f"Fold {fold}: accuracy = {accuracy_score(y[test_idx], pred):.4f}")


if __name__ == "__main__":
    main()
