"""Example: Train AutoFMRI on the Haxby 2001 dataset.

Usage::

    python examples/train_haxby.py --data-dir data/haxby2001/subj4
"""

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

from autofmri import AutoEstimator, set_seed
from autofmri.data import HaxbyDataLoader
from autofmri.models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--atlas", default="yeo")
    parser.add_argument("--kfold", type=int, default=4)
    parser.add_argument("--model", default="rf")
    args = parser.parse_args()

    set_seed(42)

    loader = HaxbyDataLoader(data_dir=args.data_dir)
    X, y = loader.load_volume_data()
    print(f"Loaded: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    all_y, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf = AutoEstimator(
            stage1_model=get_model(args.model),
            stage2_model=get_model(args.model),
            stage1_model_type=args.model,
            stage2_model_type=args.model,
            kfold=args.kfold,
            atlas_name=args.atlas,
            ref_img=loader.reference_img,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        print(f"Fold {fold}: accuracy = {acc:.4f}")
        all_y.extend(y[test_idx])
        all_pred.extend(pred)

    print("\n--- Overall ---")
    print(classification_report(all_y, all_pred, zero_division=0))


if __name__ == "__main__":
    main()
