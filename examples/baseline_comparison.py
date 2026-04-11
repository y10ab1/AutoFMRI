"""Example: Compare AutoFMRI with baseline methods on the same dataset.

Usage::

    python examples/baseline_comparison.py \
        --data-dir data/haxby2001/subj4 \
        --mask-file data/haxby2001/subj4/mask_vt.nii
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from autofmri import AutoEstimator, set_seed
from autofmri.data import HaxbyDataLoader
from autofmri.models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mask-file", default=None)
    parser.add_argument("--kfold", type=int, default=4)
    args = parser.parse_args()

    set_seed(42)

    loader = HaxbyDataLoader(
        data_dir=args.data_dir, mask_file=args.mask_file
    )
    X, y = loader.load_volume_data()
    print(f"Loaded: X={X.shape}, y classes={np.unique(y)}")

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    X_flat = X.reshape(X.shape[0], -1) if X.ndim > 2 else X

    # --- Baseline 1: Plain ROI ---
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    scores = cross_val_score(rf, X_flat, y, cv=skf, scoring="accuracy")
    print(f"Baseline (plain ROI):     {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # --- Baseline 2: ROI + ANOVA ---
    pipe_anova = Pipeline([
        ("anova", SelectPercentile(f_classif, percentile=10)),
        ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])
    scores = cross_val_score(pipe_anova, X_flat, y, cv=skf, scoring="accuracy")
    print(f"Baseline (ROI + ANOVA):   {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # --- Baseline 3: ROI + RFE ---
    pipe_rfe = Pipeline([
        ("rfe", RFE(
            RandomForestClassifier(n_estimators=50, n_jobs=-1),
            n_features_to_select=0.1,
            step=0.1,
        )),
        ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1)),
    ])
    scores = cross_val_score(pipe_rfe, X_flat, y, cv=skf, scoring="accuracy")
    print(f"Baseline (ROI + RFE):     {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # --- AutoFMRI ---
    if X.ndim == 4:
        accs = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            clf = AutoEstimator(
                stage1_model=get_model("rf"),
                stage2_model=get_model("rf"),
                stage1_model_type="rf",
                stage2_model_type="rf",
                kfold=args.kfold,
                atlas_name="yeo",
                ref_img=loader.reference_img,
            )
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            accs.append(np.mean(y[test_idx] == pred))
        print(f"AutoFMRI (two-stage):     {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    else:
        print("(AutoFMRI requires 4-D unmasked data, skipped)")


if __name__ == "__main__":
    main()
