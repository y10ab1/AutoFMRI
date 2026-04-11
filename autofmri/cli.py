"""Unified command-line interface for AutoFMRI training and evaluation.

Usage examples::

    # Two-stage AutoFMRI method on Haxby
    autofmri train --dataset haxby --data-dir ./data/haxby2001/subj4 \
        --method autofmri --stage1-model rf --stage2-model rf --atlas yeo

    # Baseline: ROI with ANOVA feature selection
    autofmri train --dataset hcp_emotion --data-dir ./data/emotion \
        --method baseline_anova --subjects 1 2 3

    # Baseline: ROI with RFE
    autofmri train --dataset hcp_gambling --data-dir ./data/gambling \
        --method baseline_rfe --subjects 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from autofmri.estimator import AutoEstimator
from autofmri.models.factory import get_model
from autofmri.utils.seed import set_seed


# ---------- Dataset registry ----------

def _load_dataset(args):
    """Return (X, y, reference_img) based on CLI args."""
    ds = args.dataset.lower()

    if ds == "haxby":
        from autofmri.data.haxby import HaxbyDataLoader
        loader = HaxbyDataLoader(
            data_dir=args.data_dir,
            subjects=args.subjects,
            mask_file=args.mask_file,
            mask_threshold=args.mask_threshold,
        )
        X, y = loader.load_volume_data()
        return X, y, loader.reference_img

    if ds.startswith("hcp_") or ds in ("emotion", "gambling", "language"):
        from autofmri.data.hcp import HCPDataLoader
        task = ds.replace("hcp_", "")
        loader = HCPDataLoader(
            data_dir=args.data_dir,
            subjects=args.subjects,
            task=task,
            mask_file=args.mask_file,
            mask_threshold=args.mask_threshold,
        )
        X, y = loader.load_volume_data()
        return X, y, loader.reference_img

    if ds == "vkmc":
        from autofmri.data.vkmc import VKMCDataLoader
        loader = VKMCDataLoader(
            data_dir=args.data_dir,
            subjects=args.subjects,
            mask_file=args.mask_file,
            mask_threshold=args.mask_threshold,
            augment=args.augment,
        )
        X, y = loader.load_volume_data()
        return X, y, loader.reference_img

    if ds == "gtzan":
        from autofmri.data.gtzan import GTZANDataLoader
        loader = GTZANDataLoader(
            data_dir=args.data_dir,
            subjects=args.subjects,
            mask_file=args.mask_file,
            mask_threshold=args.mask_threshold,
        )
        X, y = loader.load_volume_data()
        return X, y, loader.reference_img

    raise ValueError(
        f"Unknown dataset '{ds}'. Supported: haxby, hcp_emotion, "
        "hcp_gambling, hcp_language, vkmc, gtzan."
    )


# ---------- Training methods ----------

def _train_autofmri(X, y, ref_img, args):
    """Run the two-stage AutoFMRI pipeline with K-Fold CV."""
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)

    all_y, all_pred = [], []
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = AutoEstimator(
            stage1_model=get_model(args.stage1_model, n_jobs=args.n_jobs),
            stage2_model=get_model(args.stage2_model, n_jobs=args.n_jobs),
            stage1_model_type=args.stage1_model,
            stage2_model_type=args.stage2_model,
            kfold=args.kfold,
            atlas_name=args.atlas,
            n_jobs=args.n_jobs,
            max_patches_ratio=args.max_patches_ratio,
            ref_img=ref_img,
            result_dir=args.result_dir,
            random_search=not args.no_random_search,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if hasattr(clf, "img_high_performance_voxels_mask_"):
            out = os.path.join(
                args.result_dir,
                f"high_performance_voxels_mask_fold_{fold}.nii",
            )
            clf.img_high_performance_voxels_mask_.to_filename(out)

        all_y.extend(y_test)
        all_pred.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold}: accuracy = {acc:.4f}")
        results.append({"fold": fold, "accuracy": acc})

    _save_results(all_y, all_pred, results, args)


def _train_baseline(X, y, ref_img, args):
    """Run a baseline ROI pipeline with optional feature selection."""
    from sklearn.feature_selection import RFE, SelectPercentile, f_classif
    from sklearn.ensemble import RandomForestClassifier

    model = get_model(args.stage1_model, n_jobs=args.n_jobs)

    method = args.method.lower()
    if method == "baseline_anova":
        pipe = Pipeline([
            ("anova", SelectPercentile(f_classif, percentile=10)),
            ("clf", model),
        ])
    elif method == "baseline_rfe":
        rfe_estimator = RandomForestClassifier(n_estimators=50, n_jobs=args.n_jobs)
        pipe = Pipeline([
            ("rfe", RFE(rfe_estimator, n_features_to_select=0.1, step=0.1)),
            ("clf", model),
        ])
    else:
        pipe = model

    # Flatten spatial dims if needed
    if X.ndim == 4:
        n = X.shape[0]
        X = X.reshape(n, -1)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipe, X, y, cv=skf, scoring="accuracy", return_train_score=False
    )

    accs = cv_results["test_score"]
    print(f"Baseline ({method}): mean accuracy = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    results = [{"fold": i + 1, "accuracy": float(a)} for i, a in enumerate(accs)]
    _save_results_baseline(results, args)


def _save_results(all_y, all_pred, fold_results, args):
    os.makedirs(args.result_dir, exist_ok=True)

    report = classification_report(all_y, all_pred, output_dict=True, zero_division=0)
    print("\nOverall classification report:")
    print(classification_report(all_y, all_pred, zero_division=0))

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    fold_results.append({"fold": "mean", "accuracy": float(mean_acc)})

    with open(os.path.join(args.result_dir, "results.json"), "w") as f:
        json.dump({"folds": fold_results, "classification_report": report}, f, indent=2)

    pd.DataFrame(report).T.to_csv(
        os.path.join(args.result_dir, "classification_report.csv")
    )

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        cm = confusion_matrix(all_y, all_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=np.unique(all_y))
        disp.plot(cmap="Blues", values_format=".2f", xticks_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_dir, "confusion_matrix.png"))
        plt.close()
    except ImportError:
        pass

    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def _save_results_baseline(fold_results, args):
    os.makedirs(args.result_dir, exist_ok=True)
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    fold_results.append({"fold": "mean", "accuracy": float(mean_acc)})

    with open(os.path.join(args.result_dir, "results.json"), "w") as f:
        json.dump({"folds": fold_results}, f, indent=2)

    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


# ---------- Argument parser ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autofmri",
        description="AutoFMRI: Automatic two-stage thresholding for fMRI decoding",
    )
    sub = p.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Train and evaluate a model")

    train.add_argument(
        "--dataset",
        required=True,
        help="Dataset name: haxby, hcp_emotion, hcp_gambling, hcp_language, vkmc, gtzan",
    )
    train.add_argument("--data-dir", required=True, help="Path to dataset root")
    train.add_argument(
        "--method",
        default="autofmri",
        choices=["autofmri", "baseline", "baseline_anova", "baseline_rfe"],
        help="Training method",
    )
    train.add_argument("--subjects", nargs="*", default=None, help="Subject IDs")
    train.add_argument("--mask-file", default=None, help="Path to mask NIfTI")
    train.add_argument("--mask-threshold", type=float, default=None)

    train.add_argument("--stage1-model", default="rf", help="Stage 1 model name")
    train.add_argument("--stage2-model", default="rf", help="Stage 2 model name")
    train.add_argument(
        "--atlas",
        default="yeo",
        help="Atlas for patchification",
    )
    train.add_argument("--kfold", type=int, default=4)
    train.add_argument("--n-jobs", type=int, default=-1)
    train.add_argument("--max-patches-ratio", type=float, default=0.5)
    train.add_argument("--no-random-search", action="store_true")
    train.add_argument("--augment", type=int, default=0, help="Temporal augmentation (VKMC)")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--result-dir",
        default=None,
        help="Output directory (default: results/<timestamp>)",
    )

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    set_seed(args.seed)

    if args.result_dir is None:
        args.result_dir = os.path.join(
            "results", time.strftime("%Y-%m-%d-%H-%M-%S")
        )
    os.makedirs(args.result_dir, exist_ok=True)

    X, y, ref_img = _load_dataset(args)

    if args.method == "autofmri":
        _train_autofmri(X, y, ref_img, args)
    else:
        _train_baseline(X, y, ref_img, args)


if __name__ == "__main__":
    main()
