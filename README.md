# AutoFMRI

**Automatic Two-Stage Thresholding for fMRI Decoding**

[![PyPI version](https://img.shields.io/pypi/v/autofmri)](https://pypi.org/project/autofmri/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AutoFMRI addresses the dimensionality challenge in fMRI decoding — where voxel counts vastly exceed trial counts — by introducing an automatic two-stage voxel selection framework:

1. **Stage 1 (Atlas-based Patch Scoring + SHAP):** Partition the brain into atlas-defined regions, score each region via cross-validated classification, then compute SHAP values to identify the most informative voxels within each top-performing region.
2. **Stage 2 (Refined Classification):** Merge the top-k SHAP-selected voxels across the best patches and train a final classifier. Hyperparameters are automatically tuned via `RandomizedSearchCV`.

## Installation

```bash
pip install autofmri
```

Or install from source:

```bash
git clone https://github.com/y10ab1/AutoFMRI.git
cd AutoFMRI
pip install -e .
```

### Optional dependencies

```bash
# CNN support (PyTorch + skorch)
pip install autofmri[cnn]

# XGBoost support
pip install autofmri[xgboost]

# Visualization (matplotlib + seaborn)
pip install autofmri[viz]

# Everything
pip install autofmri[all]
```

## Quick Start

```python
from autofmri import AutoEstimator, set_seed
from autofmri.data import HaxbyDataLoader
from autofmri.models import get_model
from sklearn.model_selection import StratifiedKFold

set_seed(42)

# Load data
loader = HaxbyDataLoader(data_dir="data/haxby2001/subj4")
X, y = loader.load_volume_data()

# Train with two-stage AutoFMRI
clf = AutoEstimator(
    stage1_model=get_model("rf"),
    stage2_model=get_model("rf"),
    stage1_model_type="rf",
    stage2_model_type="rf",
    atlas_name="yeo",
    ref_img=loader.reference_img,
)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
train_idx, test_idx = next(iter(skf.split(X, y)))

clf.fit(X[train_idx], y[train_idx])
accuracy = clf.score(X[test_idx], y[test_idx])
print(f"Accuracy: {accuracy:.4f}")
```

## CLI Usage

AutoFMRI provides a unified command-line interface:

```bash
# Two-stage AutoFMRI method
autofmri train --dataset haxby --data-dir ./data/haxby2001/subj4 \
    --method autofmri --stage1-model rf --stage2-model rf --atlas yeo

# Baseline: ROI + ANOVA feature selection
autofmri train --dataset hcp_emotion \
    --data-dir ./data/emotion --method baseline_anova --subjects 1 2 3

# Baseline: ROI + RFE
autofmri train --dataset hcp_gambling \
    --data-dir ./data/gambling --method baseline_rfe
```

### Available options

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset` | `haxby`, `hcp_emotion`, `hcp_gambling`, `hcp_language`, `vkmc`, `gtzan` | (required) |
| `--data-dir` | Path to dataset root | (required) |
| `--method` | `autofmri`, `baseline`, `baseline_anova`, `baseline_rfe` | `autofmri` |
| `--stage1-model` | Model for stage 1: `rf`, `svm`, `xgb`, `cnn` | `rf` |
| `--stage2-model` | Model for stage 2 | `rf` |
| `--atlas` | Atlas for patchification: `yeo`, `yeo400`, `harvard_oxford`, `hcp_mmp_sym` | `yeo` |
| `--kfold` | Number of CV folds | `4` |
| `--n-jobs` | Parallel jobs | `-1` |
| `--result-dir` | Output directory | `results/<timestamp>` |

## Supported Datasets

| Dataset | Loader | Task |
|---------|--------|------|
| [Haxby 2001](https://doi.org/10.1126/science.1063736) | `HaxbyDataLoader` | Visual object recognition |
| HCP Emotion | `EmotionDataLoader` | Emotion recognition |
| HCP Gambling | `GamblingDataLoader` | Gambling task |
| HCP Language | `LanguageDataLoader` | Language processing |
| VKMC | `VKMCDataLoader` | Musical pitch decoding |
| GTZAN-fMRI | `GTZANDataLoader` | Music genre classification |

## Supported Models

| Name | Type | Dependency |
|------|------|-----------|
| `rf` | Random Forest Classifier | scikit-learn |
| `rfr` | Random Forest Regressor | scikit-learn |
| `svm` | Support Vector Classifier | scikit-learn |
| `svr` | Support Vector Regressor | scikit-learn |
| `xgb` | XGBoost Classifier | `pip install autofmri[xgboost]` |
| `xgbr` | XGBoost Regressor | `pip install autofmri[xgboost]` |
| `cnn` | 1D CNN (via skorch) | `pip install autofmri[cnn]` |

## Project Structure

```
autofmri/
├── __init__.py          # Public API
├── estimator.py         # AutoEstimator (core algorithm)
├── patchify.py          # Atlas loading & brain patchification
├── cli.py               # Unified CLI
├── models/
│   ├── factory.py       # get_model() factory
│   └── cnn.py           # 1D CNN architecture
├── data/
│   ├── base.py          # BaseFMRIDataLoader (ABC)
│   ├── haxby.py         # Haxby 2001
│   ├── hcp.py           # HCP tasks (Emotion/Gambling/Language)
│   ├── vkmc.py          # VKMC musical pitch
│   └── gtzan.py         # GTZAN music genre
└── utils/
    ├── shap.py          # SHAP value computation
    ├── image.py         # NIfTI image utilities
    └── seed.py          # Reproducibility & tSNR
```

## Citation

If you use AutoFMRI in your research, please cite:

```bibtex
@article{autofmri2024,
  title={AutoFMRI: An Automatic Two-Stage Thresholding for fMRI Decoding},
  author={y10ab1},
  year={2024},
  url={https://github.com/y10ab1/AutoFMRI}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
