"""1-D CNN model for fMRI classification/regression."""

import warnings

import torch
from torch import nn


class CNN(nn.Module):
    """Lightweight 1-D CNN with GELU activations for voxel-level features."""

    def __init__(self, task: str = "classification", num_classes: int = 8):
        super().__init__()
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Lazy modules are a new feature"
        )

        self.conv = nn.Sequential(
            nn.LazyConv1d(out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.dense = nn.Sequential(
            nn.LazyLinear(2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, 500),
        )

        self.regression_head = nn.LazyLinear(1)
        self.classification_head = nn.LazyLinear(num_classes)
        self.out = self.regression_head if task != "classification" else self.classification_head

    def forward(self, x):
        if x.dtype != torch.float:
            x = x.float()
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dense(x)
        x = self.out(x.reshape(x.shape[0], -1))
        return x.squeeze(1)
