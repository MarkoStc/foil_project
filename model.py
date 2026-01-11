from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPConfig:
    input_dim: int = 28 * 28
    hidden_dim1: int = 400
    hidden_dim2: int = 400
    num_classes: int = 10
    use_batchnorm: bool = False


class MNISTMLP(nn.Module):
    """
    Fully-connected MLP for MNIST-style inputs (1x28x28).
    Architecture: 784 -> 400 -> 400 -> 10 with ReLU.
    Optionally BatchNorm after hidden layers (disabled for PermutedMNIST replication here).
    """
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim1)
        self.fc2 = nn.Linear(cfg.hidden_dim1, cfg.hidden_dim2)
        self.fc3 = nn.Linear(cfg.hidden_dim2, cfg.num_classes)

        if cfg.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(cfg.hidden_dim1)
            self.bn2 = nn.BatchNorm1d(cfg.hidden_dim2)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        logits = self.fc3(x)
        return logits
