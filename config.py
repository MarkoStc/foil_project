from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PathsConfig:
    data_root: str = "./data"
    artifacts_dir: str = "./artifacts"
    results_dir: str = "./results"


@dataclass(frozen=True)
class PermutedMNISTConfig:
    num_tasks: int = 10
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 1234

@dataclass(frozen=True)
class RotatedMNISTConfig:
    """
    RotatedMNIST task sequence:
      task i has rotation angle = angle_step_deg * (i-1)
      (e.g., 0, 10, 20, ..., 90 degrees for 10 tasks).
    """
    num_tasks: int = 10
    angle_step_deg: float = 10.0
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 1234


@dataclass(frozen=True)
class TrainingConfig:
    # Step 9 hyperparameters (as currently used in your code)
    epochs_per_task: int = 30
    lr: float = 8.5e-3
    momentum: float = 0.9

    # Reproducibility (training runs)
    seed: int = 1234

    # Regularization strengths (Step 9)
    lambda_l2: float = 1.0
    lambda_ewc: float = 100.0

    # Fisher estimation (Step 9)
    fisher_num_samples: int = 1_000
    fisher_seed: int = 1234
    fisher_num_workers: int = 0  # keep deterministic & simple


@dataclass(frozen=True)
class ModelConfig:
    # MLP 784-400-400-10, no BN for permuted
    input_dim: int = 28 * 28
    hidden_dim1: int = 400
    hidden_dim2: int = 400
    num_classes: int = 10
    use_batchnorm: bool = False
