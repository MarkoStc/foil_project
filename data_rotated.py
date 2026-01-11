from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from .data import TaskDataset
from .utils import seed_worker, set_global_seed


def rotation_angles_deg(num_tasks: int, angle_step_deg: float) -> List[float]:
    return [angle_step_deg * i for i in range(num_tasks)]


def make_rotate_transform(angle_deg: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Rotates an MNIST tensor image (C,H,W) by a fixed angle (degrees).
    Keeps output size 28x28 (expand=False).
    """
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        # x: Tensor [1,28,28]
        return TF.rotate(
            x,
            angle=angle_deg,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
        )
    return _rotate


def build_rotated_mnist_loaders(
    data_root: str,
    num_tasks: int,
    angle_step_deg: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[List[DataLoader], List[DataLoader], List[float]]:
    """
    Returns per-task train/test DataLoaders for RotatedMNIST.

    Tasks:
      angle_i = angle_step_deg * i  for i=0..num_tasks-1
    """
    set_global_seed(seed)

    base_transform = transforms.ToTensor()
    train_base = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=base_transform)
    test_base  = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=base_transform)

    angles = rotation_angles_deg(num_tasks=num_tasks, angle_step_deg=angle_step_deg)

    train_datasets = [TaskDataset(train_base, make_rotate_transform(a)) for a in angles]
    test_datasets  = [TaskDataset(test_base,  make_rotate_transform(a)) for a in angles]

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    def _worker_init_fn(worker_id: int) -> None:
        seed_worker(worker_id=worker_id, base_seed=seed)

    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []

    for i in range(num_tasks):
        train_loaders.append(
            DataLoader(
                train_datasets[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn if num_workers > 0 else None,
                generator=generator,
                persistent_workers=(num_workers > 0),
            )
        )
        test_loaders.append(
            DataLoader(
                test_datasets[i],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn if num_workers > 0 else None,
                generator=generator,
                persistent_workers=(num_workers > 0),
            )
        )

    return train_loaders, test_loaders, angles
