from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from .utils import seed_worker, set_global_seed


class TaskDataset(Dataset):
    """
    Wraps a base dataset and applies an additional task-specific transform to x.
    Labels are unchanged.
    """
    def __init__(self, base_ds: Dataset, x_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.base_ds = base_ds
        self.x_transform = x_transform

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.base_ds[idx]  # x: Tensor [1,28,28], y: int
        if self.x_transform is not None:
            x = self.x_transform(x)
        return x, y


def generate_permutations(num_tasks: int, seed: int, save_path: str) -> List[torch.Tensor]:
    """
    Generates num_tasks random permutations of 784 indices with a fixed seed.
    Saves them to disk to ensure identical task definitions across methods.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(28 * 28, generator=g) for _ in range(num_tasks)]
    torch.save({"seed": seed, "num_tasks": num_tasks, "permutations": perms}, save_path)
    return perms


def load_or_create_permutations(num_tasks: int, seed: int, save_path: str) -> List[torch.Tensor]:
    if os.path.exists(save_path):
        blob = torch.load(save_path, map_location="cpu")
        if blob.get("seed") == seed and blob.get("num_tasks") == num_tasks:
            return blob["permutations"]
    return generate_permutations(num_tasks=num_tasks, seed=seed, save_path=save_path)


def make_permute_transform(perm: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    perm: LongTensor of shape [784]. Applies permutation to flattened pixels.
    """
    perm = perm.clone()

    def _permute(x: torch.Tensor) -> torch.Tensor:
        flat = x.view(-1)          # [784]
        flat_p = flat[perm]        # [784]
        return flat_p.view(1, 28, 28)
    return _permute


def build_permuted_mnist_loaders(
    data_root: str,
    artifacts_dir: str,
    num_tasks: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[List[DataLoader], List[DataLoader], str]:
    """
    Returns per-task train/test DataLoaders for PermutedMNIST (10 tasks).
    """
    set_global_seed(seed)

    base_transform = transforms.ToTensor()
    train_base = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=base_transform)
    test_base = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=base_transform)

    perms_path = os.path.join(artifacts_dir, f"permutations_seed{seed}_numtasks{num_tasks}.pt")
    perms = load_or_create_permutations(num_tasks=num_tasks, seed=seed, save_path=perms_path)

    train_datasets = [TaskDataset(train_base, make_permute_transform(perms[i])) for i in range(num_tasks)]
    test_datasets  = [TaskDataset(test_base,  make_permute_transform(perms[i])) for i in range(num_tasks)]

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

    return train_loaders, test_loaders, perms_path
