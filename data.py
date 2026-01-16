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


def generate_permutations(num_tasks: int, seed: int, save_path: str, input_dim: int = 28) -> List[torch.Tensor]:
    """
    Generates num_tasks random permutations of input_dim*input_dim indices with a fixed seed.
    Saves them to disk to ensure identical task definitions across methods.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(input_dim * input_dim, generator=g) for _ in range(num_tasks)]
    torch.save({"seed": seed, "num_tasks": num_tasks, "input_dim": input_dim, "permutations": perms}, save_path)
    return perms


def load_or_create_permutations(num_tasks: int, seed: int, save_path: str, input_dim: int = 28) -> List[torch.Tensor]:
    if os.path.exists(save_path):
        blob = torch.load(save_path, map_location="cpu")
        if blob.get("seed") == seed and blob.get("num_tasks") == num_tasks and blob.get("input_dim", 28) == input_dim:
            return blob["permutations"]
    return generate_permutations(num_tasks=num_tasks, seed=seed, save_path=save_path, input_dim=input_dim)


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


def make_permute_transform_generic(perm: torch.Tensor, input_dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    perm: LongTensor of shape [input_dim*input_dim]. Applies permutation to center
    input_dim x input_dim region of a 28x28 image, leaving the rest unchanged.
    """
    perm = perm.clone()
    offset = (28 - input_dim) // 2

    def _permute(x: torch.Tensor) -> torch.Tensor:
        # x: [1, 28, 28]
        result = x.clone()
        region = x[:, offset:offset + input_dim, offset:offset + input_dim]  # [1, input_dim, input_dim]
        flat = region.reshape(-1)               # [input_dim*input_dim]
        flat_p = flat[perm]                     # [input_dim*input_dim]
        result[:, offset:offset + input_dim, offset:offset + input_dim] = flat_p.view(1, input_dim, input_dim)
        return result
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

    perms_path = os.path.join(artifacts_dir, f"permutations_seed{seed}_numtasks{num_tasks}_dim28.pt")
    perms = load_or_create_permutations(num_tasks=num_tasks, seed=seed, save_path=perms_path, input_dim=28)

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


def build_permuted_mnist_loaders_plotC(
    data_root: str,
    artifacts_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[List[DataLoader], List[DataLoader], List[str]]:
    """
    Returns per-task train/test DataLoaders for 3 tasks with input dimensions 8, 8, 26.
    Each task permutes only the top-left input_dim x input_dim region of the 28x28 image.
    """
    set_global_seed(seed)

    input_dims = [8, 8, 26]
    num_tasks = len(input_dims)

    base_transform = transforms.ToTensor()
    train_base = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=base_transform)
    test_base = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=base_transform)

    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []
    perms_paths: List[str] = []

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    def _worker_init_fn(worker_id: int) -> None:
        seed_worker(worker_id=worker_id, base_seed=seed)

    for i, dim in enumerate(input_dims):
        perms_path = os.path.join(artifacts_dir, f"permutations_plotC_seed{seed}_task{i}_dim{dim}.pt")
        perms = load_or_create_permutations(num_tasks=1, seed=seed + i, save_path=perms_path, input_dim=dim)
        perms_paths.append(perms_path)

        train_ds = TaskDataset(train_base, make_permute_transform_generic(perms[0], dim))
        test_ds = TaskDataset(test_base, make_permute_transform_generic(perms[0], dim))

        train_loaders.append(
            DataLoader(
                train_ds,
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
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_worker_init_fn if num_workers > 0 else None,
                generator=generator,
                persistent_workers=(num_workers > 0),
            )
        )

    return train_loaders, test_loaders, perms_paths