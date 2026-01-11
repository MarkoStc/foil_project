from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings (optional; may slightly reduce throughput)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_seen_tasks(
    model: nn.Module,
    test_loaders: List[DataLoader],
    upto_task_inclusive: int,
    device: torch.device,
) -> List[float]:
    return [evaluate_accuracy(model, test_loaders[k], device) for k in range(upto_task_inclusive + 1)]


def snapshot_params(model: nn.Module, device: torch.device) -> List[torch.Tensor]:
    return [p.detach().clone().to(device) for p in model.parameters()]


def init_state_dict_from_model(model: nn.Module) -> Dict[str, torch.Tensor]:
    # Store on CPU for portability
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_into_model(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)
