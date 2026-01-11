from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import evaluate_seen_tasks, set_global_seed


@dataclass
class BaselineConfig:
    epochs_per_task: int = 30
    lr: float = 8.5e-3
    momentum: float = 0.9

    use_lr_decay: bool = False
    lr_decay_gamma: float = 0.9

    # Needed for Fig 2A dynamics
    track_epoch_dynamics: bool = False
    dynamics_tasks: int = 3

    seed: int = 1234


def run_baseline_sgd(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_tasks: int,
    cfg: BaselineConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seed(cfg.seed)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    acc_matrix = torch.full((num_tasks, num_tasks), float("nan"))
    avg_acc_curve: List[float] = []
    train_losses: List[List[float]] = []
    task_boundary_accs: List[List[float]] = []

    epoch_dynamics: Dict[str, object] = {
        "global_epoch_index": [],
        "task_switch_markers": [],
        "acc_by_task": {k: [] for k in range(min(cfg.dynamics_tasks, num_tasks))},
        "loss_over_time": [],
    }

    global_epoch = 0
    t0 = time.time()

    for task_id in range(num_tasks):
        if cfg.track_epoch_dynamics:
            epoch_dynamics["task_switch_markers"].append((global_epoch, task_id))

        tloss_task: List[float] = []

        for _ in range(cfg.epochs_per_task):
            model.train()
            running_loss, n = 0.0, 0

            for x, y in train_loaders[task_id]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                running_loss += loss.item() * bs
                n += bs

            loss_ep = running_loss / max(n, 1)
            tloss_task.append(loss_ep)

            if cfg.track_epoch_dynamics and task_id < cfg.dynamics_tasks:
                epoch_dynamics["global_epoch_index"].append(global_epoch)
                epoch_dynamics["loss_over_time"].append(loss_ep)
                for k in epoch_dynamics["acc_by_task"].keys():
                    epoch_dynamics["acc_by_task"][k].append(
                        _eval_one(model, test_loaders[k], device)
                    )

            global_epoch += 1

        train_losses.append(tloss_task)

        if cfg.use_lr_decay and task_id < num_tasks - 1:
            for pg in optimizer.param_groups:
                pg["lr"] *= cfg.lr_decay_gamma

        accs_seen = evaluate_seen_tasks(model, test_loaders, upto_task_inclusive=task_id, device=device)
        task_boundary_accs.append(accs_seen)

        for k, acc in enumerate(accs_seen):
            acc_matrix[task_id, k] = acc
        avg_acc_curve.append(float(np.mean(accs_seen)))

    _ = time.time() - t0
    return {
        "acc_matrix": acc_matrix,
        "avg_acc_curve": avg_acc_curve,
        "task_boundary_accs": task_boundary_accs,
        "train_losses": train_losses,
        "epoch_dynamics": epoch_dynamics,
        "cfg": cfg,
        "device": str(device),
    }


@torch.no_grad()
def _eval_one(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = torch.argmax(model(x), dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
