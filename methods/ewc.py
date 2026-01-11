from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ..utils import evaluate_seen_tasks, set_global_seed, snapshot_params


@dataclass
class EWCConfig:
    epochs_per_task: int = 30
    lr: float = 8.5e-3
    momentum: float = 0.9

    lambda_ewc: float = 100.0

    fisher_num_samples: int = 1_000
    fisher_seed: int = 1234
    fisher_num_workers: int = 0

    use_lr_decay: bool = False
    lr_decay_gamma: float = 0.9

    # Needed for Fig 2A dynamics
    track_epoch_dynamics: bool = False
    dynamics_tasks: int = 3

    seed: int = 1234


def compute_fisher_diagonal_per_sample(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_samples: int = 1_000,
    seed: int = 1234,
    num_workers: int = 0,
) -> List[torch.Tensor]:
    """
    Empirical diagonal Fisher approximation via per-sample gradients (batch_size=1):
      F ≈ E[(∂ log p(y|x,θ))^2]
    Implemented as average of squared gradients of CE loss over a random subset.
    """
    model.eval()

    n = len(dataset)
    num_samples = min(num_samples, n)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:num_samples].tolist()
    subset = Subset(dataset, idx)

    loader = DataLoader(
        subset,
        batch_size=1,              # per-sample gradients
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    fisher = [torch.zeros_like(p, device=device) for p in model.parameters()]
    counted = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()

        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                fisher[i] += (p.grad.detach() ** 2)

        counted += 1

    for i in range(len(fisher)):
        fisher[i] /= max(counted, 1)

    return fisher


def ewc_penalty_multicenter(
    model: nn.Module,
    fisher_hist: List[List[torch.Tensor]],
    theta_star_hist: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Multi-center EWC penalty:
      sum_{j} sum_p F_{j,p} * (theta_p - theta^*_{j,p})^2
    """
    penalty = 0.0
    params = list(model.parameters())
    for fisher_j, theta_j in zip(fisher_hist, theta_star_hist):
        for p, f, th in zip(params, fisher_j, theta_j):
            penalty = penalty + torch.sum(f * (p - th) ** 2)
    return penalty


def run_ewc_multicenter(
    model: nn.Module,
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    num_tasks: int,
    cfg: EWCConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Multi-center EWC continual learning:
      L = CE + (lambda/2) * sum_{j < t} sum_p F_{j,p} (theta_p - theta^*_{j,p})^2
    where each past task contributes its own Fisher and anchor parameters.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seed(cfg.seed)
    model = model.to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    fisher_hist: List[List[torch.Tensor]] = []
    theta_star_hist: List[List[torch.Tensor]] = []

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

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = ce(logits, y)

                if task_id > 0 and cfg.lambda_ewc > 0.0 and len(fisher_hist) > 0:
                    reg = ewc_penalty_multicenter(model, fisher_hist, theta_star_hist)
                    loss = loss + 0.5 * cfg.lambda_ewc * reg

                loss.backward()
                opt.step()

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

        # Store per-task center (theta*) and Fisher
        theta_star_t = snapshot_params(model, device=device)
        task_dataset = train_loaders[task_id].dataset
        fisher_t = compute_fisher_diagonal_per_sample(
            model=model,
            dataset=task_dataset,
            device=device,
            num_samples=cfg.fisher_num_samples,
            seed=cfg.fisher_seed + task_id,
            num_workers=cfg.fisher_num_workers,
        )

        fisher_hist.append([f.detach().clone() for f in fisher_t])
        theta_star_hist.append([th.detach().clone() for th in theta_star_t])

        if cfg.use_lr_decay and task_id < num_tasks - 1:
            for pg in opt.param_groups:
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
