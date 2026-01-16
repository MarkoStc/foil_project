from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch

from .config import ModelConfig, PathsConfig, PermutedMNISTConfig, RotatedMNISTConfig, TrainingConfig
from .data import build_permuted_mnist_loaders, build_permuted_mnist_loaders_plotC
from .data_rotated import build_rotated_mnist_loaders
from .model import MLPConfig, MNISTMLP
from .utils import get_device, init_state_dict_from_model, load_state_dict_into_model, set_global_seed

from .methods.sgd import BaselineConfig, run_baseline_sgd
from .methods.l2 import NaiveL2Config, run_naive_l2
from .methods.ewc import EWCConfig, run_ewc_multicenter, compute_fisher_information


def _fresh_model_from_init(model_cfg: MLPConfig, init_state: Dict[str, torch.Tensor]) -> MNISTMLP:
    m = MNISTMLP(model_cfg)
    load_state_dict_into_model(m, init_state)
    return m


def run_full_permuted_experiment(
    compute_l2: bool,
    paths: PathsConfig = PathsConfig(),
    ds_cfg: PermutedMNISTConfig = PermutedMNISTConfig(),
    tr_cfg: TrainingConfig = TrainingConfig(),
    mdl_cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Runs the full 10-task PermutedMNIST experiment and returns a dict suitable for plotting Fig 2B.
    Also saves results (npz + json meta) into paths.results_dir.
    """
    if device is None:
        device = get_device()

    set_global_seed(ds_cfg.seed)

    train_loaders, test_loaders, perms_path = build_permuted_mnist_loaders(
        data_root=paths.data_root,
        artifacts_dir=paths.artifacts_dir,
        num_tasks=ds_cfg.num_tasks,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        seed=ds_cfg.seed,
    )

    model_cfg = MLPConfig(
        input_dim=mdl_cfg.input_dim,
        hidden_dim1=mdl_cfg.hidden_dim1,
        hidden_dim2=mdl_cfg.hidden_dim2,
        num_classes=mdl_cfg.num_classes,
        use_batchnorm=mdl_cfg.use_batchnorm,
    )

    # Shared initialization across methods
    set_global_seed(tr_cfg.seed)
    _m0 = MNISTMLP(model_cfg)
    init_state = init_state_dict_from_model(_m0)

    # Single-task baseline (Task 1 only)
    cfg_single = BaselineConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        use_lr_decay=False,
        track_epoch_dynamics=False,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_single = run_baseline_sgd(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders[:1],
        test_loaders=test_loaders[:1],
        num_tasks=1,
        cfg=cfg_single,
        device=device,
    )
    single_task_acc = float(out_single["acc_matrix"][0, 0])

    # SGD baseline (no reg)
    cfg_sgd = BaselineConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        use_lr_decay=False,
        track_epoch_dynamics=False,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_sgd = run_baseline_sgd(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=ds_cfg.num_tasks,
        cfg=cfg_sgd,
        device=device,
    )

    out_l2 = None
    if compute_l2:
        cfg_l2 = NaiveL2Config(
            epochs_per_task=tr_cfg.epochs_per_task,
            lr=tr_cfg.lr,
            momentum=tr_cfg.momentum,
            lambda_l2=tr_cfg.lambda_l2,
            use_lr_decay=False,
            track_epoch_dynamics=False,
            dynamics_tasks=3,
            seed=tr_cfg.seed,
        )
        out_l2 = run_naive_l2(
            model=_fresh_model_from_init(model_cfg, init_state),
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_tasks=ds_cfg.num_tasks,
            cfg=cfg_l2,
            device=device,
        )

    # EWC multi-center
    cfg_ewc = EWCConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        lambda_ewc=tr_cfg.lambda_ewc,
        fisher_num_samples=tr_cfg.fisher_num_samples,
        fisher_seed=tr_cfg.fisher_seed,
        fisher_num_workers=tr_cfg.fisher_num_workers,
        use_lr_decay=False,
        track_epoch_dynamics=False,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_ewc = run_ewc_multicenter(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=ds_cfg.num_tasks,
        cfg=cfg_ewc,
        device=device,
    )

    full_results: Dict[str, object] = {
        "scenario": "permuted",
        "num_tasks": ds_cfg.num_tasks,
        "permutations_path": perms_path,
        "single_task_acc": single_task_acc,
        "epochs_per_task": tr_cfg.epochs_per_task,
        "lr": tr_cfg.lr,
        "momentum": tr_cfg.momentum,
        "seed": tr_cfg.seed,
        "lambda_l2": tr_cfg.lambda_l2,
        "lambda_ewc": tr_cfg.lambda_ewc,
        "fisher_num_samples": tr_cfg.fisher_num_samples,
        "sgd": out_sgd,
        "ewc": out_ewc,
        "l2": out_l2,  # may be None
    }

    # Save
    os.makedirs(paths.results_dir, exist_ok=True)
    npz_path = os.path.join(paths.results_dir, "permuted_fig2_results.npz")
    meta_path = os.path.join(paths.results_dir, "permuted_fig2_meta.json")

    save_dict = {
        "single_task_acc": single_task_acc,
        "sgd_acc_matrix": out_sgd["acc_matrix"].cpu().numpy(),
        "sgd_avg_acc_curve": np.array(out_sgd["avg_acc_curve"], dtype=np.float32),
        "ewc_acc_matrix": out_ewc["acc_matrix"].cpu().numpy(),
        "ewc_avg_acc_curve": np.array(out_ewc["avg_acc_curve"], dtype=np.float32),
    }
    if out_l2 is not None:
        save_dict.update({
            "l2_acc_matrix": out_l2["acc_matrix"].cpu().numpy(),
            "l2_avg_acc_curve": np.array(out_l2["avg_acc_curve"], dtype=np.float32),
        })

    np.savez(npz_path, **save_dict)

    meta = {
        "paths": asdict(paths),
        "dataset": asdict(ds_cfg),
        "training": asdict(tr_cfg),
        "model": asdict(mdl_cfg),
        "device": str(device),
        "compute_l2": bool(compute_l2),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return full_results


def run_fig2a_dynamics_permuted_abc(
    compute_l2: bool,
    paths: PathsConfig = PathsConfig(),
    ds_cfg: PermutedMNISTConfig = PermutedMNISTConfig(),
    tr_cfg: TrainingConfig = TrainingConfig(),
    mdl_cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Runs the 3-task (A,B,C) dynamics experiment recording test accuracy every epoch,
    returning epoch_dynamics for each method needed for Fig 2A plotting.
    """
    if device is None:
        device = get_device()

    set_global_seed(ds_cfg.seed)

    train_loaders, test_loaders, _ = build_permuted_mnist_loaders(
        data_root=paths.data_root,
        artifacts_dir=paths.artifacts_dir,
        num_tasks=ds_cfg.num_tasks,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        seed=ds_cfg.seed,
    )

    # Only first 3 tasks
    train_loaders = train_loaders[:3]
    test_loaders = test_loaders[:3]

    model_cfg = MLPConfig(
        input_dim=mdl_cfg.input_dim,
        hidden_dim1=mdl_cfg.hidden_dim1,
        hidden_dim2=mdl_cfg.hidden_dim2,
        num_classes=mdl_cfg.num_classes,
        use_batchnorm=mdl_cfg.use_batchnorm,
    )

    set_global_seed(tr_cfg.seed)
    _m0 = MNISTMLP(model_cfg)
    init_state = init_state_dict_from_model(_m0)

    # SGD dynamics
    cfg_sgd = BaselineConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        use_lr_decay=False,
        track_epoch_dynamics=True,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_sgd = run_baseline_sgd(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=3,
        cfg=cfg_sgd,
        device=device,
    )

    out_l2 = None
    if compute_l2:
        cfg_l2 = NaiveL2Config(
            epochs_per_task=tr_cfg.epochs_per_task,
            lr=tr_cfg.lr,
            momentum=tr_cfg.momentum,
            lambda_l2=tr_cfg.lambda_l2,
            use_lr_decay=False,
            track_epoch_dynamics=True,
            dynamics_tasks=3,
            seed=tr_cfg.seed,
        )
        out_l2 = run_naive_l2(
            model=_fresh_model_from_init(model_cfg, init_state),
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_tasks=3,
            cfg=cfg_l2,
            device=device,
        )

    # EWC dynamics
    cfg_ewc = EWCConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        lambda_ewc=tr_cfg.lambda_ewc,
        fisher_num_samples=tr_cfg.fisher_num_samples,
        fisher_seed=tr_cfg.fisher_seed,
        fisher_num_workers=tr_cfg.fisher_num_workers,
        use_lr_decay=False,
        track_epoch_dynamics=True,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_ewc = run_ewc_multicenter(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=3,
        cfg=cfg_ewc,
        device=device,
    )

    return {
        "scenario": "permuted",
        "epochs_per_task": tr_cfg.epochs_per_task,
        "compute_l2": bool(compute_l2),
        "sgd_dyn": out_sgd["epoch_dynamics"],
        "l2_dyn": None if out_l2 is None else out_l2["epoch_dynamics"],
        "ewc_dyn": out_ewc["epoch_dynamics"],
    }


def run_full_rotated_experiment(
    compute_l2: bool,
    paths: PathsConfig = PathsConfig(),
    ds_cfg: RotatedMNISTConfig = RotatedMNISTConfig(),
    tr_cfg: TrainingConfig = TrainingConfig(),
    mdl_cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Runs the full 10-task RotatedMNIST experiment and returns a dict suitable for plotting Fig 2B.
    Saves results (npz + json meta) into paths.results_dir with rotated-specific filenames.
    """
    if device is None:
        device = get_device()

    set_global_seed(ds_cfg.seed)

    train_loaders, test_loaders, angles = build_rotated_mnist_loaders(
        data_root=paths.data_root,
        num_tasks=ds_cfg.num_tasks,
        angle_step_deg=ds_cfg.angle_step_deg,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        seed=ds_cfg.seed,
    )

    model_cfg = MLPConfig(
        input_dim=mdl_cfg.input_dim,
        hidden_dim1=mdl_cfg.hidden_dim1,
        hidden_dim2=mdl_cfg.hidden_dim2,
        num_classes=mdl_cfg.num_classes,
        use_batchnorm=mdl_cfg.use_batchnorm,
    )

    # Shared initialization across methods
    set_global_seed(tr_cfg.seed)
    _m0 = MNISTMLP(model_cfg)
    init_state = init_state_dict_from_model(_m0)

    # Single-task baseline (Task 1 only)
    cfg_single = BaselineConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        use_lr_decay=False,
        track_epoch_dynamics=False,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_single = run_baseline_sgd(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders[:1],
        test_loaders=test_loaders[:1],
        num_tasks=1,
        cfg=cfg_single,
        device=device,
    )
    single_task_acc = float(out_single["acc_matrix"][0, 0])

    # SGD baseline (no reg)
    cfg_sgd = BaselineConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        use_lr_decay=False,
        track_epoch_dynamics=False,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_sgd = run_baseline_sgd(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=ds_cfg.num_tasks,
        cfg=cfg_sgd,
        device=device,
    )

    out_l2 = None
    if compute_l2:
        cfg_l2 = NaiveL2Config(
            epochs_per_task=tr_cfg.epochs_per_task,
            lr=tr_cfg.lr,
            momentum=tr_cfg.momentum,
            lambda_l2=tr_cfg.lambda_l2,
            use_lr_decay=False,
            track_epoch_dynamics=False,
            dynamics_tasks=3,
            seed=tr_cfg.seed,
        )
        out_l2 = run_naive_l2(
            model=_fresh_model_from_init(model_cfg, init_state),
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_tasks=ds_cfg.num_tasks,
            cfg=cfg_l2,
            device=device,
        )

    # EWC multi-center
    cfg_ewc = EWCConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        lambda_ewc=tr_cfg.lambda_ewc,
        fisher_num_samples=tr_cfg.fisher_num_samples,
        fisher_seed=tr_cfg.fisher_seed,
        fisher_num_workers=tr_cfg.fisher_num_workers,
        use_lr_decay=False,
        track_epoch_dynamics=False,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_ewc = run_ewc_multicenter(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=ds_cfg.num_tasks,
        cfg=cfg_ewc,
        device=device,
    )

    full_results: Dict[str, object] = {
        "scenario": "rotated",
        "num_tasks": ds_cfg.num_tasks,
        "angles_deg": angles,
        "single_task_acc": single_task_acc,
        "epochs_per_task": tr_cfg.epochs_per_task,
        "lr": tr_cfg.lr,
        "momentum": tr_cfg.momentum,
        "seed": tr_cfg.seed,
        "lambda_l2": tr_cfg.lambda_l2,
        "lambda_ewc": tr_cfg.lambda_ewc,
        "fisher_num_samples": tr_cfg.fisher_num_samples,
        "sgd": out_sgd,
        "ewc": out_ewc,
        "l2": out_l2,  # may be None
    }

    # Save (rotated-specific filenames)
    os.makedirs(paths.results_dir, exist_ok=True)
    npz_path = os.path.join(paths.results_dir, "rotated_fig2_results.npz")
    meta_path = os.path.join(paths.results_dir, "rotated_fig2_meta.json")

    save_dict = {
        "single_task_acc": single_task_acc,
        "sgd_acc_matrix": out_sgd["acc_matrix"].cpu().numpy(),
        "sgd_avg_acc_curve": np.array(out_sgd["avg_acc_curve"], dtype=np.float32),
        "ewc_acc_matrix": out_ewc["acc_matrix"].cpu().numpy(),
        "ewc_avg_acc_curve": np.array(out_ewc["avg_acc_curve"], dtype=np.float32),
        "angles_deg": np.array(angles, dtype=np.float32),
    }
    if out_l2 is not None:
        save_dict.update({
            "l2_acc_matrix": out_l2["acc_matrix"].cpu().numpy(),
            "l2_avg_acc_curve": np.array(out_l2["avg_acc_curve"], dtype=np.float32),
        })

    np.savez(npz_path, **save_dict)

    meta = {
        "paths": asdict(paths),
        "dataset": asdict(ds_cfg),
        "training": asdict(tr_cfg),
        "model": asdict(mdl_cfg),
        "device": str(device),
        "compute_l2": bool(compute_l2),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return full_results


def run_fig2a_dynamics_rotated_abc(
    compute_l2: bool,
    paths: PathsConfig = PathsConfig(),
    ds_cfg: RotatedMNISTConfig = RotatedMNISTConfig(),
    tr_cfg: TrainingConfig = TrainingConfig(),
    mdl_cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Runs the 3-task (A,B,C) dynamics experiment (first 3 rotated tasks)
    recording test accuracy every epoch, returning epoch_dynamics for Fig 2A plotting.
    """
    if device is None:
        device = get_device()

    set_global_seed(ds_cfg.seed)

    train_loaders, test_loaders, angles = build_rotated_mnist_loaders(
        data_root=paths.data_root,
        num_tasks=ds_cfg.num_tasks,
        angle_step_deg=ds_cfg.angle_step_deg,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        seed=ds_cfg.seed,
    )

    # Only first 3 tasks
    train_loaders = train_loaders[:3]
    test_loaders = test_loaders[:3]
    angles_abc = angles[:3]

    model_cfg = MLPConfig(
        input_dim=mdl_cfg.input_dim,
        hidden_dim1=mdl_cfg.hidden_dim1,
        hidden_dim2=mdl_cfg.hidden_dim2,
        num_classes=mdl_cfg.num_classes,
        use_batchnorm=mdl_cfg.use_batchnorm,
    )

    set_global_seed(tr_cfg.seed)
    _m0 = MNISTMLP(model_cfg)
    init_state = init_state_dict_from_model(_m0)

    # SGD dynamics
    cfg_sgd = BaselineConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        use_lr_decay=False,
        track_epoch_dynamics=True,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_sgd = run_baseline_sgd(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=3,
        cfg=cfg_sgd,
        device=device,
    )

    out_l2 = None
    if compute_l2:
        cfg_l2 = NaiveL2Config(
            epochs_per_task=tr_cfg.epochs_per_task,
            lr=tr_cfg.lr,
            momentum=tr_cfg.momentum,
            lambda_l2=tr_cfg.lambda_l2,
            use_lr_decay=False,
            track_epoch_dynamics=True,
            dynamics_tasks=3,
            seed=tr_cfg.seed,
        )
        out_l2 = run_naive_l2(
            model=_fresh_model_from_init(model_cfg, init_state),
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_tasks=3,
            cfg=cfg_l2,
            device=device,
        )

    # EWC dynamics
    cfg_ewc = EWCConfig(
        epochs_per_task=tr_cfg.epochs_per_task,
        lr=tr_cfg.lr,
        momentum=tr_cfg.momentum,
        lambda_ewc=tr_cfg.lambda_ewc,
        fisher_num_samples=tr_cfg.fisher_num_samples,
        fisher_seed=tr_cfg.fisher_seed,
        fisher_num_workers=tr_cfg.fisher_num_workers,
        use_lr_decay=False,
        track_epoch_dynamics=True,
        dynamics_tasks=3,
        seed=tr_cfg.seed,
    )
    out_ewc = run_ewc_multicenter(
        model=_fresh_model_from_init(model_cfg, init_state),
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_tasks=3,
        cfg=cfg_ewc,
        device=device,
    )

    return {
        "scenario": "rotated",
        "angles_deg_abc": angles_abc,
        "epochs_per_task": tr_cfg.epochs_per_task,
        "compute_l2": bool(compute_l2),
        "sgd_dyn": out_sgd["epoch_dynamics"],
        "l2_dyn": None if out_l2 is None else out_l2["epoch_dynamics"],
        "ewc_dyn": out_ewc["epoch_dynamics"],
    }


def run_permuted_experiment_2C(
    paths: PathsConfig = PathsConfig(),
    ds_cfg: PermutedMNISTConfig = PermutedMNISTConfig(),
    tr_cfg: TrainingConfig = TrainingConfig(),
    mdl_cfg: ModelConfig = ModelConfig(),
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Runs the 3-task experiment for Fig 2C with input dimensions 8, 8, 26.
    Each task permutes only the center input_dim x input_dim region of the 28x28 image.
    
    Trains a single model sequentially on all 3 tasks with SGD.
    After each task, saves model state and computes Fisher information on that task's data.
    Computes Fisher overlap per layer: one value for A vs B, one for mean(A vs C, B vs C).
    """
    if device is None:
        device = get_device()

    set_global_seed(ds_cfg.seed)

    train_loaders, test_loaders, perms_paths, input_dims = build_permuted_mnist_loaders_plotC(
        data_root=paths.data_root,
        artifacts_dir=paths.artifacts_dir,
        batch_size=ds_cfg.batch_size,
        num_workers=ds_cfg.num_workers,
        seed=ds_cfg.seed,
    )

    num_tasks = len(train_loaders)

    model_cfg = MLPConfig(
        input_dim=mdl_cfg.input_dim,
        hidden_dim1=mdl_cfg.hidden_dim1,
        hidden_dim2=mdl_cfg.hidden_dim2,
        num_classes=mdl_cfg.num_classes,
        use_batchnorm=mdl_cfg.use_batchnorm,
    )

    # Single model trained sequentially
    set_global_seed(tr_cfg.seed)
    model = MNISTMLP(model_cfg)
    model.to(device)

    saved_states = []
    fisher_dicts = []

    for task_idx in range(num_tasks):
        # Train on this task
        cfg_sgd = BaselineConfig(
            epochs_per_task=tr_cfg.epochs_per_task,
            lr=tr_cfg.lr,
            momentum=tr_cfg.momentum,
            use_lr_decay=False,
            track_epoch_dynamics=False,
            dynamics_tasks=1,
            seed=tr_cfg.seed,
        )
        
        run_baseline_sgd(
            model=model,
            train_loaders=[train_loaders[task_idx]],
            test_loaders=[test_loaders[task_idx]],
            num_tasks=1,
            cfg=cfg_sgd,
            device=device,
        )
        
        # Save model state after this task
        saved_states.append({k: v.clone().cpu() for k, v in model.state_dict().items()})
        
        # Compute Fisher information on this task's data
        fisher = compute_fisher_information(
            model=model,
            data_loader=train_loaders[task_idx],
            num_samples=tr_cfg.fisher_num_samples,
            device=device,
            seed=tr_cfg.fisher_seed,
        )
        fisher_dicts.append(fisher)

    # All 6 parameter groups (weight and bias for each layer)
    layer_names = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
    
    # Compute overlap per layer
    overlap_fisher = []
    
    for layer_name in layer_names:
        fisher_A = fisher_dicts[0][layer_name]
        fisher_B = fisher_dicts[1][layer_name]
        fisher_C = fisher_dicts[2][layer_name]
        
        # Overlap A-B
        overlap_AB = _compute_fisher_overlap(fisher_A, fisher_B)
        
        # Overlap A-C and B-C, then take mean
        overlap_AC = _compute_fisher_overlap(fisher_A, fisher_C)
        overlap_BC = _compute_fisher_overlap(fisher_B, fisher_C)
        overlap_mean_C = (overlap_AC + overlap_BC) / 2.0
        
        overlap_fisher.append([overlap_AB, overlap_mean_C])

    full_results: Dict[str, object] = {
        "scenario": "permuted_2C",
        "num_tasks": num_tasks,
        "input_dims": input_dims,
        "permutations_paths": perms_paths,
        "epochs_per_task": tr_cfg.epochs_per_task,
        "lr": tr_cfg.lr,
        "momentum": tr_cfg.momentum,
        "seed": ds_cfg.seed,
        "layer_names": layer_names,
        "overlap_fisher": overlap_fisher,
        "fisher_dicts": fisher_dicts,
        "saved_states": saved_states,
    }

    # Save
    os.makedirs(paths.results_dir, exist_ok=True)
    npz_path = os.path.join(paths.results_dir, "permuted_fig2C_results.npz")
    meta_path = os.path.join(paths.results_dir, "permuted_fig2C_meta.json")

    save_dict = {
        "input_dims": np.array(input_dims, dtype=np.int32),
        "overlap_fisher": np.array(overlap_fisher, dtype=np.float32),
        "layer_names": np.array(layer_names, dtype=object),
    }

    np.savez(npz_path, **save_dict)

    meta = {
        "paths": asdict(paths),
        "dataset": asdict(ds_cfg),
        "training": asdict(tr_cfg),
        "model": asdict(mdl_cfg),
        "device": str(device),
        "input_dims": input_dims,
        "layer_names": layer_names,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return full_results


def _compute_fisher_overlap(fisher1: torch.Tensor, fisher2: torch.Tensor) -> float:
    """
    Compute overlap between two Fisher information tensors.
    
    Each Fisher is first normalized to unit trace, then the squared Fréchet distance is computed:
    d^2(F1, F2) = (1/2) * tr(F1 + F2 - 2(F1*F2)^{1/2})
                = (1/2) * || F1^{1/2} - F2^{1/2} ||_F^2
    
    Overlap is defined as 1 - d^2.
    """
    f1 = fisher1.flatten().float()
    f2 = fisher2.flatten().float()
    
    # Clamp to nonnegative for numerical safety
    f1 = torch.clamp(f1, min=0.0)
    f2 = torch.clamp(f2, min=0.0)
    
    # Compute sums
    f1_sum = f1.sum().item()
    f2_sum = f2.sum().item()
    
    # Handle degenerate cases
    if f1_sum == 0 and f2_sum == 0:
        return 1.0  # Both zero, consider identical
    if f1_sum == 0 or f2_sum == 0:
        return 0.0  # One is zero, no overlap
    
    # Normalize each Fisher to unit trace
    f1 = f1 / f1_sum
    f2 = f2 / f2_sum
    
    # Element-wise square root (since Fisher is diagonal)
    f1_sqrt = torch.sqrt(f1)
    f2_sqrt = torch.sqrt(f2)
    
    # Squared Fréchet distance: (1/2) * || F1^{1/2} - F2^{1/2} ||_F^2
    diff = f1_sqrt - f2_sqrt
    frechet_dist_sq = 0.5 * torch.sum(diff ** 2).item()
    
    # Overlap = 1 - d^2
    overlap = 1.0 - frechet_dist_sq
    
    return max(0.0, min(1.0, overlap))  # Clamp to [0, 1]
