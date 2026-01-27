from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_fig2b_paper_like(full_results: Dict[str, object], compute_l2: bool, title: Optional[str] = None) -> None:
    """
    Figure 2B-style (paper-like) plot for PermutedMNIST:
    - Average test accuracy vs number of tasks
    - Circle markers at points
    - SGD (blue) + EWC (red), optional L2 (dark green)
    - Dashed single-task baseline
    - y-axis as fraction correct (not percent)
    """
    num_tasks = int(full_results["num_tasks"])
    x = np.arange(1, num_tasks + 1)

    # Fractions (0..1)
    sgd_curve = np.array(full_results["sgd"]["avg_acc_curve"], dtype=np.float32)
    ewc_curve = np.array(full_results["ewc"]["avg_acc_curve"], dtype=np.float32)
    single = float(full_results["single_task_acc"])

    # Paper-like colors
    COLOR_EWC = "#b54a4a"
    COLOR_SGD = "#4f6fb3"
    COLOR_L2  = "#2f6b2f"

    sgd_label = "SGD+dropout" if bool(full_results.get("sgd_use_dropout", False)) else "SGD"

    plt.figure(figsize=(7.6, 4.8))

    plt.plot(x, ewc_curve, color=COLOR_EWC, marker="o", linewidth=2.0, label="EWC")
    plt.plot(x, sgd_curve, color=COLOR_SGD, marker="o", linewidth=2.0, label=sgd_label)

    if compute_l2 and full_results.get("l2", None) is not None:
        l2_curve = np.array(full_results["l2"]["avg_acc_curve"], dtype=np.float32)
        plt.plot(x, l2_curve, color=COLOR_L2, marker="o", linewidth=2.0, label="L2")

    plt.axhline(single, color="black", linestyle="--", linewidth=1.5, label="single task performance")

    if title is not None:
        plt.title(title)

    plt.xlabel("Number of tasks")
    plt.ylabel("Fraction correct")
    plt.xticks(x)
    plt.grid(False)
    plt.tight_layout()
    plt.legend(frameon=False, loc="best")
    plt.show()


def plot_fig2a_paper_like(dyn_results: Dict[str, object], title: Optional[str] = None) -> None:
    """
    Figure 2A-style (paper-like) plot:
    - 3 stacked subplots (Task A, Task B, Task C)
    - Each subplot contains 3 lines: EWC (red), L2 (dark green), SGD (blue)
    - Vertical dashed lines at task switch boundaries
    - y-limits fixed to [0.75, 1.0]
    """
    dyn_sgd = dyn_results["sgd_dyn"]
    dyn_l2 = dyn_results.get("l2_dyn", None)
    dyn_ewc = dyn_results["ewc_dyn"]

    # Colors to match your paper-style target
    COLOR_EWC = "#b54a4a"
    COLOR_SGD = "#4f6fb3"
    COLOR_L2  = "#2f6b2f"

    xs = np.array(dyn_sgd["global_epoch_index"], dtype=np.int32) + 1

    switch_lines = []
    for g_ep, _task_id in dyn_sgd["task_switch_markers"]:
        if g_ep > 0:
            switch_lines.append(g_ep)

    task_labels = ["Task A", "Task B", "Task C"]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.2, 5.6), sharex=True)

    if title is not None:
        fig.suptitle(title)

    for t in range(3):
        ax = axes[t]

        y_ewc = np.array(dyn_ewc["acc_by_task"][t], dtype=np.float32)
        y_sgd = np.array(dyn_sgd["acc_by_task"][t], dtype=np.float32)

        ax.plot(xs, y_ewc, color=COLOR_EWC, linewidth=2.0, label="EWC" if t == 0 else None)
        if dyn_l2 is not None:
            y_l2 = np.array(dyn_l2["acc_by_task"][t], dtype=np.float32)
            ax.plot(xs, y_l2, color=COLOR_L2, linewidth=2.0, label="L2" if t == 0 else None)
        ax.plot(xs, y_sgd, color=COLOR_SGD, linewidth=2.0, label="SGD" if t == 0 else None)

        for v in switch_lines:
            ax.axvline(v, color="gray", linestyle="--", linewidth=1.5, dashes=(4, 4))

        ax.set_ylim(0.75, 1.0)
        ax.set_ylabel(task_labels[t])

        if t == 0:
            ax.legend(loc="center right", frameon=False)

    axes[-1].set_xlabel("Training time (cumulative epochs)")
    plt.tight_layout()
    plt.show()
