# EE411-Project — Reproducibility Challenge: Overcoming Catastrophic Forgetting in Neural Networks

**About • Data • Installation • Method • How To Use • Credits • Code Structure • Results • License**

---

## Team

- **[Sofija Orlovic]** 
- **[Jane Klavir]**
- **[Enric Guasch Mesia]** 
- **[Petar Damjanovic]** 
- **[Marko Stojanovic]** 

---

## About

This repository contains our work for the **EE411 / Fundamentals of Inference and Learning** reproducibility challenge.  
Our goal is to reproduce key results from:

- **Kirkpatrick et al. (2017)**, *Overcoming catastrophic forgetting in neural networks* (EWC)

We study **catastrophic forgetting** in sequential task learning and evaluate how **Elastic Weight Consolidation (EWC)** mitigates forgetting compared to:
- **SGD** (no regularization)
- **L2 regularization**
- **EWC**

We reproduce continual-learning diagnostics on **PermutedMNIST** and **RotatedMNIST**, and include an optional **Atari notebook** used for exploratory / reduced-setting experiments.

---

## Data

We use MNIST in two continual-learning task streams:

- **PermutedMNIST**: each task applies a *fixed random permutation* of pixel indices to all MNIST images for that task.
- **RotatedMNIST**: each task applies a *fixed rotation* to MNIST images (commonly in steps of 10 degrees).

Dataset utilities are implemented in:
- `data.py` (PermutedMNIST task stream)
- `data_rotated.py` (RotatedMNIST task stream)

---

## Installation

Create an environment and install dependencies.

### Option A — `venv`
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install torch torchvision numpy matplotlib pandas jupyter

## Method

We train a single neural network sequentially on tasks (Task 1 → Task 2 → …) and evaluate accuracy on all tasks throughout training to quantify forgetting.

### Baselines

- **SGD**: standard training sequentially across tasks.
- **L2**: adds an L2 penalty on network parameters to discourage large weights.

### Elastic Weight Consolidation (EWC)

EWC penalizes changes to parameters that are important for previously learned tasks using a diagonal Fisher approximation:

\[
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta)
+ \lambda \sum_i F_i(\theta_i - \theta_i^\star)^2.
\]

where \(F_i\) is the estimated Fisher information for parameter \(\theta_i\), \(\theta^\star\) are consolidated parameters from previous tasks, and \(\lambda\) controls the strength of the regularizer.

Method implementations:
- `methods/sgd.py` — SGD baseline  
- `methods/l2.py` — L2 baseline  
- `methods/ewc.py` — EWC (Fisher estimation + consolidation + penalty)  

Core experiment/plot logic:
- `experiments.py` — experiment runners / orchestration  
- `model.py` — model architecture  
- `plotting.py` — figure generation helpers  
- `utils.py` — shared utilities  
- `config.py` — configuration (hyperparameters, seeds, paths)  

---

## How To Use

### Recommended: run via notebooks (figure reproduction)

Main entry points:
- `notebooks/run_fig2A_fig2B.ipynb` — Fig. 2A / Fig. 2B-style experiments for PermutedMNIST and RotatedMNIST  
- `notebooks/run_fig2C.ipynb` — Fig. 2C-style diagnostic  
- `notebooks/atari (1).ipynb` — optional Atari exploration (reduced setting)

Run:
```bash
jupyter notebook


EE411-Project/
├── methods/                          # Continual-learning methods (regularizers / training variants)
│   ├── __init__.py
│   ├── ewc.py                        # EWC implementation (Fisher + consolidation + penalty)
│   ├── l2.py                         # L2 regularization baseline
│   └── sgd.py                        # Plain SGD baseline
│
├── notebooks/                        # Reproduction notebooks (main entry points)
│   ├── atari (1).ipynb               # Optional Atari / reduced setting exploration
│   ├── run_fig2A_fig2B.ipynb         # Main notebook: Fig2A/Fig2B-style results
│   └── run_fig2C.ipynb               # Main notebook: Fig2C-style results
│
├── .gitignore
├── __init__.py
├── config.py                         # Central config (hyperparameters, seeds, paths, etc.)
├── data.py                           # PermutedMNIST dataset/task stream utilities
├── data_rotated.py                   # RotatedMNIST dataset/task stream utilities
├── experiments.py                    # Experiment runners / orchestration
├── model.py                          # Model architecture(s)
├── plotting.py                       # Plotting helpers for report figures
└── utils.py                          # General utilities (logging, metrics, helpers)


