# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating adversarial robustness and loss landscape sharpness of higher-order optimizers (Gauss-Newton/Levenberg-Marquardt) vs. Adam. Models are trained on MNIST and CIFAR-10, then analyzed for adversarial vulnerability and sharpness characteristics.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (outputs saved to outputs/[timestamp]/)
python -m src.train                         # default: CNN + Adam + MNIST
python -m src.train training=gn             # Gauss-Newton optimizer
python -m src.train training=sgd            # SGD optimizer
python -m src.train data=cifar              # CIFAR-10 dataset
python -m src.train model=vit               # Vision Transformer

# Analysis (requires checkpoint path from a prior training run)
python -m src.analyze_robustness checkpoint_path="/path/to/outputs/.../checkpoint/model"
python -m src.analyze_sharpness  checkpoint_path="/path/to/outputs/.../checkpoint/model" rho=0.01
python -m src.generate_adversarial checkpoint_path="/path/to/outputs/.../checkpoint/model" epsilon=0.1
```

There are no test or lint commands configured.

## Architecture

### Configuration (Hydra)

`conf/config.yaml` composes sub-configs from three groups:
- `conf/training/` — optimizer configs: `default.yaml` (Adam), `gn.yaml` (Gauss-Newton), `sgd.yaml`
- `conf/model/` — architecture configs: `default.yaml` (CNN), `vit.yaml` (Vision Transformer)
- `conf/data/` — dataset configs: `mnist.yaml`, `cifar.yaml`

Analysis tools have their own configs: `conf/adversarial.yaml`, `conf/analyze_robustness.yaml`, `conf/analyze_sharpness.yaml`.

### Pipeline

```
src/train.py  →  outputs/[timestamp]/checkpoint/model/
                    ↓
    ├── src/analyze_robustness.py   (gradient norms, logit margins)
    ├── src/analyze_sharpness.py    (SAM-style loss curvature)
    └── src/generate_adversarial.py (FGSM attacks)
```

All scripts are Hydra entry points. Training saves checkpoints with Orbax (includes params + full config). Analysis scripts load these checkpoints and write result plots alongside them.

### Models (`src/models.py`)

- **CNN**: Configurable conv layers → average pool → dense layers → classifier
- **ViT**: Patch embedding + CLS token + positional embeddings → Transformer encoder blocks → classifier from CLS token

### Optimizers (`src/train.py`)

- **Adam / SGD**: Standard Optax optimizers
- **Gauss-Newton** (`training=gn`): Custom implementation using SGD base but preconditioning gradients with `(J^T H J + λI)^{-1} g`, where J is the per-sample Jacobian and H is the softmax Hessian. Levenberg-Marquardt damping controlled by `gn_param`.

### Datasets

Downloaded from HuggingFace `datasets` on first run, then cached as `.npy` files (`processed_mnist_*.npy`, `processed_cifar10_*.npy`). CIFAR-10 config uses a subset of 4 classes by default.

### Output Structure

Hydra creates timestamped output directories:
```
outputs/[YYYY-MM-DD]/[HH-MM-SS]/
├── .hydra/              # Hydra run metadata
├── checkpoint/model/    # Orbax checkpoint (params + config)
├── training_metrics.npz / .png
└── [analysis]_*.png     # Plots from analysis scripts
```

`src/multirun/` holds Hydra multirun outputs from parameter sweeps.
