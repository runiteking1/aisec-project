"""
Loss landscape visualization using filter-normalized random directions
(Li et al., 2018: "Visualizing the Loss Landscape of Neural Nets").

Plots 2D contour maps of the loss surface around two trained models
for side-by-side comparison of landscape sharpness.
"""
import logging
import os

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models import CNN
from src.data import get_datasets

log = logging.getLogger(__name__)


def load_model(checkpoint_path: str, num_classes: int) -> TrainState:
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    restored = checkpointer.restore(checkpoint_path)
    model_config = OmegaConf.create(restored['config'])
    model = CNN(
        features_per_layer=tuple(model_config.model.features_per_layer),
        kernel_size=tuple(model_config.model.kernel_size),
        dense_features=tuple(model_config.model.dense_features),
        num_classes=num_classes,
    )
    return TrainState.create(
        apply_fn=model.apply, params=restored['state']['params'], tx=optax.adam(1)
    )


def filter_normalize(direction: dict, params: dict) -> dict:
    """
    Scale each filter of `direction` so it has the same norm as the
    corresponding filter in `params` (Li et al. 2018, Section 4).

    For weight tensors (ndim >= 2) the first axis is treated as the filter axis.
    For bias vectors (ndim == 1) a single scalar normalization is applied.
    """
    def _normalize(d, p):
        if p.ndim >= 2:
            # (num_filters, -1) view
            p_flat = p.reshape(p.shape[0], -1)
            d_flat = d.reshape(d.shape[0], -1)
            p_norms = jnp.linalg.norm(p_flat, axis=1, keepdims=True)          # (F, 1)
            d_norms = jnp.linalg.norm(d_flat, axis=1, keepdims=True) + 1e-10  # (F, 1)
            return (d_flat * p_norms / d_norms).reshape(d.shape)
        else:
            return d * jnp.linalg.norm(p) / (jnp.linalg.norm(d) + 1e-10)

    return jax.tree_util.tree_map(_normalize, direction, params)


def compute_landscape(state: TrainState, dir1: dict, dir2: dict,
                      images: jnp.ndarray, labels: jnp.ndarray,
                      num_classes: int, grid_size: int, coord_range: float):
    """
    Evaluate loss on a (grid_size × grid_size) grid of parameter perturbations.

    Returns:
        coords: 1-D array of coordinate values (same for both axes)
        losses: (grid_size, grid_size) array of loss values
    """
    coords = np.linspace(-coord_range, coord_range, grid_size)

    # Flatten everything for fast arithmetic
    flat_params, unravel = jax.flatten_util.ravel_pytree(state.params)
    flat_d1, _ = jax.flatten_util.ravel_pytree(dir1)
    flat_d2, _ = jax.flatten_util.ravel_pytree(dir2)

    @jax.jit
    def loss_at(alpha, beta):
        perturbed = unravel(flat_params + alpha * flat_d1 + beta * flat_d2)
        logits = state.apply_fn({'params': perturbed}, images)
        one_hot = jax.nn.one_hot(labels, num_classes)
        return jnp.mean(-jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

    losses = np.zeros((grid_size, grid_size))
    total = grid_size * grid_size
    with tqdm(total=total, desc="  grid", leave=False) as pbar:
        for i, alpha in enumerate(coords):
            for j, beta in enumerate(coords):
                losses[i, j] = float(loss_at(jnp.array(alpha), jnp.array(beta)))
                pbar.update(1)

    return coords, losses


@hydra.main(version_base=None, config_path="../../conf", config_name="landscape")
def visualize_landscape(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    state_a = load_model(cfg.checkpoint_path_a, cfg.data.num_classes)
    state_b = load_model(cfg.checkpoint_path_b, cfg.data.num_classes)

    # Fixed random subset of test set for loss evaluation
    key = jax.random.PRNGKey(cfg.seed)
    test_ds = get_datasets()['test']
    idx = jax.random.choice(key, len(test_ds['image']), shape=(cfg.num_samples,), replace=False)
    images = test_ds['image'][idx]
    labels = test_ds['label'][idx]

    log.info(
        f"Grid: {cfg.grid_size}×{cfg.grid_size}, "
        f"range ±{cfg.coord_range}, "
        f"{cfg.num_samples} test samples"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, state, name in zip(axes,
                                [state_a, state_b],
                                [cfg.label_a, cfg.label_b]):
        # Sample two independent Gaussian directions, then filter-normalize
        key, k1, k2 = jax.random.split(key, 3)
        dir1 = jax.tree_util.tree_map(lambda p: jax.random.normal(k1, p.shape), state.params)
        dir2 = jax.tree_util.tree_map(lambda p: jax.random.normal(k2, p.shape), state.params)
        dir1 = filter_normalize(dir1, state.params)
        dir2 = filter_normalize(dir2, state.params)

        log.info(f"Computing {name} landscape...")
        coords, losses = compute_landscape(
            state, dir1, dir2, images, labels,
            cfg.data.num_classes, cfg.grid_size, cfg.coord_range,
        )

        center_loss = losses[cfg.grid_size // 2, cfg.grid_size // 2]
        log.info(
            f"  {name}: loss at solution={center_loss:.4f}, "
            f"min={losses.min():.4f}, max={losses.max():.4f}"
        )

        contour = ax.contourf(coords, coords, losses, levels=40, cmap='viridis')
        ax.contour(coords, coords, losses, levels=40, colors='white', alpha=0.15, linewidths=0.4)
        fig.colorbar(contour, ax=ax, label='Cross-entropy loss')
        ax.set_title(f'{name}', fontsize=13)
        ax.set_xlabel('Direction 1 (filter-normalized)')
        ax.set_ylabel('Direction 2 (filter-normalized)')
        # Mark the solution (origin)
        ax.plot(0, 0, 'r*', markersize=14, label='Trained solution', zorder=5)
        ax.legend(fontsize=9)

    fig.suptitle('Loss Landscape (filter-normalized random directions)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_landscape.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    log.info(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    visualize_landscape()
