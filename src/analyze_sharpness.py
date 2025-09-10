# analyze_sharpness.py
import logging
import os
import matplotlib.pyplot as plt
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from jax.tree_util import tree_map

from src.models import CNN
from train import get_datasets

log = logging.getLogger(__name__)


# analyze_sharpness.py (Corrected function)
def calculate_sam_loss_increase(
        state: TrainState,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        rho: float
) -> jnp.ndarray:
    """
    Calculates the SAM-style loss increase for a batch of images and labels.
    """

    # Define a single loss function for the entire batch
    def batch_loss_fn(params, batch_images, batch_labels):
        logits = state.apply_fn({'params': params}, batch_images)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_labels
        ).mean()
        return loss

    # 1. Get the gradient of the mean batch loss with respect to the model parameters
    grad_fn = jax.grad(batch_loss_fn)
    grads = grad_fn(state.params, images, labels)

    # 2. Compute the perturbation direction and scale it by rho
    def perturb_params(params, grads, rho):
        # Calculate the L2 norm of the gradient tree
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))

        # Add a small epsilon for stability
        epsilon_hat = tree_map(lambda g: g / (grad_norm + 1e-12), grads)

        # Perturb the parameters
        return tree_map(lambda p, g: p + rho * g, params, epsilon_hat)

    perturbed_params = perturb_params(state.params, grads, rho)

    # 3. Calculate the loss at the original and perturbed parameters
    original_loss = batch_loss_fn(state.params, images, labels)
    perturbed_loss = batch_loss_fn(perturbed_params, images, labels)

    # The loss increase is the difference
    loss_increase = perturbed_loss - original_loss

    # Return a batch-sized array to match the plotting function's expectations.
    # Note: For simplicity, we'll repeat the single loss increase value for the batch.
    return jnp.full(images.shape[0], loss_increase)

def plot_distribution(data, title, xlabel, output_dir, filename):
    """
    Plots the distribution of the given data.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


@hydra.main(version_base=None, config_path="../conf", config_name="analyze_sharpness")
def analyze_sharpness(cfg: DictConfig) -> None:
    """
    Main function to analyze model sharpness by calculating the SAM-style loss increase.
    """
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Load the model from the specified checkpoint path
    checkpoint_path = cfg.checkpoint_path
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    restored = checkpointer.restore(checkpoint_path)
    state = restored['state']
    model_config = OmegaConf.create(restored['config'])

    # Recreate the model and state
    model = CNN(
        features_per_layer=tuple(model_config.model.features_per_layer),
        kernel_size=tuple(model_config.model.kernel_size),
        dense_features=tuple(model_config.model.dense_features),
        num_classes=cfg.data.num_classes
    )
    state = TrainState.create(apply_fn=model.apply, params=state['params'], tx=optax.adam(1))

    # Load the test dataset
    datasets = get_datasets()
    test_ds = datasets['test']
    test_images = test_ds['image']
    test_labels = test_ds['label']

    rho = cfg.rho
    batch_size = cfg.batch_size
    num_test_images = len(test_images)
    num_batches = len(range(0, num_test_images, batch_size))

    all_loss_increases = []

    log.info(f"Calculating SAM-style loss increase on {num_test_images} test images with rho={rho}...")

    pbar = trange(num_batches, desc="Processing batches")
    for i in pbar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_images)

        image_batch = test_images[start_idx:end_idx]
        label_batch = test_labels[start_idx:end_idx]

        loss_increases = calculate_sam_loss_increase(state, image_batch, label_batch, rho)
        all_loss_increases.append(loss_increases)

    # Concatenate results from all batches
    all_loss_increases = jnp.concatenate(all_loss_increases, axis=0)

    # Plot the distribution
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    plot_distribution(
        all_loss_increases,
        "Distribution of SAM-style Loss Increase",
        "Loss Increase (at Perturbed Weights)",
        output_dir,
        "sam_loss_increase_distribution.png"
    )

    log.info("--- Sharpness Analysis Results ---")
    log.info(f"Mean Loss Increase: {jnp.mean(all_loss_increases):.4f}")
    log.info(f"Median Loss Increase: {jnp.median(all_loss_increases):.4f}")

if __name__ == "__main__":
    analyze_sharpness()
