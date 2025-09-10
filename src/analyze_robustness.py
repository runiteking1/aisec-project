import logging
import os
from matplotlib import pyplot as plt
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from src.models import CNN
from train import get_datasets

log = logging.getLogger(__name__)

def calculate_metrics(
        state: TrainState,
        images: jnp.ndarray,
        labels: jnp.ndarray
):
    """
    Calculates the input gradient norm and logit margin for a batch of images.
    """
    # Define a loss function that takes the image as the main argument
    def loss_fn(params, img, label):
        logits = state.apply_fn({'params': params}, jnp.expand_dims(img, axis=0))
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=jnp.array([label])
        ).mean()
        return loss, logits

    # Use jax.grad to get the gradient of the loss w.r.t the image
    grad_fn = jax.vmap(jax.grad(lambda img, label: loss_fn(state.params, img, label)[0]), in_axes=(0, 0))

    # Use jax.vmap to calculate logits for the whole batch
    logits_fn = jax.vmap(lambda img: state.apply_fn({'params': state.params}, jnp.expand_dims(img, axis=0))[0],
                         in_axes=0)

    # Calculate gradients for the batch
    gradients = grad_fn(images, labels)

    # Calculate logits for the batch
    logits_batch = logits_fn(images)

    # Calculate the L2 norm of the gradients (input gradient norm)
    # The norm is calculated for each image's gradient vector
    input_gradient_norms = jnp.linalg.norm(gradients.reshape(gradients.shape[0], -1), axis=-1)

    # Calculate the logit margin
    # Sort logits and take the difference between the correct logit and the
    # second-highest logit (which is the highest incorrect logit)
    true_class_logits = logits_batch[jnp.arange(labels.shape[0]), labels]

    # Create a mask to zero out the true class logit for each example
    mask = jnp.arange(logits_batch.shape[1]) != labels[:, None]

    # Find the second-highest logit by using the mask
    # We set the true class logit to a very low value before finding the max
    masked_logits = jnp.where(mask, logits_batch, -jnp.inf)

    second_highest_logits = jnp.max(masked_logits, axis=-1)

    # Logit margin is the difference between the true logit and the highest other logit
    logit_margins = true_class_logits - second_highest_logits

    return input_gradient_norms, logit_margins


def plot_distributions(gradient_norms, logit_margins, output_dir):
    """
    Plots the distributions of the input gradient norms and logit margins.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Input Gradient Norm distribution
    plt.figure(figsize=(10, 6))
    plt.hist(gradient_norms, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Input Gradient Norms")
    plt.xlabel("L2 Norm of Input Gradient")
    plt.ylabel("Frequency")
    plt.xlim(xmin=0, xmax=4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "input_gradient_norm_distribution.png"))
    plt.close()

    # Plot Logit Margin distribution
    plt.figure(figsize=(10, 6))
    plt.hist(logit_margins, bins=100, alpha=0.7, color='green', edgecolor='black')
    plt.title("Distribution of Logit Margins")
    plt.xlabel("Logit Margin (True Logit - Max Incorrect Logit)")
    plt.ylabel("Frequency")
    plt.xlim(xmin=-1, xmax=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "logit_margin_distribution.png"))
    plt.close()

    log.info(f"Plots saved to: {output_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="analyze_robustness")
def analyze_robustness(cfg: DictConfig) -> None:
    """
    Main function to analyze model robustness by calculating and plotting
    input gradient norms and logit margins.
    """
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Load the model
    checkpoint_path = cfg.checkpoint_path
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    restored = checkpointer.restore(checkpoint_path)
    state = restored['state']
    model_config = OmegaConf.create(restored['config'])

    model = CNN(
        features_per_layer=tuple(model_config.model.features_per_layer),
        kernel_size=tuple(model_config.model.kernel_size),
        dense_features=tuple(model_config.model.dense_features),
        num_classes=cfg.data.num_classes
    )

    state = TrainState.create(apply_fn=model.apply, params=state['params'], tx=optax.adam(1))

    # Load datasets
    datasets = get_datasets()
    test_ds = datasets['test']
    test_images = test_ds['image']
    test_labels = test_ds['label']

    batch_size = cfg.batch_size
    num_test_images = len(test_images)
    num_batches = len(range(0, num_test_images, batch_size))

    all_gradient_norms = []
    all_logit_margins = []

    log.info(f"Calculating metrics on {num_test_images} test images...")

    pbar = trange(num_batches, desc="Processing batches")
    for i in pbar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_images)

        image_batch = test_images[start_idx:end_idx]
        label_batch = test_labels[start_idx:end_idx]

        norms, margins = calculate_metrics(state, image_batch, label_batch)

        all_gradient_norms.append(norms)
        all_logit_margins.append(margins)

    # Concatenate results from all batches
    all_gradient_norms = jnp.concatenate(all_gradient_norms, axis=0)
    all_logit_margins = jnp.concatenate(all_logit_margins, axis=0)

    # Plot the distributions
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    plot_distributions(all_gradient_norms, all_logit_margins, output_dir)

    log.info("--- Robustness Analysis Results ---")
    log.info(f"Mean Input Gradient Norm: {jnp.mean(all_gradient_norms):.4f}")
    log.info(f"Median Input Gradient Norm: {jnp.median(all_gradient_norms):.4f}")
    log.info(f"Mean Logit Margin: {jnp.mean(all_logit_margins):.4f}")
    log.info(f"Median Logit Margin: {jnp.median(all_logit_margins):.4f}")


if __name__ == "__main__":
    analyze_robustness()