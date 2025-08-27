import logging
import os
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from PIL import Image
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tqdm import trange  # For the progress bar

from src.models import CNN
from train import get_datasets

log = logging.getLogger(__name__)


def save_image(image_array: jnp.ndarray, path: str):
    img_np = jnp.array(image_array)

    img_scaled = (img_np * 255).astype(jnp.uint8)
    img_scaled = np.array(img_scaled).reshape((28, 28))

    img_pil = Image.fromarray(img_scaled)
    img_pil.save(path)


@partial(jax.jit, static_argnames=("epsilon",))
def fgsm_attack(state: TrainState, image: jnp.ndarray, label: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    """
    Performs the Fast Gradient Sign Method (FGSM) attack on a single image.

    This function calculates the gradient of the loss with respect to the input
    image, and then creates a perturbation by taking the sign of the gradient
    multiplied by epsilon.

    Args:
        state: The trained TrainState of the model, containing params and apply_fn.
        image: The input image as a jnp.ndarray (H, W, C). It should be normalized,
               e.g., in the [0, 1] range.
        label: The true integer label of the image.
        epsilon: The perturbation magnitude. A small float value (e.g., 0.1).

    Returns:
        The adversarially perturbed image, clipped to the valid range [0, 1].
    """

    # Define a loss function that takes the image as the main argument
    # to be differentiated by jax.grad.
    def loss_fn(img):
        # The model's apply_fn expects a batch of images.
        # We add a batch dimension to the single image.
        logits = state.apply_fn({'params': state.params}, jnp.expand_dims(img, axis=0))
        # We calculate the loss against the true label. The goal of the attack
        # is to create a perturbation that maximizes this loss.
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=jnp.array([label])
        ).mean()
        return loss

    # Calculate the gradient of the loss with respect to the input image.
    gradient = jax.grad(loss_fn)(image)

    # Find the sign of the gradient. This determines the direction of the
    # perturbation for each pixel.
    signed_grad = jnp.sign(gradient)

    # Create the perturbation by scaling the signed gradient by epsilon.
    perturbation = epsilon * signed_grad

    # Add the perturbation to the original image to create the adversarial example.
    adversarial_image = image + perturbation

    # Clip the resulting image to maintain the original data range (e.g., [0, 1])
    # and ensure it's still a valid image.
    adversarial_image = jnp.clip(adversarial_image, 0, 1)

    return adversarial_image


@hydra.main(version_base=None, config_path="../conf", config_name="adversarial")
def train_adversarial(cfg: DictConfig) -> None:
    """
    Main training loop for generating adversarial examples
    """
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # Load the model first; should make this in Hydra; but for now hard code
    checkpoint_path = '/home/marshall/PycharmProjects/AISEC-project/src/outputs/2025-08-21/21-50-45/checkpoints/model'
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    restored = checkpointer.restore(checkpoint_path)
    state = restored['state']
    model_config = OmegaConf.create(restored['config'])

    # Create and initialize model
    model = CNN(
        features_per_layer=tuple(model_config.model.features_per_layer),
        kernel_size=tuple(model_config.model.kernel_size),
        dense_features=tuple(model_config.model.dense_features),
        num_classes=model_config.model.num_classes
    )

    # Create state for easy porting; but no optimizer needed
    state = TrainState.create(apply_fn=model.apply, params=state['params'], tx=optax.adam(1))

    datasets = get_datasets()
    test_ds = datasets['test']
    test_images = test_ds['image']
    test_labels = test_ds['label']
    epsilon = cfg.epsilon

    # in_axes maps arguments: None=broadcast, 0=map over the first axis
    vmapped_fgsm_attack = jax.vmap(fgsm_attack, in_axes=(None, 0, 0, None))

    batch_size = cfg.batch_size
    num_test_images = len(test_images)
    num_batches = len(range(0, num_test_images, batch_size))

    total_successful_attacks = 0
    total_initially_correct = 0

    # Create a directory to save the images
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_path = os.path.join(output_dir, "fgsm_examples")
    os.makedirs(save_path, exist_ok=True)
    log.info(f"Saving example images to: {save_path}")

    # Initialize a counter for saved images
    saved_count = 0
    save_limit = cfg.output_images  # Set the maximum number of images to save

    log.info(f"Starting FGSM attack on {num_test_images} test images with epsilon={epsilon}...")

    # 4. Process the dataset in batches
    pbar = trange(num_batches, desc="Attacking batches")
    for i in pbar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_images)

        image_batch = test_images[start_idx:end_idx]
        label_batch = test_labels[start_idx:end_idx]

        # Get the model's predictions on the original, clean images
        original_logits = state.apply_fn({'params': state.params}, image_batch)
        original_preds = jnp.argmax(original_logits, axis=-1)

        # Identify which images the model correctly classified *before* the attack
        correctly_classified_mask = (original_preds == label_batch)
        initially_correct_in_batch = correctly_classified_mask.sum()
        total_initially_correct += initially_correct_in_batch

        # Only attack images that were correctly classified
        if initially_correct_in_batch == 0:
            continue

        attack_images = image_batch[correctly_classified_mask]
        attack_labels = label_batch[correctly_classified_mask]

        # Generate adversarial examples for the correctly classified subset
        adversarial_batch = vmapped_fgsm_attack(state, attack_images, attack_labels, epsilon)

        # Get predictions on the new adversarial images
        adversarial_logits = state.apply_fn({'params': state.params}, adversarial_batch)
        adversarial_preds = jnp.argmax(adversarial_logits, axis=-1)

        # An attack is successful if the prediction is no longer correct
        successful_attacks_in_batch_mask = (adversarial_preds != attack_labels)
        successful_attacks_in_batch = (adversarial_preds != attack_labels).sum()
        total_successful_attacks += successful_attacks_in_batch

        # Save image
        if saved_count < save_limit and successful_attacks_in_batch > 0:
            successful_attack_indices = jnp.where(successful_attacks_in_batch_mask)[0]

            for idx in successful_attack_indices:
                if saved_count >= save_limit:
                    break

                # Get the original and adversarial image from the subset
                original_img = attack_images[idx]
                adversarial_img = adversarial_batch[idx]

                # Save the original image
                orig_filename = f"original_{total_successful_attacks}_{saved_count}_label_{attack_labels[idx]}.png"
                orig_path = os.path.join(save_path, orig_filename)
                save_image(original_img, orig_path)

                # Save the adversarial image
                adv_filename = f"adversarial_{total_successful_attacks}_{saved_count}_label_{adversarial_preds[idx]}.png"
                adv_path = os.path.join(save_path, adv_filename)
                save_image(adversarial_img, adv_path)

                saved_count += 1

    if total_initially_correct > 0:
        attack_success_rate = (total_successful_attacks / total_initially_correct) * 100
    else:
        attack_success_rate = 0.0

    log.info("--- FGSM Attack Results ---")
    log.info(f"Total test images correctly classified by the model: {total_initially_correct}/{num_test_images}")
    log.info(f"Successful attacks (fooled the model): {total_successful_attacks}")
    log.info(f"Attack Success Rate: {attack_success_rate:.2f}%")


# Run the training script
if __name__ == "__main__":
    train_adversarial()
