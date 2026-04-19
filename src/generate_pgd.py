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
from tqdm import trange

from src.models import CNN
from src.train import get_datasets

log = logging.getLogger(__name__)


def save_image(image_array: jnp.ndarray, path: str):
    img_np = jnp.array(image_array)
    img_scaled = (img_np * 255).astype(jnp.uint8)
    img_scaled = np.array(img_scaled).reshape((28, 28))
    img_pil = Image.fromarray(img_scaled)
    img_pil.save(path)


@partial(jax.jit, static_argnames=("epsilon", "alpha", "num_steps"))
def pgd_attack(
    state: TrainState,
    image: jnp.ndarray,
    label: jnp.ndarray,
    epsilon: float,
    alpha: float,
    num_steps: int,
) -> jnp.ndarray:
    """
    Performs a PGD (Projected Gradient Descent) L∞ attack on a single image.

    Starting from the clean image, takes `num_steps` gradient steps of size
    `alpha`, projecting back onto the L∞ ball of radius `epsilon` around the
    original image after each step.

    Args:
        state: Trained TrainState containing params and apply_fn.
        image: Clean input image, shape (H, W, C), values in [0, 1].
        label: True integer label.
        epsilon: L∞ perturbation budget.
        alpha: Step size per iteration (typically epsilon / 10).
        num_steps: Number of gradient steps.

    Returns:
        Adversarially perturbed image, clipped to [0, 1].
    """

    def loss_fn(img):
        logits = state.apply_fn({'params': state.params}, jnp.expand_dims(img, axis=0))
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=jnp.array([label])
        ).mean()

    def step(_, x_adv):
        grad = jax.grad(loss_fn)(x_adv)
        x_adv = x_adv + alpha * jnp.sign(grad)
        # Project onto the L∞ ball of radius epsilon around the original image
        x_adv = jnp.clip(x_adv, image - epsilon, image + epsilon)
        # Keep pixel values in valid range
        x_adv = jnp.clip(x_adv, 0, 1)
        return x_adv

    # jax.lax.fori_loop is JIT-compatible with static num_steps
    return jax.lax.fori_loop(0, num_steps, step, image)


@hydra.main(version_base=None, config_path="../conf", config_name="pgd")
def generate_pgd(cfg: DictConfig) -> None:
    """
    Runs a PGD attack on the test set and reports the attack success rate.
    """
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

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

    datasets = get_datasets()
    test_ds = datasets['test']
    test_images = test_ds['image']
    test_labels = test_ds['label']

    epsilon = cfg.epsilon
    alpha = cfg.alpha
    num_steps = cfg.num_steps

    # in_axes: state=broadcast, image=batch, label=batch, epsilon/alpha/num_steps=broadcast
    vmapped_pgd_attack = jax.vmap(pgd_attack, in_axes=(None, 0, 0, None, None, None))

    batch_size = cfg.batch_size
    num_test_images = len(test_images)
    num_batches = len(range(0, num_test_images, batch_size))

    total_successful_attacks = 0
    total_initially_correct = 0

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_path = os.path.join(output_dir, "pgd_examples")
    os.makedirs(save_path, exist_ok=True)
    log.info(f"Saving example images to: {save_path}")

    saved_count = 0
    save_limit = cfg.output_images

    log.info(
        f"Starting PGD attack on {num_test_images} test images with "
        f"epsilon={epsilon}, alpha={alpha}, num_steps={num_steps}..."
    )

    pbar = trange(num_batches, desc="Attacking batches")
    for i in pbar:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test_images)

        image_batch = test_images[start_idx:end_idx]
        label_batch = test_labels[start_idx:end_idx]

        original_logits = state.apply_fn({'params': state.params}, image_batch)
        original_preds = jnp.argmax(original_logits, axis=-1)

        correctly_classified_mask = (original_preds == label_batch)
        initially_correct_in_batch = correctly_classified_mask.sum()
        total_initially_correct += initially_correct_in_batch

        if initially_correct_in_batch == 0:
            continue

        attack_images = image_batch[correctly_classified_mask]
        attack_labels = label_batch[correctly_classified_mask]

        adversarial_batch = vmapped_pgd_attack(state, attack_images, attack_labels, epsilon, alpha, num_steps)

        adversarial_logits = state.apply_fn({'params': state.params}, adversarial_batch)
        adversarial_preds = jnp.argmax(adversarial_logits, axis=-1)

        successful_attacks_in_batch_mask = (adversarial_preds != attack_labels)
        successful_attacks_in_batch = successful_attacks_in_batch_mask.sum()
        total_successful_attacks += successful_attacks_in_batch

        if saved_count < save_limit and successful_attacks_in_batch > 0:
            successful_attack_indices = jnp.where(successful_attacks_in_batch_mask)[0]
            for idx in successful_attack_indices:
                if saved_count >= save_limit:
                    break
                original_img = attack_images[idx]
                adversarial_img = adversarial_batch[idx]

                orig_filename = f"original_{total_successful_attacks}_{saved_count}_label_{attack_labels[idx]}.png"
                save_image(original_img, os.path.join(save_path, orig_filename))

                adv_filename = f"adversarial_{total_successful_attacks}_{saved_count}_label_{adversarial_preds[idx]}.png"
                save_image(adversarial_img, os.path.join(save_path, adv_filename))

                saved_count += 1

    if total_initially_correct > 0:
        attack_success_rate = (total_successful_attacks / total_initially_correct) * 100
    else:
        attack_success_rate = 0.0

    log.info("--- PGD Attack Results ---")
    log.info(f"Total test images correctly classified by the model: {total_initially_correct}/{num_test_images}")
    log.info(f"Successful attacks (fooled the model): {total_successful_attacks}")
    log.info(f"Attack Success Rate: {attack_success_rate:.2f}%")


if __name__ == "__main__":
    generate_pgd()
