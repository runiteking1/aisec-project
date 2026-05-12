"""
Square Attack: black-box L∞ adversarial attack (Andriushchenko et al., 2020).

Queries only model outputs (scores/logits), not gradients. Demonstrates
whether adversarial robustness holds when the attacker has no gradient
access — i.e., rules out gradient masking as an explanation.
"""
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
    img_scaled = (np.array(image_array) * 255).astype(np.uint8).reshape(28, 28)
    Image.fromarray(img_scaled).save(path)


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


@partial(jax.jit, static_argnames=("epsilon", "num_queries"))
def square_attack(state: TrainState, image: jnp.ndarray, label: jnp.ndarray,
                  epsilon: float, num_queries: int, key: jnp.ndarray) -> jnp.ndarray:
    """
    L∞ Square Attack on a single image (Andriushchenko et al. 2020).

    Each iteration proposes a uniformly-signed p×p square perturbation
    and accepts it if the model's loss on the true class increases.
    The square size p follows a decreasing schedule over the query budget.

    Args:
        state:       Trained model (used for score queries only, no gradients).
        image:       Clean image (H, W, C) in [0, 1].
        label:       True integer label.
        epsilon:     L∞ perturbation budget.
        num_queries: Total number of score queries.
        key:         JAX PRNG key.

    Returns:
        Best adversarial image found within the query budget.
    """
    H, W, C = image.shape

    def score(x):
        """CE loss on the true class — higher means a better attack."""
        logits = state.apply_fn({'params': state.params}, jnp.expand_dims(x, 0))
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.array([label])
        ).squeeze()

    # Initialise on the boundary: random ±epsilon stripe pattern (one sign per column)
    key, k0 = jax.random.split(key)
    init_signs = epsilon * jax.random.choice(k0, jnp.array([-1., 1.]), shape=(1, W, C))
    x_adv = jnp.clip(image + init_signs, 0, 1)
    init_score = score(x_adv)

    def step(carry, i):
        x_adv, best_score, key = carry
        key, k1, k2, k3 = jax.random.split(key, 4)

        # Decreasing square size: large at start, small at end
        # p_frac goes from 0.8 to 0.1 linearly over the budget
        p_frac = 0.8 - 0.7 * i / num_queries
        # p: side length of the square (at least 1)
        p = jnp.maximum(1, jnp.floor(p_frac * jnp.sqrt(H * W)).astype(jnp.int32))

        # Random top-left corner, clamped so the square stays inside the image
        r_h = jnp.minimum(jax.random.randint(k1, (), 0, H), H - p)
        r_w = jnp.minimum(jax.random.randint(k2, (), 0, W), W - p)

        # One ±epsilon sign per channel, uniform across the entire square
        # shape: (1, 1, C) — broadcasts over (H, W, C)
        sign = epsilon * jax.random.choice(k3, jnp.array([-1., 1.]), shape=(1, 1, C))

        # Boolean mask for the p×p square region — shape: (H, W, 1)
        h_idx = jnp.arange(H)
        w_idx = jnp.arange(W)
        mask = (
            (h_idx[:, None] >= r_h) & (h_idx[:, None] < r_h + p) &
            (w_idx[None, :] >= r_w) & (w_idx[None, :] < r_w + p)
        )[:, :, None]  # (H, W, 1)

        # Proposed image: apply sign perturbation inside the square
        x_new = jnp.where(mask, image + sign, x_adv)
        # Project onto L∞ ball and valid pixel range
        x_new = jnp.clip(x_new, image - epsilon, image + epsilon)
        x_new = jnp.clip(x_new, 0, 1)

        new_score = score(x_new)
        # Greedy accept: keep x_new if it scores higher
        accept = new_score > best_score
        x_adv = jnp.where(accept, x_new, x_adv)
        best_score = jnp.maximum(best_score, new_score)

        return (x_adv, best_score, key), None

    (x_adv, _, _), _ = jax.lax.scan(step, (x_adv, init_score, key), jnp.arange(num_queries))
    return x_adv


@hydra.main(version_base=None, config_path="../../conf", config_name="square_attack")
def run_square_attack(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    state = load_model(cfg.checkpoint_path, cfg.data.num_classes)

    test_ds = get_datasets()['test']
    test_images = test_ds['image']
    test_labels = test_ds['label']

    epsilon    = cfg.epsilon
    num_queries = cfg.num_queries
    batch_size  = cfg.batch_size
    N = len(test_images)
    num_batches = (N + batch_size - 1) // batch_size

    # Per-image PRNG keys — required because Square Attack is randomised
    root_key = jax.random.PRNGKey(cfg.seed)

    # vmap: broadcast state/epsilon/num_queries, map over (image, label, key)
    vmapped_square = jax.vmap(square_attack, in_axes=(None, 0, 0, None, None, 0))

    save_path = os.path.join(output_dir, "square_examples")
    os.makedirs(save_path, exist_ok=True)
    log.info(f"Saving example images to: {save_path}")

    total_initially_correct = 0
    total_successful = 0
    saved_count = 0
    save_limit = cfg.output_images

    log.info(
        f"Starting Square Attack on {N} images, "
        f"epsilon={epsilon}, num_queries={num_queries}..."
    )

    for i in trange(num_batches, desc="Attacking batches"):
        start = i * batch_size
        end   = min(start + batch_size, N)
        imgs  = test_images[start:end]
        lbls  = test_labels[start:end]

        # Only attack images the model gets right
        preds = jnp.argmax(state.apply_fn({'params': state.params}, imgs), -1)
        correct_mask = preds == lbls
        total_initially_correct += int(correct_mask.sum())
        if not correct_mask.any():
            continue

        attack_imgs = imgs[correct_mask]
        attack_lbls = lbls[correct_mask]

        # Generate a unique key per image in this batch
        root_key, batch_key = jax.random.split(root_key)
        keys = jax.random.split(batch_key, len(attack_imgs))

        adv_imgs = vmapped_square(state, attack_imgs, attack_lbls, epsilon, num_queries, keys)

        adv_preds = jnp.argmax(state.apply_fn({'params': state.params}, adv_imgs), -1)
        success_mask = adv_preds != attack_lbls
        total_successful += int(success_mask.sum())

        if saved_count < save_limit and success_mask.any():
            for idx in jnp.where(success_mask)[0]:
                if saved_count >= save_limit:
                    break
                save_image(attack_imgs[idx],
                           os.path.join(save_path, f"original_{saved_count}_label_{attack_lbls[idx]}.png"))
                save_image(adv_imgs[idx],
                           os.path.join(save_path, f"adversarial_{saved_count}_pred_{adv_preds[idx]}.png"))
                saved_count += 1

    rate = (total_successful / total_initially_correct * 100) if total_initially_correct > 0 else 0.0

    log.info("--- Square Attack Results ---")
    log.info(f"  epsilon={epsilon}, num_queries={num_queries}")
    log.info(f"  Correctly classified before attack: {total_initially_correct}/{N}")
    log.info(f"  Successful attacks:                 {total_successful}")
    log.info(f"  Attack Success Rate:                {rate:.2f}%")


if __name__ == "__main__":
    run_square_attack()
