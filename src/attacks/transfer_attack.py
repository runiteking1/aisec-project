"""
Transfer attack: generate PGD adversarial examples with a source model,
evaluate success on a separate target model. Used to test whether adversarial
robustness is genuine (transferable) vs. gradient-masking (source-specific).
"""
import logging
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from src.models import CNN
from src.train import get_datasets

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


@partial(jax.jit, static_argnames=("epsilon", "alpha", "num_steps"))
def pgd_attack(state: TrainState, image: jnp.ndarray, label: jnp.ndarray,
               epsilon: float, alpha: float, num_steps: int) -> jnp.ndarray:
    """Single-image PGD attack (L∞)."""

    def loss_fn(img):
        logits = state.apply_fn({'params': state.params}, jnp.expand_dims(img, 0))
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.array([label])
        ).mean()

    def step(_, x):
        grad = jax.grad(loss_fn)(x)
        x = x + alpha * jnp.sign(grad)
        x = jnp.clip(x, image - epsilon, image + epsilon)
        x = jnp.clip(x, 0, 1)
        return x

    return jax.lax.fori_loop(0, num_steps, step, image)


@hydra.main(version_base=None, config_path="../../conf", config_name="transfer_attack")
def run_transfer_attack(cfg: DictConfig) -> None:
    """
    Generates PGD adversarial examples using `source_checkpoint_path`,
    then measures how many of them fool `target_checkpoint_path`.

    Run twice with source/target swapped to get both transfer directions.
    """
    log.info(OmegaConf.to_yaml(cfg))

    source = load_model(cfg.source_checkpoint_path, cfg.data.num_classes)
    target = load_model(cfg.target_checkpoint_path, cfg.data.num_classes)

    test_ds = get_datasets()['test']
    images, labels = test_ds['image'], test_ds['label']

    epsilon   = cfg.epsilon
    alpha     = cfg.alpha
    num_steps = cfg.num_steps
    batch_size = cfg.batch_size
    N = len(images)
    num_batches = (N + batch_size - 1) // batch_size

    # vmap over the batch; state/hyperparams are broadcast
    vmapped_pgd = jax.vmap(pgd_attack, in_axes=(None, 0, 0, None, None, None))

    total_correct_source = 0
    total_correct_both   = 0   # correct on both source and target (denominator)
    total_transferred    = 0   # fooled target using source-generated adversarial

    for i in trange(num_batches, desc=f"Transfer (eps={epsilon})"):
        start = i * batch_size
        end   = min(start + batch_size, N)
        imgs  = images[start:end]
        lbls  = labels[start:end]

        # Identify images the source model gets right
        src_preds = jnp.argmax(source.apply_fn({'params': source.params}, imgs), -1)
        src_mask  = src_preds == lbls
        total_correct_source += int(src_mask.sum())
        if not src_mask.any():
            continue

        attack_imgs = imgs[src_mask]
        attack_lbls = lbls[src_mask]

        # Generate adversarial examples using the SOURCE model's gradients
        adv_imgs = vmapped_pgd(source, attack_imgs, attack_lbls, epsilon, alpha, num_steps)

        # Only count images that the TARGET also got right on the clean input
        tgt_clean_preds = jnp.argmax(target.apply_fn({'params': target.params}, attack_imgs), -1)
        tgt_mask = tgt_clean_preds == attack_lbls
        total_correct_both += int(tgt_mask.sum())

        # Evaluate transferred adversarial examples on the TARGET
        tgt_adv_preds = jnp.argmax(target.apply_fn({'params': target.params}, adv_imgs), -1)
        transferred = ((tgt_adv_preds != attack_lbls) & tgt_mask).sum()
        total_transferred += int(transferred)

    rate = (total_transferred / total_correct_both * 100) if total_correct_both > 0 else 0.0

    log.info("--- Transfer Attack Results ---")
    log.info(f"  epsilon={epsilon}, alpha={alpha}, steps={num_steps}")
    log.info(f"  Source correctly classified:          {total_correct_source}/{N}")
    log.info(f"  Target correct on same images:        {total_correct_both}")
    log.info(f"  Successful transfers (target fooled): {total_transferred}")
    log.info(f"  Transfer Success Rate:                {rate:.2f}%")


if __name__ == "__main__":
    run_transfer_attack()
