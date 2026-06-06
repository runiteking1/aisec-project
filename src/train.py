from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import optax
import hydra
import orbax.checkpoint as ocp
import logging
from jax import Array
from jax.scipy.sparse.linalg import cg as jax_cg
from omegaconf import DictConfig, OmegaConf
from flax.training.train_state import TrainState

from src.models import CNN, VisionTransformer
from src.data import get_datasets, get_cifar10_datasets
from src.plotting import plot_training_metrics

log = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=(2, 4, 5, 6))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_classes: int,
    gauss_newton: float = None,
    solve_method: str = 'direct',
    cg_steps: int = 150,
    use_line_search: bool = False,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    images, labels = batch

    def cross_entropy_loss(params, batch):
        imgs, lbls = batch
        logits = state.apply_fn({'params': params}, imgs)
        one_hot = jax.nn.one_hot(lbls, num_classes=num_classes)
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits)) / len(imgs)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == lbls)
        return loss, (logits, accuracy)

    (loss, (logits, accuracy)), grads = jax.value_and_grad(
        cross_entropy_loss, has_aux=True
    )(state.params, batch)

    if gauss_newton is not None:
        raw_grads = grads
        d = _gauss_newton_step(
            state, images, logits, grads, num_classes, gauss_newton,
            solve_method=solve_method, cg_steps=cg_steps,
        )
        if use_line_search:
            g_flat, _ = jax.flatten_util.ravel_pytree(raw_grads)
            d_flat, _ = jax.flatten_util.ravel_pytree(d)
            g_dot_d = jnp.dot(g_flat, d_flat)

            def ls_cond(carry):
                alpha, k = carry
                trial_params = jax.tree.map(lambda p, di: p - alpha * di, state.params, d)
                trial_logits = state.apply_fn({'params': trial_params}, images)
                one_hot = jax.nn.one_hot(labels, num_classes)
                f_new = -jnp.sum(one_hot * jax.nn.log_softmax(trial_logits)) / images.shape[0]
                return (f_new > loss - 1e-4 * alpha * g_dot_d) & (k < 10)

            def ls_body(carry):
                alpha, k = carry
                return alpha * 0.5, k + 1

            alpha, _ = jax.lax.while_loop(ls_cond, ls_body, (jnp.array(1.0), jnp.array(0)))
            new_params = jax.tree.map(lambda p, di: p - alpha * di, state.params, d)
            return state.replace(params=new_params, step=state.step + 1), loss, accuracy

        return state.apply_gradients(grads=d), loss, accuracy

    return state.apply_gradients(grads=grads), loss, accuracy


def _gauss_newton_step(
    state: TrainState,
    images: jnp.ndarray,
    logits: jnp.ndarray,
    grads,
    num_classes: int,
    lam: float,
    return_ggn: bool = False,
    solve_method: str = 'direct',
    cg_steps: int = 150,
):
    """Replace grads with (GGN + lam*I)^{-1} g (Levenberg-Marquardt step).

    return_ggn=True makes the function return (preconditioned_grads, ggn_matrix)
    instead of just preconditioned_grads, for testing and debugging.

    solve_method='cg' uses conjugate gradients (jax.scipy.sparse.linalg.cg)
    instead of a dense direct solve, which can scale better to larger models.
    """
    bsz = images.shape[0]

    # J: full Jacobian (bsz*num_classes, P)
    jac_pytree = jax.jacobian(lambda p: state.apply_fn({'params': p}, images))(state.params)
    leaves, _ = jax.tree.flatten(jac_pytree)
    reshaped = [leaf.reshape(bsz * num_classes, 1, -1) for leaf in leaves]
    J = jnp.concatenate(reshaped, axis=-1).reshape(bsz * num_classes, -1)
    P = J.shape[1]

    # GGN of the MEAN loss = (1/bsz) sum_i J_i^T H_i J_i,  H_i = diag(p_i) - p_i p_i^T.
    # The 1/bsz keeps the GGN at the same (mean) scale as the gradient `grads`
    # below, which is the gradient of the mean cross-entropy loss. Without it the
    # damping `lam` and the effective step size would both depend on batch size.
    probs = jax.nn.softmax(logits, axis=-1)
    h_blocks = jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)
    J_split = J.reshape(bsz, num_classes, P)
    ggn = jnp.einsum('bni,bnm,bmj->ij', J_split, h_blocks, J_split) / bsz

    g, unravel = jax.flatten_util.ravel_pytree(grads)
    A = ggn + lam * jnp.eye(P)
    if solve_method == 'cg':
        altered, _ = jax_cg(lambda v: A @ v, g, maxiter=cg_steps, tol=1e-6)
    else:
        altered = jnp.linalg.solve(A, g)

    if return_ggn:
        return unravel(altered), ggn
    return unravel(altered)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")
    log.info(f"Using {jax.default_backend()}")

    key = jax.random.PRNGKey(cfg.seed)

    if hasattr(cfg.model, 'model_type') and cfg.model.model_type == 'vit':
        model = VisionTransformer(
            patch_size=cfg.model.patch_size,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            mlp_dim=cfg.model.mlp_dim,
            dropout_rate=cfg.model.dropout_rate,
            num_classes=cfg.data.num_classes,
        )
    else:
        model = CNN(
            features_per_layer=tuple(cfg.model.features_per_layer),
            kernel_size=tuple(cfg.model.kernel_size),
            dense_features=tuple(cfg.model.dense_features),
            num_classes=cfg.data.num_classes,
        )

    gn_param = None
    gn_solve = 'direct'
    gn_cg_steps = 150
    gn_line_search = False
    if cfg.training.optimizer == 'adam':
        optimizer = optax.adam(learning_rate=cfg.training.learning_rate)
    elif cfg.training.optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=cfg.training.learning_rate)
    elif cfg.training.optimizer == 'gn':
        optimizer = optax.sgd(learning_rate=cfg.training.learning_rate)
        gn_param = cfg.training.gn_param
        gn_solve = getattr(cfg.training, 'gn_solve', 'direct')
        gn_cg_steps = getattr(cfg.training, 'gn_cg_steps', 150)
        gn_line_search = getattr(cfg.training, 'gn_line_search', False)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}.")

    if cfg.data.name == 'cifar10':
        datasets = get_cifar10_datasets(cfg.data.num_classes)
    elif cfg.data.name == 'mnist':
        datasets = get_datasets(cfg.data.poison, cfg.seed)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}.")

    train_ds = datasets['train']
    test_ds = datasets['test']

    dummy_input = jnp.ones((1, cfg.data.image_size, cfg.data.image_size, cfg.data.channels))
    params = model.init(key, dummy_input)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    log.info(f"Total parameters: {param_count}")

    num_train = len(train_ds['image'])
    num_test = len(test_ds['image'])
    batch_size = cfg.training.batch_size
    batches_per_epoch = num_train // batch_size
    log.info(f"Train {num_train} | Test {num_test} | Steps/epoch {batches_per_epoch}")

    test_accuracies = np.zeros(cfg.training.epochs)
    all_batch_losses = []
    all_batch_accuracies = []

    for epoch in range(cfg.training.epochs):
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, num_train)

        epoch_loss = epoch_accuracy = 0.0
        num_batches = 0

        for i in range(0, num_train, batch_size):
            batch = (train_ds['image'][indices[i:i + batch_size]],
                     train_ds['label'][indices[i:i + batch_size]])
            state, loss, accuracy = train_step(
                state, batch, cfg.data.num_classes,
                gauss_newton=gn_param,
                solve_method=gn_solve,
                cg_steps=gn_cg_steps,
                use_line_search=gn_line_search,
            )
            all_batch_losses.append(loss)
            all_batch_accuracies.append(accuracy)
            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1

        correct = 0
        for i in range(0, num_test, batch_size):
            image_batch = test_ds['image'][i:i + batch_size]
            label_batch = test_ds['label'][i:i + batch_size]
            logits = state.apply_fn({'params': state.params}, image_batch)
            correct += int(jnp.sum(jnp.argmax(logits, -1) == label_batch))
        test_accuracies[epoch] = correct / num_test

        log.info(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"Loss: {epoch_loss / num_batches:.4f} | "
            f"Train Acc: {epoch_accuracy / num_batches:.4f} | "
            f"Test Acc: {test_accuracies[epoch]:.4f}"
        )

    import os
    np.savez_compressed(
        os.path.join(output_dir, 'training_metrics.npz'),
        batch_losses=np.array(all_batch_losses),
        batch_accuracies=np.array(all_batch_accuracies),
        test_accuracies=test_accuracies,
    )
    plot_training_metrics(
        np.array(all_batch_accuracies), np.array(all_batch_losses),
        test_accuracies, batches_per_epoch, output_dir,
    )

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        os.path.join(output_dir, 'checkpoint', 'model'),
        item={'state': state, 'config': OmegaConf.to_container(cfg, resolve=True)},
    )
    log.info("Checkpoint saved.")


if __name__ == "__main__":
    train()
