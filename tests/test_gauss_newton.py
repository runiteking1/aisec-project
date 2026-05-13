"""
Tests for the Generalized Gauss-Newton (GGN) optimizer in src/train.py.

The GGN computation (train_step with gauss_newton != None) does:
  1. Compute J: per-sample Jacobian of model outputs w.r.t. all params,
     flattened and reshaped to (bsz * num_classes, P).
  2. Build GGN = sum_i J_i^T H_i J_i, where H_i = diag(p_i) - p_i p_i^T
     is the per-sample softmax Hessian.
  3. Solve (GGN + lambda*I) x = g and use x as the preconditioned gradient.
"""

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import pytest
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from src.train import _gauss_newton_step


class _TinyLinear(nn.Module):
    """Single Dense layer with no nonlinearity.

    For a linear model, GGN == exact Hessian of the cross-entropy loss,
    which lets us write a closed-form ground-truth comparison.
    """
    num_classes: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.num_classes)(x.reshape((x.shape[0], -1)))


@pytest.fixture(scope="module")
def ctx():
    key = jax.random.PRNGKey(42)
    num_classes = 3
    bsz = 5
    input_dim = 4

    model = _TinyLinear(num_classes=num_classes)
    key, img_key, param_key = jax.random.split(key, 3)
    images = jax.random.normal(img_key, (bsz, input_dim))
    labels = jnp.array([0, 1, 2, 0, 1])
    params = model.init(param_key, images)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(1.0))
    logits = model.apply({'params': params}, images)

    def mean_loss(p):
        lgs = model.apply({'params': p}, images)
        one_hot = jax.nn.one_hot(labels, num_classes)
        return -jnp.sum(one_hot * jax.nn.log_softmax(lgs)) / bsz

    grads = jax.grad(mean_loss)(params)

    return dict(
        model=model, params=params, images=images,
        labels=labels, num_classes=num_classes, bsz=bsz,
        state=state, logits=logits, grads=grads,
    )


def test_ggn_is_symmetric(ctx):
    """GGN = J^T H J must be symmetric since H is symmetric.

    Tolerance is relaxed to 1e-3 due to float32 rounding in the einsum.
    """
    _, ggn = _gauss_newton_step(
        ctx['state'], ctx['images'], ctx['logits'],
        ctx['grads'], ctx['num_classes'], lam=0.01, return_ggn=True,
    )
    np.testing.assert_allclose(np.array(ggn), np.array(ggn).T, atol=1e-3)


def test_ggn_is_psd(ctx):
    """GGN must be positive semi-definite since H = diag(p) - p p^T is PSD.

    Tolerance is relaxed to 1e-3 due to float32 rounding in eigvalsh.
    """
    _, ggn = _gauss_newton_step(
        ctx['state'], ctx['images'], ctx['logits'],
        ctx['grads'], ctx['num_classes'], lam=0.01, return_ggn=True,
    )
    eigenvalues = np.linalg.eigvalsh(np.array(ggn))
    assert np.all(eigenvalues >= -1e-3), (
        f"GGN has negative eigenvalues (min={eigenvalues.min():.2e})"
    )


def test_ggn_equals_hessian_for_linear_model(ctx):
    """For a linear model, the GGN is the exact Hessian of the cross-entropy loss.

    GGN is computed as a sum over the batch; the Hessian is of the mean loss,
    so we compare ggn / bsz against jax.hessian of the mean loss.
    """
    model, params, images, labels, num_classes, bsz = (
        ctx['model'], ctx['params'], ctx['images'], ctx['labels'],
        ctx['num_classes'], ctx['bsz'],
    )
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    def mean_loss(fp):
        logits = model.apply({'params': unravel(fp)}, images)
        one_hot = jax.nn.one_hot(labels, num_classes)
        return -jnp.sum(one_hot * jax.nn.log_softmax(logits)) / bsz

    hessian = jax.hessian(mean_loss)(flat_params)

    _, ggn = _gauss_newton_step(
        ctx['state'], images, ctx['logits'],
        ctx['grads'], num_classes, lam=0.01, return_ggn=True,
    )
    np.testing.assert_allclose(np.array(ggn / bsz), np.array(hessian), atol=1e-4)


def test_gauss_newton_step_matches_direct_solve(ctx):
    """_gauss_newton_step output matches (GGN + lam*I)^{-1} g for several lam values.

    Uses return_ggn=True so the reference solve uses the exact same GGN matrix
    computed internally, avoiding any cross-implementation comparison.
    """
    g, _ = jax.flatten_util.ravel_pytree(ctx['grads'])

    for lam in (1e-3, 1e-2, 1e-1):
        result_pytree, ggn = _gauss_newton_step(
            ctx['state'], ctx['images'], ctx['logits'],
            ctx['grads'], ctx['num_classes'], lam=lam, return_ggn=True,
        )
        result_flat, _ = jax.flatten_util.ravel_pytree(result_pytree)
        P = ggn.shape[0]

        reference = jnp.linalg.solve(ggn + lam * jnp.eye(P), g)
        np.testing.assert_allclose(
            np.array(result_flat), np.array(reference), atol=1e-4,
            err_msg=f"Preconditioned gradient mismatch at lambda={lam}",
        )


def test_preconditioned_gradient_is_descent_direction(ctx):
    """(GGN + lambda*I)^{-1} g must satisfy dot(result, g) > 0.

    Since GGN + lambda*I is positive definite for any lambda > 0, the
    preconditioned gradient always points in a descent direction.
    """
    g, _ = jax.flatten_util.ravel_pytree(ctx['grads'])

    for lam in (1e-3, 1e-2, 1e-1):
        result_pytree = _gauss_newton_step(
            ctx['state'], ctx['images'], ctx['logits'],
            ctx['grads'], ctx['num_classes'], lam=lam,
        )
        result_flat, _ = jax.flatten_util.ravel_pytree(result_pytree)
        inner = float(jnp.dot(result_flat, g))
        assert inner > 0, (
            f"Preconditioned gradient is not a descent direction at lambda={lam} "
            f"(dot product={inner:.4f})"
        )
