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

    return dict(
        model=model, params=params, images=images,
        labels=labels, num_classes=num_classes, bsz=bsz,
    )


def _compute_ggn(model, params, images, num_classes):
    """Replicates the GGN matrix computation from train_step verbatim."""
    bsz = images.shape[0]
    logits = model.apply({'params': params}, images)
    probs = jax.nn.softmax(logits, axis=-1)

    jacobian = jax.jacobian(lambda p: model.apply({'params': p}, images))(params)
    flattened, _ = jax.tree.flatten(jacobian)
    reshaped_leaves = [leaf.reshape(bsz * num_classes, 1, -1) for leaf in flattened]
    aggregated = jnp.concatenate(reshaped_leaves, axis=-1)
    J = aggregated.reshape(bsz * num_classes, -1)
    P = J.shape[1]

    h_blocks = jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)
    J_split = J.reshape(bsz, num_classes, P)
    ggn = jnp.einsum('bni,bnm,bmj->ij', J_split, h_blocks, J_split)
    return J, ggn


def test_jacobian_matches_finite_differences(ctx):
    """Jacobian reshaping in the GGN code produces the correct (bsz*C, P) matrix.

    Verified by comparing against a central finite-difference Jacobian.
    Any axis-swap or leaf-ordering bug in the reshape would show up here.
    """
    model, params, images, num_classes, bsz = (
        ctx['model'], ctx['params'], ctx['images'],
        ctx['num_classes'], ctx['bsz'],
    )
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)
    P = int(flat_params.shape[0])
    eps = 1e-3

    fd_J = np.zeros((bsz * num_classes, P))
    for i in range(P):
        delta = jnp.zeros(P).at[i].set(eps)
        out_p = model.apply({'params': unravel(flat_params + delta)}, images)
        out_m = model.apply({'params': unravel(flat_params - delta)}, images)
        fd_J[:, i] = np.array(((out_p - out_m) / (2 * eps)).reshape(-1))

    analytic_J, _ = _compute_ggn(model, params, images, num_classes)
    # float32 central FD has O(eps * machine_eps / eps) ~ 1e-3 cancellation error
    np.testing.assert_allclose(np.array(analytic_J), fd_J, atol=1e-3)


def test_ggn_is_symmetric(ctx):
    """GGN = J^T H J must be symmetric since H is symmetric.

    Tolerance is relaxed to 1e-3 due to float32 rounding in the einsum.
    """
    _, ggn = _compute_ggn(ctx['model'], ctx['params'], ctx['images'], ctx['num_classes'])
    np.testing.assert_allclose(np.array(ggn), np.array(ggn).T, atol=1e-3)


def test_ggn_is_psd(ctx):
    """GGN must be positive semi-definite since H = diag(p) - p p^T is PSD.

    Tolerance is relaxed to 1e-3 due to float32 rounding in eigvalsh.
    """
    _, ggn = _compute_ggn(ctx['model'], ctx['params'], ctx['images'], ctx['num_classes'])
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
    _, ggn = _compute_ggn(model, params, images, num_classes)

    np.testing.assert_allclose(np.array(ggn / bsz), np.array(hessian), atol=1e-4)


def test_preconditioned_gradient_is_descent_direction(ctx):
    """(GGN + lambda*I)^{-1} g must satisfy dot(result, g) > 0.

    Since GGN + lambda*I is positive definite for any lambda > 0, the
    preconditioned gradient always points in a descent direction.
    """
    model, params, images, labels, num_classes, bsz = (
        ctx['model'], ctx['params'], ctx['images'], ctx['labels'],
        ctx['num_classes'], ctx['bsz'],
    )

    def mean_loss(p):
        logits = model.apply({'params': p}, images)
        one_hot = jax.nn.one_hot(labels, num_classes)
        return -jnp.sum(one_hot * jax.nn.log_softmax(logits)) / bsz

    grad_pytree = jax.grad(mean_loss)(params)
    g, _ = jax.flatten_util.ravel_pytree(grad_pytree)

    _, ggn = _compute_ggn(model, params, images, num_classes)
    P = ggn.shape[0]

    for lam in (1e-3, 1e-2, 1e-1):
        precond_g = jnp.linalg.solve(ggn + lam * jnp.eye(P), g)
        inner = float(jnp.dot(precond_g, g))
        assert inner > 0, (
            f"Preconditioned gradient is not a descent direction at lambda={lam} "
            f"(dot product={inner:.4f})"
        )