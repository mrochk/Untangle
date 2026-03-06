import jax, jax.numpy as jnp
from scipy.interpolate import BSpline, make_smoothing_spline
import matplotlib.pyplot as plt

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, relative_error, make_log
from untangle.algorithm.common import normalize_columns_V, fit_internals, make_g, inference
from untangle.decomposition.common import column_normalize

from .spline import fit_with_deriv
from scipy.interpolate import CubicSpline

@jaxtyped(typechecker=beartype)
def init_cmtf(tensor: Float[Array, 'n m N'], rank: int, key: Array):
    n, m, N = tensor.shape
    keys = jax.random.split(key, num=4)

    W = jax.random.normal(keys[0], shape=(n, rank))
    V = jax.random.normal(keys[1], shape=(m, rank))
    H = jax.random.normal(keys[2], shape=(N, rank))
    R = jax.random.normal(keys[3], shape=(N, rank))

    return W, V, H, R

@jax.jit(static_argnames=('lam',))
@jaxtyped(typechecker=beartype)
def cmtf_lstsq(X1, X2, Y1, Y2, lam: float):
    X = jnp.concatenate([X1, lam*X2], axis=0)
    Y = jnp.concatenate([Y1, lam*Y2], axis=0)
    return jnp.linalg.lstsq(X, Y)[0].T

@jaxtyped(typechecker=beartype)
def cmtf_ssd(
    J: Float[Array, 'n m N'],
    Y: Float[Array, 'N n'],
    X: Float[Array, 'N m'],
    rank: int,
    lam: float = 0.1,
    smoothing: float = 1.0,
    gamma: float = 1.0,
    max_iters: int = 50,
    random_state: Array = get_random_key(),
    verbose: int = 0,
):
    log = make_log(verbose, '<CMTF-SSD>: ')
    best_errors = []

    W, V, H, R = init_cmtf(J, rank, random_state)

    lstsq = jax.jit(jnp.linalg.lstsq)

    for iteration in range(max_iters):
        W = cmtf_lstsq(X1=khatri_rao(H, V), X2=R, Y1=unfold_kolda(J, 0).T, Y2=Y, lam=lam)
        V = lstsq(khatri_rao(H, W), unfold_kolda(J, 1).T)[0].T
        W, V = normalize_columns_V(W, V)

        H = lstsq(khatri_rao(V, W), unfold_kolda(J, 2).T)[0].T
        R = lstsq(W, Y.T)[0].T

        H, R, weights = smoothing_splines_projection(H, R, X @ V, gamma, smoothing)

        error = relative_error(J, (W, V, H), weights)

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {max_iters}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    g = make_g(fit_internals(X @ V, H, R, use='R'))
    return inference(W, V, g), best, jnp.array(best_errors)

def smoothing_splines_projection(H, R, U, gamma: float, smoothing: float):
    weights = []
    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        idx = jnp.argsort(u)
        us, hs, rs = u[idx], h[idx], r[idx]

        m = fit_with_deriv(us, rs, hs, gamma, smoothing)

        spline  = CubicSpline(us, m)
        dspline = spline.derivative()

        bias = (hs - dspline(us)).mean()

        H = H.at[:, rank].set(dspline(u) + bias)
        R = R.at[:, rank].set(spline(u))

        scale = jnp.linalg.norm(H[:, rank])
        R = R.at[:, rank].set(R[:, rank] / scale)
        H = H.at[:, rank].set(H[:, rank] / scale)

        weights.append(scale)

    return H, R, jnp.array(weights)
