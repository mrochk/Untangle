import jax, jax.numpy as jnp
from scipy.interpolate import BSpline, make_smoothing_spline
import matplotlib.pyplot as plt

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, relative_error, make_log
from untangle.algorithm.common import normalize_columns_V, fit_internals, make_g, inference
from untangle.decomposition.common import column_normalize

from .spline import fit_with_derivatives, find_lambda_with_derivatives
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

        H, R = smoothing_splines_projection(H, R, X @ V, iteration+1)

        error = relative_error(J, (W, V, H))

        if jnp.isnan(error):
            W, V, H, R = init_cmtf(J, rank, random_state)
            continue

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        #g = make_g(fit_internals(X @ V, H, R, use='R'))
        #inf = inference(W, V, g)
        #inference_error = jnp.linalg.norm(jnp.array([inf(x) for x in X]) - Y) / jnp.linalg.norm(Y)
        log(f'Iteration [{iteration+1} / {max_iters}]: error = {error:.4f}, best = {best_error:.4f}')
        #log(f'inference error = {inference_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    g = make_g(fit_internals(X @ V, H, R, use='H'))
    return inference(W, V, g), best, jnp.array(best_errors)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import numpy as np

def smoothing_splines_projection(H, R, U, i = None):
    gamma = 1.0

    rank = H.shape[1]
    fig, ax = plt.subplots(rank, 2, figsize=(10, 20))

    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        idx = jnp.argsort(u)
        inv = jnp.argsort(idx)

        ax[rank, 0].scatter(u, r, color='blue')
        ax[rank, 0].scatter(u, h, color='red')

        us, hs, rs = np.array(u[idx]), np.array(h[idx]), np.array(r[idx])

        dy_smooth = make_smoothing_spline(us, hs)(us)

        ax[rank, 0].scatter(u, dy_smooth[inv], color='purple')

        smoothing = find_lambda_with_derivatives(us, rs, dy_smooth, gamma)

        smoothing = min(smoothing, 1)

        m, dm, D = fit_with_derivatives(us, rs, dy_smooth, gamma, smoothing, dy_smooth)

        m = m + (rs - m).mean()
        dm = dm + (hs - dm).mean()

        ax[rank, 1].scatter(u, m[inv], color='blue')
        ax[rank, 1].scatter(u, dm[inv], color='red')
        
        ax[rank, 0].set_ylim((min(r.min(), h.min())-1, max(r.max(), h.max())+1))
        ax[rank, 1].set_ylim((min(r.min(), h.min())-1, max(r.max(), h.max())+1))

        H = H.at[:, rank].set(dm[inv])
        R = R.at[:, rank].set(m[inv])
    
    plt.suptitle(f'Iteration #{i}')
    plt.tight_layout()
    fig.savefig(f'iter{i}.png')
    plt.close()

    return H, R