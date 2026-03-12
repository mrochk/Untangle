import numpy as np
import jax, jax.numpy as jnp
from scipy.interpolate import make_smoothing_spline
from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
import matplotlib.pyplot as plt

from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, relative_error, make_log
from untangle.algorithm.common import (
    normalize_columns_V, 
    fit_internals, 
    make_g,
    inference,
    init_cmtf,
    cmtf_lstsq,
)
import untangle.algorithm.cmtf_ssd.spline as spline

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
    log = make_log(verbose, '|CMTF-SSD| -> ')
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

        log(f'Iteration [{iteration+1} / {max_iters}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    g = make_g(fit_internals(X @ V, H, R, use='H'))
    return inference(W, V, g), best, jnp.array(best_errors)

def smoothing_splines_projection(H, R, U, i: int = None):
    fig, ax = plt.subplots(H.shape[1], 2, figsize=(10, 20))

    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        idx = jnp.argsort(u)
        inv = jnp.argsort(idx)

        us, hs, rs = u[idx], h[idx], r[idx]

        hs_smooth = jnp.array(make_smoothing_spline(us, hs)(us))

        m, dm = spline.fit_smoothing_spline(us, rs, hs_smooth)

        H = H.at[:, rank].set(dm[inv])
        R = R.at[:, rank].set(m[inv])

        ax[rank, 0].scatter(u, r, color='blue')
        ax[rank, 0].scatter(u, h, color='red')
        ax[rank, 0].scatter(u, hs_smooth[inv], color='purple')
        ax[rank, 1].scatter(u, m[inv], color='blue')
        ax[rank, 1].scatter(u, dm[inv], color='red')
        ax[rank, 0].set_ylim((min(r.min(), h.min())-1, max(r.max(), h.max())+1))
        ax[rank, 1].set_ylim((min(r.min(), h.min())-1, max(r.max(), h.max())+1))
    
    plt.suptitle(f'Iteration #{i}')
    plt.tight_layout()
    fig.savefig(f'plots/iter{i}.png')
    plt.close()

    return H, R
