import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Tuple, Optional, Any
from beartype import beartype
import warnings

from untangle.algorithm._cmtf import cmtf
from untangle import _ops as ops 
from untangle.algorithm import Decoupling
from untangle._common import *

@jaxtyped(typechecker=beartype)
def cmtf_psd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    niters: int = 100,
    gamma: float = 0.1,
    dof: Optional[int] = None,
    lam_nvalues: int = 100,
    lam_nvalues_init: int = 1000,
    degree: int = 3,
    key: Optional[Array] = None,
    show_progress: Optional[bool] = True,
) -> Tuple[Decoupling, Array]:

    if dof is None: dof = default_dof(X.shape[0])
    if key is None: key = get_random_key()

    proj_params = dict(
        dof=dof, 
        gamma=gamma, 
        degree=degree, 
        lam_nvalues=lam_nvalues, 
        lam_nvalues_init=lam_nvalues_init,
    )

    factors, (_, coefs, knots), error = cmtf(
        X, Y, J, rank, niters, gamma,
        pspl_project, proj_params,
        key, '|CMTF-PSD|',
        show_progress=show_progress,
    )

    internals = fit_internals_with_best_coefs(coefs, knots, degree)
    return (Decoupling(factors, make_internals(internals)), error)

def pspl_project(
    H: Float[Array, 'N r'],
    R: Float[Array, 'N r'],
    Z: Float[Array, 'N r'],
    out: Any, dof: int, degree: int, gamma: float,
    lam_nvalues: int,
    lam_nvalues_init: int,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    lam_in = [None for _ in range(H.shape[1])] if out is None else out[0]
    (lam_out, coefs_out, knots_out) = [], [], []

    for rank in range(H.shape[1]):
        (z, h, r) = Z[:, rank], H[:, rank], R[:, rank]

        is_degenerate = (jnp.max(z) - jnp.min(z)) < 1e-6

        if is_degenerate:
            warnings.warn(f'Internal {rank} is degenerate (max - min < 1e-6).')
            H = H.at[:, rank].set(jnp.zeros_like(H[:, rank]))
            R = R.at[:, rank].set(jnp.zeros_like(R[:, rank]))
            lam_out.append(lam_in[rank]); coefs_out.append(None); knots_out.append(None)
            continue

        knots = determine_knots(z, dof, degree, 'even')
        B = get_design_matrix(z, knots, degree)
        dB = get_design_dmatrix(z, knots, degree)
        D = ops.second_diff_matrix(B.shape[1])
        A = jnp.vstack([dB, jnp.sqrt(gamma)*B])
        y = jnp.concatenate([h, jnp.sqrt(gamma)*r])
        ll = gcv_grid_search(A, y, D, len(z), lam_in[rank], lam_nvalues_init, lam_nvalues)

        lam_out.append(ll)
        lam = 10**ll

        A = jnp.concatenate([A, jnp.sqrt(lam) * D])
        y = jnp.concatenate([y, jnp.zeros(D.shape[0])])
        coefs = ops.lstsq(A, y).T
        (H, R) = bspline_project(rank, coefs, B, dB, H, R)

        # return the coefs for fitting the internals later
        coefs_out.append(ops.lstsq(B, R[:, rank]).T)
        knots_out.append(knots)

    return H, R, (jnp.array(lam_out), coefs_out, knots_out)

def gcv_grid_search(
    X: Array, 
    y: Array, 
    D: Array, 
    n: int, 
    _ll: Optional[float],
    nvalues_init: int,
    nvalues: int,
) -> Array:

    y = jnp.concatenate([y, jnp.zeros(D.shape[0])])

    if _ll is None:
        lls_init = jnp.linspace(-6, 3, nvalues_init)
        scores = jax.vmap(lambda ll: gcv_score(ll, X, D, y, n))(lls_init)
        _ll = lls_init[jnp.argmin(scores)]

    lls = jnp.linspace(_ll-1, _ll+1, nvalues)
    scores = jax.vmap(lambda ll: gcv_score(ll, X, D, y, n))(lls)
    return lls[jnp.argmin(scores)]

@jax.jit(static_argnames='n')
def gcv_score(ll: Array, X: Array, D: Array, y: Array, n: int) -> Array:
    lam = 10.0 ** ll
    X = jnp.concatenate([X, jnp.sqrt(lam)*D])
    coefs = ops.lstsq(X, y).T
    residuals = y[:2*n] - X[:2*n] @ coefs
    rss = jnp.sum(residuals ** 2)
    Q = jnp.linalg.qr(X)[0]
    df = jnp.trace(Q[:2*n] @ Q[:2*n].T)
    return (n * rss) / (n - df)**2
