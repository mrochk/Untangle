import jax, jax.numpy as jnp
from functools import partial
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Tuple, Optional

from untangle.algorithm import Decoupling
from untangle.utils import cpd_error
from untangle import _ops as ops
from untangle._common import (
    get_random_key,
    make_log,
    bspline_project,
    fit_internals, 
    initialize,
    make_internals, 
    determine_knots_quantiles,
    get_design_matrix, 
    get_design_dmatrix,
)

@jaxtyped(typechecker=beartype)
def cmtf_bsd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    niters: int = 100,
    dof: Optional[int] = None,
    gamma: float = 0.1,
    degree: int = 3,
    verbose: int = 0,
    key: Optional[Array] = None,
) -> Decoupling:

    if key is None: key = get_random_key() 
    if dof is None: dof = X.shape[0] // 2

    prefix = '|CMTF-BSD|'
    log = make_log(verbose, prefix)

    best_errors = []

    W, V, H, R = initialize(J, rank, key, with_R=True)

    J0 = ops.unfold_kolda(J, 0).T
    J1 = ops.unfold_kolda(J, 1).T
    J2 = ops.unfold_kolda(J, 2).T

    for iteration in tqdm(range(niters), desc=prefix):
        W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, Y, gamma)
        V = ops.lstsq(ops.khatri_rao(H, W), J1)
        W, V = ops.normalize_columns_V(W, V)

        H = ops.lstsq(ops.khatri_rao(V, W), J2)
        R = ops.lstsq(W, Y.T)

        Z = X @ V
        H, R = bsplines_projection(H, R, Z, dof, degree, gamma)

        error = cpd_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {niters}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    Z = X @ V

    internals = make_internals(fit_internals(Z, H, R))

    return Decoupling(best, internals)

def bsplines_projection(
    H: Float[Array, 'N r'],
    R: Float[Array, 'N r'],
    U: Float[Array, 'N r'],
    dof: int, 
    degree: int,
    gamma: float,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        knots = determine_knots_quantiles(u, dof, degree)

        B = get_design_matrix(u, knots, degree)
        dB = get_design_dmatrix(u, knots, degree)

        coefs = ops.cmtf_lstsq(dB, B, h, r, gamma)
        H, R = bspline_project(rank, coefs, B, dB, H, R)

    return H, R
