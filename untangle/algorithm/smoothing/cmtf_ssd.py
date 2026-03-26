import jax, jax.numpy as jnp
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

from untangle.algorithm import Decoupling
from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, cpd_error, make_log
from untangle.algorithm.smoothing.projection import smoothing_splines_projection
from untangle.algorithm.common import normalize_columns_V, fit_internals, cmtf_lstsq, lstsq, initialize, make_internals
    
@jaxtyped(typechecker=beartype)
def cmtf_ssd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    lam: float = 0.1,
    iterations: int = 20,
    key: Array = get_random_key(),
    verbose: int = 0,
) -> Decoupling:

    name = 'CMTF-SSD'

    log = make_log(verbose, f'|{name}| -> ')
    best_errors = []

    W, V, H, R = initialize(J, rank, key, True)

    J1 = unfold_kolda(J, 0).T
    J2 = unfold_kolda(J, 1).T
    J3 = unfold_kolda(J, 2).T

    for iteration in tqdm(range(iterations), desc=f'Computing {name}'):
        W = cmtf_lstsq(X1=khatri_rao(H, V), X2=R, Y1=J1, Y2=Y, lam=lam)
        V = lstsq(khatri_rao(H, W), J2)

        W, V = normalize_columns_V(W, V)

        H = lstsq(khatri_rao(V, W), J3)
        R = lstsq(W, Y.T)

        Z = X @ V

        H, R = smoothing_splines_projection(H, R, Z, lam, iteration+1)

        error = cpd_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {iterations}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    internals = make_internals(fit_internals(X @ V, H, R, use='H'))

    return Decoupling(best, internals)
    