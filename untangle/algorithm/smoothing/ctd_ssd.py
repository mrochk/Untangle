import jax, jax.numpy as jnp
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

from scipy.interpolate import make_smoothing_spline

from untangle.algorithm import Decoupling
from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, relative_error, make_log
from scipy.interpolate import make_smoothing_spline
from untangle.algorithm.common import (
    normalize_columns_V, 
    fit_internals, 
    lstsq,
    initialize,
    make_internals, 
)
from .projection import smoothing_splines_projection
    
@jaxtyped(typechecker=beartype)
def ctd_ssd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    iterations: int = 20,
    key: Array = get_random_key(),
    verbose: int = 0,
) -> Decoupling:

    name = 'CTD-SSD'

    log = make_log(verbose, f'|{name}| -> ')
    best_errors = []

    W, V, H, R = initialize(J, rank, key, True)

    J1 = unfold_kolda(J, 0)
    J2 = unfold_kolda(J, 1)
    J3 = unfold_kolda(J, 2)

    for iteration in tqdm(range(iterations), desc=f'Computing {name}'):
        W = lstsq(khatri_rao(H, V), J1.T)
        V = lstsq(khatri_rao(H, W), J2.T)

        W, V = normalize_columns_V(W, V)

        H = lstsq(khatri_rao(V, W), J3.T)
        R = lstsq(W, Y.T)

        Z = X @ V

        H, R = smoothing_splines_projection(H, R, Z)

        error = relative_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {iterations}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    internals = make_internals(fit_internals(X @ V, H, R, use='H'))

    return Decoupling(best, internals)
    