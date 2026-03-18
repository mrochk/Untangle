import jax, jax.numpy as jnp

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

from tqdm import tqdm

from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, relative_error, make_log
from untangle.algorithm.common import (
    initialize,
    normalize_columns_V,
    fit_internals,
    make_internals,
    inference,
)

from untangle.algorithm.smoothing.projection import smoothing_splines_projection

@jaxtyped(typechecker=beartype)
def ctd_ssd(
    J: Float[Array, 'n m N'],
    Y: Float[Array, 'N n'],
    X: Float[Array, 'N m'],
    rank: int,
    use: str = 'first',
    iterations: int = 50,
    random_state: Array = get_random_key(),
    verbose: int = 0,
):
    name = 'CTD-SSD'
    log = make_log(verbose, f'|{name}| -> ')
    best_errors = []

    W, V, H, R = initialize(J, rank, random_state, with_R=True)

    lstsq = jax.jit(lambda X, Y: jnp.linalg.lstsq(X, Y)[0].T)

    for iteration in tqdm(range(iterations), desc=f'Computing {name}'):

        W = lstsq(khatri_rao(H, V), unfold_kolda(J, 0).T)
        V = lstsq(khatri_rao(H, W), unfold_kolda(J, 1).T)
        W, V = normalize_columns_V(W, V)

        H = lstsq(khatri_rao(V, W), unfold_kolda(J, 2).T)
        R = lstsq(W, Y.T)

        U = X @ V

        H, R = smoothing_splines_projection(
            H, R, U, 
            use=use,
            plot=True,
            plot_title=f'Iteration {iteration+1}',
            plot_path=f'plots/iter{iteration+1}.png',
        )

        error = relative_error(J, (W, V, H))

        if jnp.isnan(error):
            log('error is nan')
            W, V, H, R = initialize(J, rank, random_state)
            continue

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {iterations}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    g = make_internals(fit_internals(X @ V, H, R, use='H'))
    return inference(W, V, g), best, jnp.array(best_errors)
