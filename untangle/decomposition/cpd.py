import jax, jax.numpy as jnp
from functools import partial
from tqdm import tqdm

from jaxtyping import jaxtyped, Array, Float
from beartype import beartype 
from beartype.typing import Tuple, Optional 

from untangle.utils import cpd_error
from untangle._ops import unfold_kolda, normalize_columns_simple
from untangle._common import (
    make_log,
    initialize, 
    get_random_key,
    solve_cpd_subproblem, 
    cpd_stopping_criterion, 
)

@jaxtyped(typechecker=beartype)
def cpd(
    tensor: Float[Array, 'n m N'],
    rank: int,
    maxiters: int = 100,
    key: Optional[Array] = None,
    tol: float = 1e-8,
    verbose: int = 0,
) -> Tuple[
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    Float[Array, 'r'],
    Array,
]:
    if key is None: key = get_random_key()

    prefix = '|CPD| -> '
    log = make_log(verbose, prefix)

    errors = []

    norm = jnp.linalg.norm(tensor)

    factors, weights = initialize(tensor, rank, key), jnp.ones(rank)
    W, V, H = factors

    solve_W = partial(solve_cpd_subproblem, unfolded=unfold_kolda(tensor, 0), mode=0)
    solve_V = partial(solve_cpd_subproblem, unfolded=unfold_kolda(tensor, 1), mode=1)
    solve_H = partial(solve_cpd_subproblem, unfolded=unfold_kolda(tensor, 2), mode=2)

    bar = tqdm(range(maxiters), prefix)
    for iteration in bar:
        W, _       = normalize_columns_simple(solve_W(W=W, V=V, H=H))
        V, _       = normalize_columns_simple(solve_V(W=W, V=V, H=H))
        H, weights = normalize_columns_simple(solve_H(W=W, V=V, H=H))

        factors = W, V, H

        error = cpd_error(tensor, factors, weights)

        if iteration > 0:
            diff = abs(error - errors[-1])
            log(f'iteration {iteration+1}: error = {error:.4f}, diff = {diff:.8f}')

            if cpd_stopping_criterion(diff, tol, norm):
                bar.set_postfix_str(f'(Early stopping after {iteration+1} iterations.)')
                log(f'stopping at iteration {iteration+1}')
                errors.append(error)
                break

        else: log(f'iter {iteration+1}: error = {error:.4f}')
        errors.append(error)

    return factors, weights, jnp.array(errors)
