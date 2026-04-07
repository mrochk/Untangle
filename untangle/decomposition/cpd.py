import jax, jax.numpy as jnp
from functools import partial

from jaxtyping import jaxtyped, Array, Float
from beartype import beartype 
from beartype.typing import Tuple, Optional 

from untangle.utils import cpd_error, get_random_key
from untangle._ops import unfold_kolda
from untangle._common import init_cpd, solve_subproblem, normalize_columns_simple

@jaxtyped(typechecker=beartype)
def cpd(
    tensor: Float[Array, 'n m N'],
    rank: int,
    max_iters: int = 100,
    random_state: Optional[Array] = None,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    Float[Array, 'r'],
    Array,
]:
    log = lambda *args: print(*args) if verbose else None

    norm = jnp.linalg.norm(tensor)

    if random_state is None: random_state = get_random_key()

    factors, weights = init_cpd(tensor, rank, key=random_state), jnp.ones(rank)
    W, V, H = factors

    errors = []

    solve_subproblem_W = partial(solve_subproblem, unfolded=unfold_kolda(tensor, 0), mode=0)
    solve_subproblem_V = partial(solve_subproblem, unfolded=unfold_kolda(tensor, 1), mode=1)
    solve_subproblem_H = partial(solve_subproblem, unfolded=unfold_kolda(tensor, 2), mode=2)

    for iteration in range(max_iters):
        W, _ = normalize_columns_simple(solve_subproblem_W(W=W, V=V, H=H))
        V, _ = normalize_columns_simple(solve_subproblem_V(W=W, V=V, H=H))
        H, weights = normalize_columns_simple(solve_subproblem_H(W=W, V=V, H=H))

        factors = W, V, H
        error = cpd_error(tensor, factors, weights)

        if iteration > 0:
            diff = abs(error - errors[-1])

            log(f'iteration {iteration+1}: error = {error:.4f}, diff = {diff:.8f}')

            if diff < tol * norm: log(f'stopping at iteration {iteration+1}'); break

        else: log(f'iter {iteration+1}: error = {error:.4f}')

        errors.append(error)

    return factors, weights, jnp.array(errors)
