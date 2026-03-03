import jax, jax.numpy as jnp
from functools import partial

from jaxtyping import jaxtyped, Array, Float
from beartype import beartype 
from beartype.typing import Tuple, Optional

from untangle.utils import relative_error, get_random_key
from untangle.ops import unfold_kolda, khatri_rao, reshape, block_diag, vandermonde_matrix
from untangle.decomposition.common import init_cpd, solve_subproblem
from untangle.algorithm.common import normalize_columns_V

@jaxtyped(typechecker=beartype)
def cpd_constrained(
    tensor: Float[Array, 'n m N'],
    X: Float[Array, 'N m'],
    rank: int,
    degree: int,
    max_iters: int = 100,
    random_state: Optional[Array] = None,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    Float[Array, 'd r'], 
    Array,
]:
    log = lambda *args: print(*args) if verbose else None

    norm = jnp.linalg.norm(tensor)

    if random_state is None: random_state = get_random_key()

    factors = init_cpd(tensor, rank, key=random_state)
    W, V, H = factors

    errors = []

    solve_subproblem_W = partial(solve_subproblem, unfolded=unfold_kolda(tensor, 0), mode=0)
    solve_subproblem_V = partial(solve_subproblem, unfolded=unfold_kolda(tensor, 1), mode=1)

    for iteration in range(max_iters):
        W = solve_subproblem_W(W=W, V=V, H=H)
        V = solve_subproblem_V(W=W, V=V, H=H)
        W, V = normalize_columns_V(W, V)

        H, dcoefs = update_H_with_polynomial_constraint(
            unfolded=unfold_kolda(tensor, 2),
            X=X, V=V, W=W, rank=rank, degree=degree,
        )
        
        factors = W, V, H
        error = relative_error(tensor, factors)

        if iteration > 0:
            diff = abs(error - errors[-1])
            log(f'iteration {iteration+1}: error = {error:.4f}, diff = {diff:.8f}')
            if diff < tol * norm: log(f'stopping at iteration {iteration+1}'); break

        else: log(f'iter {iteration+1}: error = {error:.4f}')

        errors.append(error)

    return factors, dcoefs, jnp.array(errors)

@jax.jit(static_argnames=('rank', 'degree'))
def update_H_with_polynomial_constraint(
    unfolded: Array,
    X: Float[Array, 'N m'],
    V: Float[Array, 'm r'],
    W: Float[Array, 'n r'],
    rank: int,
    degree: int,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'd r']]:
    Z = X @ V # inputs to g

    vand_matrices = []
    for r in range(rank):
        vand = vandermonde_matrix(Z[:, r], degree)
        vand_matrices.append(vand)

    vand_diag = block_diag(vand_matrices)

    KR = khatri_rao(V, W)
    K = jnp.kron(KR, jnp.eye(X.shape[0]))
    Z = K @ vand_diag

    Zinv = jnp.linalg.pinv(Z)
    dcoefs = Zinv @ reshape(unfolded, -1)
    dcoefs = reshape(dcoefs, (degree+1, rank))

    H = jnp.column_stack([Xi @ ci for Xi, ci in zip(vand_matrices, dcoefs.T)])
    return H, dcoefs
