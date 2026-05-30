import jax, jax.numpy as jnp
from functools import partial
from tqdm import tqdm

from jaxtyping import jaxtyped, Array, Float
from beartype import beartype 
from beartype.typing import Tuple, Optional 

from untangle.utils import get_random_key, cpd_error
from untangle import _ops as ops
from untangle._common import (
    make_log,
    initialize, 
    solve_cpd_subproblem, 
    cpd_stopping_criterion, 
)

@jaxtyped(typechecker=beartype)
def cpd_polynomial_constraint(
    X: Float[Array, 'N m'],
    J: Float[Array, 'n m N'],
    rank: int,
    degree: int = 3,
    maxiters: int = 100,
    key: Optional[Array] = None,
    tol: float = 1e-6,
    verbose: int = 0,
) -> Tuple[
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    Float[Array, 'd r'],
    Array,
]:
    if key is None: key = get_random_key()

    prefix = '|CPD-CONSTRAINED|'
    log = make_log(verbose, prefix)

    errors = []

    norm = jnp.linalg.norm(J)

    factors, weights = initialize(J, rank, key), jnp.ones(rank)
    W, V, H = factors

    J0 = ops.unfold_kolda(J, 0)
    J1 = ops.unfold_kolda(J, 1)
    J2 = ops.unfold_kolda(J, 2)

    solve_W = partial(solve_cpd_subproblem, unfolded=J0, mode=0)
    solve_V = partial(solve_cpd_subproblem, unfolded=J1, mode=1)

    bar = tqdm(range(maxiters), prefix)
    for iteration in bar:
        W = solve_W(W=W, V=V, H=H)
        V = solve_V(W=W, V=V, H=H)
        W, V = ops.normalize_columns_V(W, V)

        H, dcoefs = update_H_with_polynomial_constraint(J2, X, V, W, rank, degree)

        factors = W, V, H
        error = cpd_error(J, factors, weights)

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

    return factors, dcoefs, jnp.array(errors)

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
        vand = ops.vandermonde_matrix(Z[:, r], degree)
        vand_matrices.append(vand)

    vand_diag = ops.block_diag(vand_matrices)

    KR = ops.khatri_rao(V, W)
    K = jnp.kron(KR, jnp.eye(X.shape[0]))
    Z = K @ vand_diag

    Zinv = jnp.linalg.pinv(Z)
    dcoefs = Zinv @ ops.reshape(unfolded, -1)
    dcoefs = ops.reshape(dcoefs, (degree+1, rank))

    H = jnp.column_stack([Xi @ ci for Xi, ci in zip(vand_matrices, dcoefs.T)])
    return H, dcoefs
