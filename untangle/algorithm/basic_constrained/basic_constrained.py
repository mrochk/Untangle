'''Implementation of the algorithm with polynomial constraints as described in Gabriel Hollander's PhD thesis.'''

import jax, jax.numpy as jnp
import multiprocessing as mp
from numpy.polynomial import Polynomial

from beartype import beartype
from beartype.typing import Tuple, Callable
from jaxtyping import jaxtyped, Array, Float, ArrayLike

from untangle.decomposition import run_many_cpd_constrained
from untangle.algorithm.common import inference_polynomial
from untangle.utils import make_log, make_polynomials

def integrate(dcoefs, Z, y, W):
    assert dcoefs.shape[1] == W.shape[1]

    degree, rank = dcoefs.shape
    coefs = jnp.zeros((degree + 1, rank))

    # find non-constant coefficients from integration:
    # c_{i+1} = c'_i / (i+1)
    for d in range(degree):
        coefs = coefs.at[d + 1, :].set(dcoefs[d, :] / (d + 1))

    # estimate constants from observations
    gs_no_czero = [Polynomial(coefs[:, j]) for j in range(rank)]

    # N = n samples
    # n = n outputs of initial func

    N = Z.shape[0]
    try:
        n = y.shape[1]
    except:
        n = 1

    columns_W = jnp.zeros((N * n, rank))
    residuals = jnp.zeros(N * n)

    for i in range(N):
        # evaluate g(z) without constants
        g_vals = jnp.array([gs_no_czero[j](Z[i, j]) for j in range(rank)])
        y_pred = W @ g_vals

        for j in range(n):  # for each output
            row_idx = i * n + j  # get the correct index for output j of sample i

            # we are looking for constants that minimize the residual
            # between the y and the values we get when using the polynomials
            # without constant terms
            residuals = residuals.at[row_idx].set(y[i, j] - y_pred[j])

            # design matrix of our least-squares problem
            # it contains the corresponding column of W for each output
            # (the one that multiplies the c_0 we want to solve for, for the corresponding output)
            columns_W = columns_W.at[row_idx, :].set(W[j, :])

    # solve for constants c_{i, 0}
    c_zeros = jnp.linalg.lstsq(columns_W, residuals, rcond=None)[0]
    coefs = coefs.at[0, :].set(c_zeros)

    return coefs

@jaxtyped(typechecker=beartype)
def decoupling_basic_constrained(
    X: Float[Array, 'N m'], 
    Y: Float[Array, 'N n'], 
    J: Float[Array, 'n m N'], 
    rank: int,
    degree: int = 3,
    n_init: int = mp.cpu_count(),
    verbose: int = 0,
) -> Tuple[
    Callable,
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'd r']],
]:
    '''Basic decoupling with additional polynomial constraints on 3rd factor.

    Args description (below) assumes the initial function takes m inputs and returns n outputs.

    Args:
        X: Operating points, of shape (N, m).
        Y: Function outputs for X, of shape (N, n).
        J: Stacked jacobian of shape (n, m, N).
        rank: Rank of the CPD.
        degree: Degree of internal polynomials. Defaults to 3.
        n_init: Number of (parallel) decompositions to run. The best is kept and used for decoupling.
        verbose: Verbose output from 0 to 2. Defaults to 0.

    Returns (f, (W, V, coefs)), where f is the callable decoupling, and (W, V, coefs) are the components.
    '''    
    
    log = make_log(verbose)
    log(f'Computing CP decomposition with polynomial constraint of J with rank {rank} and {n_init} (parallel) inits...')

    factors, dcoefs = run_many_cpd_constrained(J, X, rank, degree, verbose=verbose)
    W, V, H = factors

    Z = X @ V

    log('Recovering internals by integrating...')
    coefs = integrate(dcoefs, Z, Y, W)

    return inference_polynomial(W, V, coefs.T), factors

