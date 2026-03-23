import jax, jax.numpy as jnp
import multiprocessing as mp
from numpy.polynomial import Polynomial

from beartype import beartype
from jaxtyping import jaxtyped, Array, Float

from untangle.utils import make_log
from untangle.algorithm import Decoupling
from untangle.decomposition import run_many_cpd_constrained
from untangle.algorithm.common import make_polynomials

@jaxtyped(typechecker=beartype)
def basic_decoupling_constrained(
    J: Float[Array, 'n m N'], 
    Y: Float[Array, 'N n'], 
    X: Float[Array, 'N m'], 
    rank: int,
    degree: int = 3,
    n_init: int = mp.cpu_count(),
    verbose: int = 0,
) -> Decoupling:
    
    log = make_log(verbose, '|BASIC-CONSTRAINED| -> ')
    log(f'Computing CP decomposition with polynomial constraint of J with rank {rank} and {n_init} inits...')

    factors, dcoefs = run_many_cpd_constrained(J, X, rank, degree, verbose=verbose)
    W, V, H = factors

    Z = X @ V

    log('Recovering internals by integrating...')
    coefs = integrate(dcoefs, Z, Y, W)

    internals = make_polynomials(coefs)

    return Decoupling(factors, internals)

def integrate(dcoefs, Z, Y, W):
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
    n = Y.shape[1]

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
            residuals = residuals.at[row_idx].set(Y[i, j] - y_pred[j])

            # design matrix of our least-squares problem
            # it contains the corresponding column of W for each output
            # (the one that multiplies the c_0 we want to solve for, for the corresponding output)
            columns_W = columns_W.at[row_idx, :].set(W[j, :])

    # solve for constants c_{i, 0}
    c_zeros = jnp.linalg.lstsq(columns_W, residuals, rcond=None)[0]
    coefs = coefs.at[0, :].set(c_zeros)

    return coefs
