'''Implementation of the paper https://arxiv.org/pdf/1410.4060.'''

import jax, jax.numpy as jnp
import multiprocessing as mp

from beartype import beartype
from beartype.typing import Tuple, Callable
from jaxtyping import jaxtyped, Array, Float

from untangle.algorithm.common import inference_polynomial
from untangle.decomposition import run_many_cpd
from untangle.utils import make_log, make_polynomials
from untangle.ops import vandermonde_vector, block_diag

@jaxtyped(typechecker=beartype)
def decoupling_basic(
    J: Float[Array, 'n m N'], 
    Y: Float[Array, 'N n'], 
    X: Float[Array, 'N m'], 
    rank: int,
    degree: int = 3,
    n_init: int = mp.cpu_count(),
    verbose: int = 0,
) -> Tuple[Callable, Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'r d']]]:

    '''Basic decoupling algorithm as described in https://arxiv.org/pdf/1410.4060.

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
    log(f'Computing CP decomposition of J with rank {rank} and {n_init} (parallel) inits...')

    factors, weights = run_many_cpd(J, rank, verbose=verbose, n=n_init)
    W, V, H = factors

    W = W * weights

    log('Recovering internal coefficients...')
    coefs = find_internals_coefficients(X, Y, W, V, degree)
    ret = (W, V, coefs)

    return inference_polynomial(*ret), ret

def find_internals_coefficients(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    W: Float[Array, 'n r'],
    V: Float[Array, 'm r'],
    degree: int,
) -> Float[Array, 'r d']:

    N = X.shape[0]
    rank = W.shape[1]

    def vand_diag(X, d: int):
        return block_diag([vandermonde_vector(x, d) for x in X])

    W_diag = block_diag([W for _ in range(N)])

    Z = jnp.array([V.T @ x for x in X])

    Xk = jnp.concatenate([vand_diag(z, degree) for z in Z], axis=0)

    coefs = jnp.linalg.lstsq(W_diag @ Xk, jnp.concatenate(Y))[0].reshape(rank, -1)
    assert tuple(coefs.shape) == (rank, degree+1)

    return coefs
