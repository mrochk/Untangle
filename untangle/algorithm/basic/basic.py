import jax, jax.numpy as jnp
import multiprocessing as mp

from beartype import beartype
from jaxtyping import jaxtyped, Array, Float

from untangle.utils import make_log
from untangle.algorithm import Decoupling
from untangle.decomposition import run_many_cpd
from untangle._ops import vandermonde_diag, block_diag
from untangle._common import make_polynomials

@jaxtyped(typechecker=beartype)
def basic_decoupling(
    X: Float[Array, 'N m'], 
    Y: Float[Array, 'N n'], 
    J: Float[Array, 'n m N'], 
    rank: int,
    degree: int = 3,
    n_init: int = mp.cpu_count(),
    verbose: int = 0,
) -> Decoupling:
    
    log = make_log(verbose, '|BASIC| -> ')
    log(f'Computing CP decomposition of J with rank {rank} and {n_init} inits...')

    factors, weights = run_many_cpd(J, rank, verbose=verbose, n=n_init)
    W, V, H = factors

    W = W * weights

    log('Recovering internal coefficients...')
    coefs = find_internals_coefs(X, Y, W, V, degree)

    internals = make_polynomials(coefs)

    return Decoupling((W, V, H), internals)

def find_internals_coefs(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    W: Float[Array, 'n r'],
    V: Float[Array, 'm r'],
    degree: int,
) -> Float[Array, 'r d']:

    N = X.shape[0]
    rank = W.shape[1]

    W_diag = block_diag([W for _ in range(N)])

    Z = jnp.array([V.T @ x for x in X])

    Xk = jnp.concatenate([vandermonde_diag(z, degree) for z in Z], axis=0)

    coefs = jnp.linalg.lstsq(W_diag @ Xk, jnp.concatenate(Y))[0].reshape(rank, -1)
    assert tuple(coefs.shape) == (rank, degree+1)

    return coefs
