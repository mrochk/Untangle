import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import jaxtyped, Array, Float

from untangle.algorithm import Decoupling
from untangle.decomposition import cpd
from untangle._ops import vandermonde_diag, block_diag, lstsq
from untangle._common import make_polynomials, make_log

@jaxtyped(typechecker=beartype)
def basic_decoupling(
    X: Float[Array, 'N m'], 
    Y: Float[Array, 'N n'], 
    J: Float[Array, 'n m N'], 
    rank: int,
    degree: int = 3,
    maxiters: int = 100,
    verbose: int = 0,
    key: Optional[Array] = None,
    **cpd_kwargs,
) -> Decoupling:
    
    log = make_log(verbose, '|BASIC-DECOUPLING|')

    factors, weights, errors = cpd(J, rank, maxiters, key, **cpd_kwargs)
    W, V, H = factors

    W = W * weights

    log('Recovering internal coefficients...')
    coefs = _find_internals_coefs(X, Y, W, V, degree)

    internals = make_polynomials(coefs)

    return Decoupling((W, V, H), internals), errors[-1]

def _find_internals_coefs(
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

    coefs = lstsq(W_diag @ Xk, jnp.concatenate(Y)).T.reshape(rank, degree+1)

    return coefs
