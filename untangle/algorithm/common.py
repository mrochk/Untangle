import jax, jax.numpy as jnp

from jaxtyping import Float, Array
from beartype.typing import Callable

from untangle.utils import make_polynomials

def inference_polynomial(
    W: Float[Array, 'n r'], 
    V: Float[Array, 'm r'], 
    coefs: Float[Array, 'r d'],
) -> Callable:
    return lambda x: W @ make_polynomials(coefs)(V.T @ x)

def normalize_columns_V(W: Float[Array, 'n r'], V: Float[Array, 'm r']):
    rank = W.shape[1]
    for i in range(rank):
        colV, colW = V[:, i], W[:, i]
        norm = jnp.linalg.norm(colV) + 1e-12
        V = V.at[:, i].set(colV / norm)
        W = W.at[:, i].set(colW * norm)
    return W, V
