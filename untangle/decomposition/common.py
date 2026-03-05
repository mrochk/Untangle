import jax, jax.numpy as jnp

from jaxtyping import jaxtyped, Array, Float
from beartype import beartype 
from beartype.typing import Tuple 

from untangle.utils import get_random_key
from untangle.ops import khatri_rao

@jaxtyped(typechecker=beartype)
def init_cpd(tensor: Float[Array, 'n m N'], rank: int, key: Array):
    n, m, N = tensor.shape

    W = jax.random.normal(key, shape=(n, rank))
    V = jax.random.normal(key, shape=(m, rank))
    H = jax.random.normal(key, shape=(N, rank))

    return W, V, H

@jax.jit(static_argnames=('mode',))
@jaxtyped(typechecker=beartype)
def solve_subproblem(
    unfolded: Array, 
    W: Float[Array, 'n r'], 
    V: Float[Array, 'm r'],
    H: Float[Array, 'N r'], 
    mode: int,
):
    assert 0 <= mode <= 2

    match mode:
        case 0:
            KR = khatri_rao(H, V)
            CC = H.T @ H
            BB = V.T @ V
            return unfolded @ KR @ jnp.linalg.pinv(CC * BB)

        case 1:
            KR = khatri_rao(H, W)
            CC = H.T @ H
            AA = W.T @ W
            return unfolded @ KR @ jnp.linalg.pinv(CC * AA)

        case 2:
            KR = khatri_rao(V, W)
            BB = V.T @ V
            AA = W.T @ W
            return unfolded @ KR @ jnp.linalg.pinv(BB * AA)

@jax.jit
def column_normalize(factor: Float[Array, '_ r']) -> Tuple[Float[Array, '_ r'], Float[Array, 'r']]:
    rank = factor.shape[1]; weights = []

    for r in range(rank):
        column = factor[:, r]
        norm = jnp.linalg.norm(column)
        weights.append(norm)
        factor = factor.at[:, r].set(column / norm)

    return factor, jnp.array(weights)
