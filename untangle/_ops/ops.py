'''Tensor operations that are not by default in `jax.numpy`.'''

import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype.typing import Iterable
from beartype import beartype

@jax.jit(static_argnames=('shape',))
def reshape(tensor: Array, shape):
    return jnp.reshape(tensor, shape, order='F')

@jax.jit(static_argnames=('mode',))
def unfold_kolda(tensor: ArrayLike, mode: int) -> ArrayLike:
    '''Tensor unfolding as defined in "Tensor decompositions and applications" from Kolda and Bader.'''
    return jnp.reshape(jnp.moveaxis(tensor, mode, 0), shape=(tensor.shape[mode], -1), order='F')

@jax.jit
@jaxtyped(typechecker=beartype)
def khatri_rao(A: Float[Array, 'm k'], B: Float[Array, 'n k']) -> Float[Array, 'mn k']:
    m, k = A.shape
    n, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(m*n, k)

@jax.jit
def block_diag(arrays: list[ArrayLike]) -> Float[Array, 'a b']:
    arrays = [jnp.atleast_2d(a) for a in arrays]
    rows = sum([a.shape[0] for a in arrays])
    cols = sum([a.shape[1] for a in arrays])

    result = jnp.zeros((rows, cols), dtype=arrays[0].dtype)

    r, c = 0, 0
    for a in arrays:
        rr, cc = a.shape
        result = result.at[r:r+rr, c:c+cc].set(a)
        r += rr
        c += cc

    return result

def vandermonde_vector(x: float, d: int):
    return jnp.array([x**e for e in range(d + 1)])

def vandermonde_matrix(values: Iterable[float], degree: int):
    return jnp.vstack([vandermonde_vector(v, degree) for v in values])

def vandermonde_diag(X, d: int):
    return block_diag([vandermonde_vector(x, d) for x in X])
