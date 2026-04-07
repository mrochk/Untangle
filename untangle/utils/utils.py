import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype.typing import Callable, Tuple, Optional
from beartype import beartype

from untangle._common import get_random_key
from untangle import _ops as ops

@jaxtyped(typechecker=beartype)
def collect_information(
    function: Callable[[ArrayLike], Array],
    N: int, 
    m: int, 
    key: Array = None,
    minval: float = 0.0, 
    maxval: float = 1.0,
) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:

    '''
    Generates random inputs X and collects outputs Y
    and stacked jacobian tensor J.
    '''

    assert callable(function)

    if key is None: key = get_random_key()

    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))

    X = jax.random.uniform(key, shape=(N, m), minval=minval, maxval=maxval)
    Y = function(X)
    J = jacobian(X)

    return (X, Y, J.transpose((1, 2, 0)))

@jaxtyped(typechecker=beartype)
def cpd_reconstruct(
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Optional[Float[Array, 'r']] = None,
) -> Float[Array, 'n m N']:
    rank = factors[0].shape[1]
    if weights is None: weights = jnp.ones(rank)
    return ops.reconstruct(*factors, weights)

@jaxtyped(typechecker=beartype)
def cpd_error(
    tensor: Float[Array, 'n m N'],
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Optional[Float[Array, 'r']] = None,
) -> Float[Array, '']:
    _tensor = cpd_reconstruct(factors, weights)
    return jnp.linalg.norm(tensor - _tensor) / jnp.linalg.norm(tensor)

def function_error(f: Callable, learned: Callable, X: ArrayLike) -> float:
    assert callable(f) and callable(learned)

    Y = jnp.array([f(x) for x in X])
    Y_learned = jnp.array([learned(x) for x in X])

    top = jnp.sqrt(jnp.mean((Y - Y_learned)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Y - jnp.mean(Y, axis=0))**2, axis=0))

    return top / bot * 100
