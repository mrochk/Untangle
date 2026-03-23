import random
import jax, jax.numpy as jnp
from functools import partial

from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Callable, Tuple, Optional
from beartype import beartype

def make_log(verbose: int, prefix: str = '') -> Callable[[], None]:
    def log(*args):
        if verbose <= 0: return
        print(prefix, end='')
        print(*args, flush=True)
    return log

def get_random_key() -> Array:
    return jax.random.key(random.randint(0, int(1e6)))

@jaxtyped(typechecker=beartype)
def collect_information(
    function: Callable[[Array], Array],
    N: int, m: int, range = (0, 1),
    key: Array = get_random_key(),
) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:

    '''Collect outputs Y and stacked jacobian J of the function.'''

    assert(callable(function))

    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))

    lo, hi = range
    X = jax.random.uniform(key, shape=(N, m), minval=lo, maxval=hi)

    Y = function(X)
    J = jacobian(X).transpose((1, 2, 0))

    return (X, Y, J)

@jax.jit
def reconstruct_tensor(
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Float[Array, 'r'],
) -> Float[Array, 'n m N']:
    W, V, H = factors

    N, m, n = H.shape[0], V.shape[0], W.shape[0]
    tensor = jnp.zeros(shape=(n, m, N))

    rank = W.shape[1]
    for r in range(rank):
        rank1 = W[:, r][:, None, None] * V[:, r][None, :, None] * H[:, r][None, None, :]
        tensor += weights[r] * rank1

    return tensor

@jax.jit
def relative_error(
    tensor: Float[Array, 'n m N'],
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Optional[Float[Array, 'r']] = None,
) -> Float[Array, '']:
    if weights is None: weights = jnp.ones(factors[0].shape[1])
    return jnp.linalg.norm(tensor - reconstruct_tensor(factors, weights)) / jnp.linalg.norm(tensor)
