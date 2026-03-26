import jax, jax.numpy as jnp
import random

from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype.typing import Callable, Tuple, Optional
from beartype import beartype

def make_log(verbose: int, prefix: str = '') -> Callable[[], None]:
    def log(*args):
        if verbose <= 0: return
        print(prefix, end='')
        print(*args, flush=True)
    return log

def get_random_key() -> Array:
    return jax.random.key(random.randint(0, int(1e10)))

@jaxtyped(typechecker=beartype)
def collect_information(

    function: Callable[[ArrayLike], Array],
    N: int, 
    m: int, 
    key: Array = None,
    minval: float = 0.0, 
    maxval: float = 1.0,

) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:

    '''Collect outputs Y and stacked jacobian J of the function.'''

    assert callable(function)

    if key is None: key = get_random_key()

    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))

    X = jax.random.uniform(key, shape=(N, m), minval=minval, maxval=maxval)
    Y = function(X)
    J = jacobian(X)

    return (X, Y, J.transpose((1, 2, 0)))

@jax.jit
@jaxtyped(typechecker=beartype)
def reconstruct_tensor(

    factors: Tuple[
        Float[Array, 'n r'],
        Float[Array, 'm r'],
        Float[Array, 'N r']
    ],

    weights: Optional[Float[Array, 'r']] = None,

) -> Float[Array, 'n m N']:

    W, V, H = factors
    N, m, n, r = H.shape[0], V.shape[0], W.shape[0], W.shape[1]

    if weights is None: weights = jnp.ones(r)

    N, m, n = H.shape[0], V.shape[0], W.shape[0]
    tensor = jnp.zeros(shape=(n, m, N))

    for i in range(r):
        rank1 = W[:, i][:, None, None] * V[:, i][None, :, None] * H[:, i][None, None, :]
        tensor += weights[i] * rank1

    return tensor

@jax.jit
def cpd_error(

    tensor: Float[Array, 'n m N'],
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Optional[Float[Array, 'r']] = None,

) -> Float[Array, '']:

    _tensor = reconstruct_tensor(factors, weights)

    return jnp.linalg.norm(tensor - _tensor) / jnp.linalg.norm(tensor)

def outputs_error(f: Callable, learned: Callable, X: ArrayLike):
    Y = jnp.array([f(x) for x in X])
    Y_learned = jnp.array([learned(x) for x in X])

    top = jnp.sqrt(jnp.mean((Y - Y_learned)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Y - jnp.mean(Y, axis=0))**2, axis=0))

    return top / bot * 100
