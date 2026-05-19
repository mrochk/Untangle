import jax, jax.numpy as jnp
import tqdm as tqdm_module
from tqdm import tqdm
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype.typing import Callable, Tuple, Optional
from beartype import beartype

from untangle._common import get_random_key, find_number_inputs
from untangle import _ops as ops

@jaxtyped(typechecker=beartype)
def collect_information(
    function: Callable[[ArrayLike], Array],
    N: int, 
    key: Optional[Array] = None,
    n_inputs: Optional[int] = None, 
    minval: float = 0.0, 
    maxval: float = 1.0,
    X: Optional[Float[Array, 'N m']] = None,
) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:

    '''
    Generates random inputs X and collects outputs Y
    and stacked jacobian tensor J.
    '''

    assert callable(function)

    if n_inputs is None: n_inputs = find_number_inputs(function)
    if key is None: key = get_random_key()

    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))

    if X is None: X = jax.random.uniform(key, shape=(N, n_inputs), minval=minval, maxval=maxval)
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

def function_error(target: Callable, decoupling: Callable, X: ArrayLike) -> float:
    assert callable(target) and callable(decoupling)

    Y = jax.vmap(target)(X)
    Y_learned = jax.vmap(decoupling)(X)

    top = jnp.sqrt(jnp.mean((Y - Y_learned)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Y - jnp.mean(Y, axis=0))**2, axis=0))

    return top / bot * 100

def best_of_n(
    algorithm: Callable,
    n: int = 5, 
    key: Optional[Array] = None,
) -> Tuple:
    assert n > 0

    if key is None: key = get_random_key()

    min_error = jnp.inf
    min_key = None
    min_decoupling = None

    bar = tqdm(jax.random.split(key, n), desc=f'Returning best of {n} runs')
    for k in bar:
        decoupling, error = algorithm(key=k, show_progress=False)

        if error < min_error:
            min_decoupling = decoupling
            min_error = error
            min_key = k

        bar.set_postfix_str(f'error={error:.4f}, best={min_error:.4f}')

    return (min_decoupling, min_key, min_error)

def best_of_5(
    algorithm: Callable,
    key: Optional[Array] = None,
) -> Tuple:
    return best_of_n(algorithm, 5, key) 

def best_of_10(
    algorithm: Callable,
    key: Optional[Array] = None,
) -> Tuple:
    return best_of_n(algorithm, 10, key) 
