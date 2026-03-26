import jax, jax.numpy as jnp

from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Callable, Tuple
from beartype import beartype

from untangle.utils import get_random_key

class FunctionScaler:

    @jaxtyped(typechecker=beartype)
    def __init__(self, f: Callable, n_inputs: int, key: Array = None, scaler: str = 'max', N: int = 1_000):

        assert callable(f)
        self.f = f

        if key is None: key = get_random_key()
        X = jax.random.uniform(key, shape=(N, n_inputs))

        self.outputs = jax.vmap(f)(X)

        match scaler:
            case 'max': self.factors = 1 / jnp.max(self.outputs, axis=0)
            case 'std': self.factors = 1 / jnp.std(self.outputs, axis=0)
            case _: raise ValueError('only "max" and "std" scalers supported')

    @jaxtyped(typechecker=beartype)
    def scale(self) -> Callable:
        def f_scaled(x): return self.f(x) * self.factors
        return f_scaled

    @jaxtyped(typechecker=beartype)
    def unscale(self, f_scaled: Callable) -> Callable:
        def f_unscaled(x): return f_scaled(x) / self.factors
        return f_unscaled

class JacobianScaler:

    @jaxtyped(typechecker=beartype)
    def __init__(self, J: Float[Array, 'n m N'], Y: Float[Array, 'N n']):
        self.J = J
        self.Y = Y

        n, m, N = J.shape
        self.factors = jnp.sqrt(m*N) / jnp.linalg.norm(J.reshape(n, -1), axis=1)

    @jaxtyped(typechecker=beartype)
    def scale(self) -> Tuple[Float[Array, 'n m N'], Float[Array, 'N n']]:
        J_scaled = self.J * self.factors[:, None, None]
        Y_scaled = self.Y * self.factors[None, :]
        return J_scaled, Y_scaled

    @jaxtyped(typechecker=beartype)
    def unscale(self, f_scaled: Callable) -> Callable:
        def f_unscaled(x): return f_scaled(x) / self.factors
        return f_unscaled
