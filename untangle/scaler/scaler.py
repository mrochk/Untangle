import jax, jax.numpy as jnp
from untangle.utils import get_random_key

from jaxtyping import jaxtyped
from beartype.typing import Callable

class FunctionScaler:
    def __init__(self, function: Callable, n_inputs: int, n_samples_estimate: int = 1_000):
        assert callable(function)
        self.function = function
        inputs = jax.random.uniform(get_random_key(), shape=(n_samples_estimate, n_inputs))
        self.outputs = jax.vmap(function)(inputs)

    def scale(self) -> Callable: pass

    def unscale(self, f_scaled: Callable) -> Callable: pass

class MaxFunctionScaler(FunctionScaler):
    def __init__(self, function: Callable, n_inputs: int, n_samples_estimate: int = 1000):
        super().__init__(function, n_inputs, n_samples_estimate)
        self.maximums = jnp.max(self.outputs, axis=0)

    def scale(self) -> Callable:
        return lambda x: self.function(x) / self.maximums

    def unscaled(self, f_scaled: Callable) -> Callable:
        return lambda x: f_scaled(x) * self.maximums

class StdFunctionScaler(FunctionScaler):
    def __init__(self, function: Callable, n_inputs: int, n_samples_estimate: int = 1000):
        super().__init__(function, n_inputs, n_samples_estimate)
        self.stdevs = jnp.std(self.outputs, axis=0)

    def scale(self) -> Callable:
        return lambda x: self.function(x) / self.stdevs

    def unscale(self, f_scaled: Callable) -> Callable:
        return lambda x: f_scaled(x) * self.stdevs

class TensorScaler:
    def __init__(self, tensor):
        n, m, N = tensor.shape
        self.factors = jnp.sqrt(m*N) / jnp.linalg.norm(tensor.reshape(-1), axis=1)
        self.tensor = tensor

    def scale(self, tensor = None): 
        if tensor is not None:
            return tensor * self.factors[:, None, None]
        return self.tensor * self.factors[:, None, None]

    def unscale(self, scaled_tensor): 
        return scaled_tensor / self.factors[:, None, None]
 