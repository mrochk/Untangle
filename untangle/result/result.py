import jax
from jaxtyping import ArrayLike
from beartype.typing import Callable

class Decoupling:
    factors: tuple
    internals: Callable

    def __init__(self, factors: tuple, internals: Callable):
        assert len(factors) >= 3
        assert callable(internals)
        assert factors[0].shape[-1] == factors[1].shape[-1] == factors[2].shape[-1]

        self.factors = factors
        if len(factors) == 4: self.W, self.V, self.H, self.R = factors
        else: self.W, self.V, self.H = factors
        self.internals = internals

    @jax.jit
    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.W @ self.internals(self.V.T @ x)

    def inference(self, x: ArrayLike) -> ArrayLike:
        return self.__call__(x)
