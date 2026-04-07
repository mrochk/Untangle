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
        self.internals = internals

    def __call__(self, x: ArrayLike) -> ArrayLike:
        W, V = self.factors[0], self.factors[1]
        return W @ self.internals(V.T @ x)

    def inference(self, x: ArrayLike) -> ArrayLike:
        return self.__call__(x)
