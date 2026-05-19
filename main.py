import jax, jax.numpy as jnp
from functools import partial

from untangle.algorithm import cmtf_psd
from untangle.utils import collect_information, function_error, best_of_n

key = jax.random.key(0)

def f(x):
    x1, x2, x3 = x
    return jnp.array([x1**3 - 2*x1**2 + x1*x3, x2**3 + x3**2 - x2*x1])

N = 40
rank = 5
X, Y, J = collect_information(f, N, key=key)

algorithm = partial(cmtf_psd, X, Y, J, rank)
decoupling, error, key = best_of_n(algorithm, key=key, n=20)
print(key)

print(function_error(f, decoupling, X))
