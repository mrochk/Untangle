import jax.numpy as jnp

from untangle.algorithm import cmtf_bsd
from untangle.utils import collect_information, function_error

def f(x):
    a, b = x
    return jnp.array([
        2*a**3 + b**2 + 1,
        3*b**2 + 2*a - 1,
    ])
    
m = 2
N = 30
rank = 4

X, Y, J = collect_information(f, N)

f_hat = cmtf_bsd(X, Y, J, rank)
errors = function_error(f, f_hat, X)
print('Errors:', errors)

