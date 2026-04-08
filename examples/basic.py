import jax.numpy as jnp

from untangle.algorithm import basic_decoupling, ctd_polynomial
from untangle.utils import collect_information, function_error

def f(x):
    a, b = x
    return jnp.array([
        2*a**3 + b**2 + 1,
        3*b**2 + 2*a - 1,
    ])
    
m = 2
N = 10
degree = 3
rank = 4

X, Y, J = collect_information(f, N)

print('Without polynomial constraint:')
f_hat = basic_decoupling(X, Y, J, rank, degree)
errors = function_error(f, f_hat, X)
print('Errors:', errors)

print('\nWith polynomial constraint:')
f_hat = ctd_polynomial(X, Y, J, rank, degree)
errors = function_error(f, f_hat, X)
print('Errors:', errors)
