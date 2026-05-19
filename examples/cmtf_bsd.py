import jax.numpy as jnp
from untangle.algorithm import cmtf_bsd
from untangle.scaler import JacobianScaler
from untangle.utils import collect_information, function_error

def f(x):
    a, b = x
    return jnp.array([2*a**3 + b**2 + 1, 3*b**2 + 2*a - 1])

X, Y, J = collect_information(f, N=20)

scaler = JacobianScaler(J, Y)
J_scaled, Y_scaled = scaler.scale()

decoupling = scaler.unscale(cmtf_bsd(X, Y_scaled, J_scaled, rank=3))

print(f'Errors: {function_error(f, decoupling, X)}')

