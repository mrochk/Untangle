import jax, jax.numpy as jnp

from untangle.algorithm import cmtf_bsd, cmtf_psd
from untangle.utils import collect_information, function_error
from untangle.scaler import JacobianScaler

key = jax.random.key(0)
keys = jax.random.split(key, 10)

def f(x):
    a, b = x
    return jnp.array([
        2*a**3 + b**2 + 1,
        3*b**2 + 2*a - 1,
    ])

def f(x):
    a, b = x
    return jnp.array([
        2*a**3 + b**2 + 1,
        3*b**2 + 2*a - 1,
    ])
    
N = 50
rank = 5
niters = 10
dof = N


X_test, _, _ = collect_information(f, N, jax.random.key(42))

for key in keys:

    X, Y, J = collect_information(f, N, key)

    scaler = JacobianScaler(J, Y)
    J_scaled, Y_scaled = scaler.scale()

    f_hat = scaler.unscale(cmtf_psd(X, Y_scaled, J_scaled, rank, niters=niters, dof=dof, key=key))
    errors = function_error(f, f_hat, X_test)
    print('PSD Errors:', errors)

    f_hat = scaler.unscale(cmtf_bsd(X, Y_scaled, J_scaled, rank, niters=niters, dof=dof, key=key))
    errors = function_error(f, f_hat, X_test)
    print('BSD Errors:', errors)

    print()