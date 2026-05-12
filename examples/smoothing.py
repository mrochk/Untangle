import jax, jax.numpy as jnp

from untangle.scaler import JacobianScaler
from untangle.algorithm import cmtf_bsd, cmtf_psd
from untangle.utils import collect_information, function_error

key  = jax.random.key(0)
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
        jnp.cos(4*jnp.pi*a) + b,
        -jnp.sin(3*jnp.pi*a) + a**2,
    ])

N = 50
niters = 20
dof = N//2

X_test, _, _ = collect_information(f, N, jax.random.key(42))

rank = 3

for key in keys:

    X, Y, J = collect_information(f, N, key)

    scaler = JacobianScaler(J, Y)
    J_scaled, Y_scaled = scaler.scale()

    info = (X, Y_scaled, J_scaled)

    print(rank)

    f_hat = scaler.unscale(cmtf_psd(*info, rank, niters=niters, dof=dof, key=key))
    errors = function_error(f, f_hat, X_test)
    print('PSD Errors:', errors)

    f_hat = scaler.unscale(cmtf_bsd(X, Y_scaled, J_scaled, rank, niters=niters, dof=dof, key=key))
    errors = function_error(f, f_hat, X_test)
    print('BSD Errors:', errors)


    rank += 1
