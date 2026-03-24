import jax, jax.numpy as jnp
jax.numpy.set_printoptions(precision=6, suppress=True)
from functools import partial

from untangle.scaler import scale_tensor, MaxFunctionScaler, StdFunctionScaler
from untangle.utils import collect_information, outputs_error
from untangle import algorithm

import matplotlib.pyplot as plt

def f(u):
    return jnp.array([
        100000*jnp.cos(4 * jnp.pi * u),
        -0.00001*jnp.sin(3 * jnp.pi * u),
    ])

m = 1
n = 2
rank = 2
num = 30
iterations = 10

keys = jax.random.split(jax.random.key(0), 5)

algo = partial(algorithm.cmtf_ssd, rank=rank, iterations=iterations, lam=1.0)

def no_scaling(f, keys):
    errors = []
    errors_all = []

    for key in keys:
        X, Y, J = collect_information(f, num, m, key=key)

        decoupling = algo(X, Y, J, key=key)
        errors_i = outputs_error(f, decoupling, X)
        errors.append(jnp.mean(errors_i))
        errors_all.append(errors_i)

    return errors, errors_all

def max_scaling(f, keys):
    errors = []
    errors_all = []

    scaler = MaxFunctionScaler(f, m)
    f_scaled = scaler.scale()

    for key in keys:
        X, Y, J = collect_information(f_scaled, num, m, key=key)

        decoupling_scaled = algo(X, Y, J)
        decoupling = scaler.unscale(decoupling_scaled) 

        errors_i = outputs_error(f, decoupling, X)

        errors.append(jnp.mean(errors_i))
        errors_all.append(errors_i)

    return errors, errors_all

def std_scaling(f, keys):
    errors = []
    errors_all = []

    scaler = StdFunctionScaler(f, m)
    f_scaled = scaler.scale()

    for key in keys:
        X, Y, J = collect_information(f_scaled, num, m, key=key)

        decoupling_scaled = algo(X, Y, J)
        decoupling = scaler.unscale(decoupling_scaled) 

        errors_i = outputs_error(f, decoupling, X)

        errors.append(jnp.mean(errors_i))
        errors_all.append(errors_i)

    return errors, errors_all

def tensor_scaling(f, keys):
    errors = []
    errors_all = []

    for key in keys:
        X, Y, J = collect_information(f, num, m, key=key)

        factors, J_scaled, Y_scaled = scale_tensor(J, Y)

        decoupling_scaled = algo(X, Y_scaled, J_scaled, key=key)
        #decoupling_scaled = cmtf_bsd(X, Y_scaled, J_scaled, rank, dof=40, iterations=iterations, verbose=1, key=key)
        def decoupling(x): return decoupling_scaled(x) / factors

        errors_i = outputs_error(f, decoupling, X)

        errors.append(jnp.mean(errors_i))
        errors_all.append(errors_i)

    return errors, errors_all

#print("no scaling")
#errors_no_scaling, errors_no_scaling_all = no_scaling(f, keys)

print("tensor")
errors_tensor, errors_tensor_all = tensor_scaling(f, keys)

#print("max")
#errors_max, errors_max_all = max_scaling(f, keys)

#print("std")
#errors_std, errors_std_all = std_scaling(f, keys)

#print('no scaling', jnp.min(jnp.array(errors_no_scaling)))
print('tensor scaling', jnp.min(jnp.array(errors_tensor)))

#print('max scaling', jnp.min(jnp.array(errors_max)))
#print('std scaling', jnp.min(jnp.array(errors_std)))
