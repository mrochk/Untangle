import jax, jax.numpy as jnp
jax.numpy.set_printoptions(precision=6, suppress=True)

from untangle.utils import collect_information, get_random_key
from untangle.algorithm import ctd_ssd, cmtf_ssd
from untangle.testing import *
from untangle.scaler import scale_tensor

import matplotlib.pyplot as plt

num = 100
f, m, n, family, rank = NONLINEAR_4.unpack()
rank = 6

X, Y, J = collect_information(f, num, m, key=get_random_key())
#factors, J_scaled, Y_scaled = scale_tensor(J, Y)
J_scaled, Y_scaled = J, Y

decoupling = ctd_ssd(X, Y_scaled, J_scaled, rank, verbose=1)
print(f(X[0]))
print(decoupling(X[0]))
#print(decoupling(X[0]) / factors)

decoupling = cmtf_ssd(X, Y_scaled, J_scaled, rank, 1.0, verbose=1)
print(f(X[0]))
print(decoupling(X[0]))
#print(decoupling(X[0]) / factors)
