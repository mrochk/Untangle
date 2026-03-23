import jax, jax.numpy as jnp

from untangle.utils import collect_information, get_random_key
from untangle.algorithm import cmtf_bsd
from untangle.testing import *
from untangle.scaler import scale_tensor

import matplotlib.pyplot as plt

num = 10
f, m, n, family, rank = POLYNOMIAL_1.unpack()
rank = 3

X, Y, J = collect_information(f, num, m, key=get_random_key())
factors, J_scaled, Y_scaled = scale_tensor(J, Y)

decoupling = cmtf_bsd(X, Y_scaled, J_scaled, rank, verbose=1)

print(f(X[0]))
print(decoupling(X[0]) / factors)