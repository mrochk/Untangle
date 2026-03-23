import jax, jax.numpy as jnp

from untangle.utils import collect_information, get_random_key
from untangle.algorithm import basic_decoupling
from untangle.testing import *
from untangle.scaler import MaxFunctionScaler, StdFunctionScaler

import matplotlib.pyplot as plt

num = 100
m, r, rank, f = PERIODIC_1

X, Y, J = collect_information(f, num, m, key=get_random_key())

learned = basic_decoupling(J, Y, X, rank, verbose=1)

print(f(X[0]))
print(learned(X[0]))
