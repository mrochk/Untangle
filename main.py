import jax, jax.numpy as jnp

from untangle.utils.testing import *
from untangle.algorithm import cmtf_bsd, cmtf_ssd
from untangle.utils import collect_information, get_random_key

num = 50

m, r, rank, f = POLYNOMIAL_2

X, Y, J = collect_information(f, num, m, range=(0, 1), key=get_random_key())

X, Y, J = collect_information(f, num, m, range=(0, 1), key=get_random_key())

inf, best, errs = cmtf_ssd(J, Y, X, rank, iterations=10, verbose=1)

x = jax.random.uniform(get_random_key(), shape=m)

print(f(x))
print(inf(x))
