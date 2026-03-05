import jax
from untangle.algorithm import cmtf_bsd
from untangle.utils import collect_information, get_random_key
from untangle.utils.testing import *

m, n, rank, f = POLYNOMIAL_2

X, Y, J = collect_information(f, 20, m)

d, (W, V, H, R) = cmtf_bsd(J, Y, X, rank, verbose=1)

print('factor W:')
print(W)

print('\nfactor V:')
print(V)

print()

x = jax.random.uniform(get_random_key(), m)

print(f(x))
print(d(x))
