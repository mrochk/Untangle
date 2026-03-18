import jax, jax.numpy as jnp

from untangle.utils import collect_information, get_random_key
from untangle.algorithm import cmtf_ssd, ctd_ssd, cmtf_bsd
from untangle.utils.testing import *
from untangle.scaler import MaxScaler, StdScaler

import matplotlib.pyplot as plt

srange = (0, 1)

num = 100
m, r, rank, f = POLYNOMIAL_1

scaler = StdScaler(f, m)
f_scaled = scaler.scale()

X, Y, J = collect_information(f_scaled, num, m, range=srange, key=get_random_key())

learned_scaled, best, errs = cmtf_ssd(J, Y, X, rank, iterations=10, verbose=1, use='first')

learned = scaler.unscale(learned_scaled)

x = jax.random.uniform(get_random_key(), shape=m, minval=srange[0], maxval=srange[1])

print(f(x))
print(learned(x))

plt.plot(errs)
plt.show()

#inf, best, errs = cmtf_bsd( J, Y, X, rank, iterations=40, verbose=1, lam=0.1,) print(f(x)) print(inf(x))