import jax
from untangle.algorithm import cmtf_bsd
from untangle.algorithm.cmtf_ssd.cmtf_ssd import cmtf_ssd
from untangle.utils import collect_information, get_random_key
from untangle.utils.testing import *

m, n, rank, f = POLYNOMIAL_2

X, Y, J = collect_information(f, 50, m)

#d, (W, V, H, R), _ = cmtf_bsd(J, Y, X, rank, verbose=1, max_iters=200, random_state=get_random_key())

d, (W, V, H, R), _ = cmtf_ssd(
    J, Y, X, rank, verbose=1, max_iters=50, random_state=get_random_key(),
    lam=0.1,
    smoothing=1.0,
    gamma=1.0,
)

x = jax.random.uniform(get_random_key(), m)
print(f(x))
print(d(x))
