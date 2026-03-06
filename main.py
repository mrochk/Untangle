import jax, jax.numpy as jnp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from untangle.algorithm import cmtf_bsd
from untangle.algorithm.cmtf_ssd.cmtf_ssd import cmtf_ssd
from untangle.algorithm.cmtf_ssd.spline import *
from untangle.utils import collect_information, get_random_key
from untangle.utils.testing import *

#num = 30
#x = jnp.linspace(0, 1, num)

#y = jnp.cos(4*jnp.pi*x) + jax.random.normal(jax.random.key(0), shape=num)
#dy = -jnp.sin(4*jnp.pi*x) * 4*jnp.pi + jax.random.normal(jax.random.key(1), shape=num)

#plt.scatter(x, y, label='y')
#plt.scatter(x, dy, label='dy')

#m = fit_with_deriv(x, y, dy, 10.0, 0.1)
#spline = CubicSpline(x, m)
#dspline = spline.derivative()

#bias = (dy - dspline(x)).mean()

#xx = jnp.linspace(0, 1, 100)

#plt.plot(xx, spline(xx), label='spline')
#plt.plot(xx, dspline(xx) + bias, label='dspline')

#plt.tight_layout()
#plt.legend()
#plt.savefig('fig.png')

m, n, rank, f = POLYNOMIAL_1

X, Y, J = collect_information(f, 20, m, range=(0, 1))

#d, (W, V, H, R), _ = cmtf_bsd(J, Y, X, rank, verbose=1, max_iters=200, random_state=get_random_key())

d, (W, V, H, R), _ = cmtf_ssd(
    J, Y, X, rank, verbose=1, max_iters=20, random_state=get_random_key(),
    lam=0.1,
    smoothing=0.1,
    gamma=5.0,
)

x = jax.random.uniform(get_random_key(), m)
print(f(x))
print(d(x))