import jax, jax.numpy as jnp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from untangle.algorithm import cmtf_bsd
from untangle.algorithm.cmtf_ssd.cmtf_ssd import cmtf_ssd
from untangle.algorithm.cmtf_ssd.spline import *
from untangle.utils import collect_information, get_random_key
from untangle.utils.testing import *

import warnings

warnings.simplefilter('ignore')

num = 50

m, r, rank, f = POLYNOMIAL_2

# todo: create a f_scaled function

X, Y, J = collect_information(f, num, m, range=(0, 1), key=get_random_key())

y_min = jnp.min(Y, axis=0)
y_max = jnp.max(Y, axis=0)
y_range = y_max - y_min

def f_scaled(x): return (f(x) - y_min) / y_range

X, Y, J = collect_information(f_scaled, num, m, range=(0, 1), key=get_random_key())

inf, best, errs = cmtf_ssd(J, Y, X, rank, iterations=10, verbose=1)

x = jax.random.uniform(get_random_key(), shape=m)

print(f(x))
print(inf(x) * y_range + y_min)

#def f(x): return 2*x**3 - 3*x - 1

#m = 1
#n = 1

#num = 100

#X, Y, J = collect_information(f, num, m, range=(0, 1), key=get_random_key())

#idx = jnp.argsort(X.flatten())
#inv = jnp.argsort(idx)

#Xs = X.flatten()[idx]
#Ys = Y.flatten()[idx]
#Js = J.flatten()[idx]

#Ys_noisy = Ys + jax.random.normal(get_random_key(), num)

#D = finite_difference_matrix(Xs)

#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#kernel = RBF() + WhiteKernel()
#gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
#gp.fit(Xs.reshape(-1, 1), Ys_noisy)

## Differentiate the posterior mean numerically (or analytically with GPJax)
#f_mean = lambda x: gp.predict(x.reshape(-1, 1))
#DYs = D @ f_mean(Xs)  # or use a proper GP derivative kernel

#plt.scatter(Xs, Ys, color='blue', label='y')
#plt.scatter(Xs, DYs, color='red', label='D@y')
#plt.scatter(Xs, Js, color='purple', label='y\'')
#plt.legend()
#plt.savefig('f.png')
#plt.close()

## todo: plot distribution of DYs
#plt.hist(DYs, bins=30, color='red', edgecolor='black', alpha=0.7, density=True)
#plt.axvline(jnp.mean(DYs), color='black', linestyle='--', label=f'mean={jnp.mean(DYs):.2f}')
#plt.axvline(jnp.std(DYs), color='purple', linestyle='--', label=f'std')
#plt.axvline(-jnp.std(DYs), color='purple', linestyle='--', label=f'std')
#plt.xlabel('D @ y')
#plt.ylabel('Density')
#plt.title('Distribution of DYs')
#plt.legend()
#plt.savefig('DYs_dist.png')
#plt.close()