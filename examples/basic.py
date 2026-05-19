import jax.numpy as jnp

from untangle.algorithm import basic_decoupling, ctd_polynomial
from untangle.utils import collect_information, function_error

import matplotlib.pyplot as plt

import jax

key = jax.random.key(3)

def f(x):
    a, b = x
    return jnp.array([
        2*a**3 + b**2 + 1,
        3*b**2 + 2*a - 1,
    ])
    
m = 2
N = 100
degree = 3
rank = 3

colors = [
    '#6C8EBF',
    #'#82B366',
    '#D79B00',
    '#B85450',
]

X, Y, J = collect_information(f, N, key=key)

print('Without polynomial constraint:')
f_hat = basic_decoupling(X, Y, J, rank, degree, key=key)
errors = function_error(f, f_hat, X)
print('Errors:', errors)

Z = X @ f_hat.V
H = f_hat.H

fig, ax = plt.subplots(rank, figsize=(7, 7))

for r in range(rank):
    z = Z[:, r]
    h = H[:, r]
    ax[r].scatter(z, h, color=colors[r])
    ax[r].set_xticks([])
    ax[r].set_yticks([])
    ax[r].set_ylabel(rf'$g_{{{r}}}(\mathbf{{z}}_{{{r}}})$')
    ax[r].set_xlabel(rf'$\mathbf{{z}}_{{{r}}}$')

plt.tight_layout()
plt.savefig('internals.png')
plt.close()

print('\nWith polynomial constraint:')
f_hat = ctd_polynomial(X, Y, J, rank, degree, key=key)
errors = function_error(f, f_hat, X)
print('Errors:', errors)

Z = X @ f_hat.V
H = f_hat.H

fig, ax = plt.subplots(rank, figsize=(7, 7))

for r in range(rank):
    z = Z[:, r]
    h = H[:, r]
    ax[r].scatter(z, h, color=colors[r])
    ax[r].set_xticks([])
    ax[r].set_yticks([])
    ax[r].set_ylabel(rf'$g_{{{r}}}(\mathbf{{z}}_{{{r}}})$')
    ax[r].set_xlabel(rf'$\mathbf{{z}}_{{{r}}}$')

plt.tight_layout()
plt.savefig('internals_ctd.png')
plt.close()