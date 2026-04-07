'''
https://tensorly.org/dev/user_guide/tensor_decomposition.html
'''

import jax, jax.numpy as jnp
import matplotlib.pyplot as plt

from untangle.decomposition import cpd
from untangle.utils import cpd_reconstruct

key = jax.random.key(0)

tensor = jnp.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                    [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                    [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                    [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

tensor = tensor.reshape((1, tensor.shape[0], tensor.shape[1]))
rank = 2

factors, weights, errs = cpd(tensor, rank, key=key)

print('CPD Errors:', errs)

fig, ax = plt.subplots(2)

rtensor = cpd_reconstruct(factors, weights)

ax[0].matshow(tensor[0])
ax[1].matshow(rtensor[0])

fig.savefig('plots/cpd.png')
