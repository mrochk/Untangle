import unittest
import jax, jax.numpy as jnp

from untangle.algorithm import cmtf_bsd
from untangle.utils import get_random_key, collect_information
from untangle.utils.testing import POLYNOMIAL_1, POLYNOMIAL_2

class TestDecouplingBasic(unittest.TestCase):
    def setUp(self): print()

    def test_simple_function(self):
        '''Function given in example (4) of the paper.'''

        m, n, rank, f = POLYNOMIAL_1
        N = 10

        X, Y, J = collect_information(f, N, m)

        decoupling, _, _ = cmtf_bsd(J, Y, X, rank, iterations=100)

        x = jax.random.uniform(get_random_key(), shape=m)
        truth, decoupled = f(x), decoupling(x)

        error = jnp.linalg.norm(truth - decoupled) / jnp.linalg.norm(truth)
        self.assertLess(error, 0.2)

    def test_simple_function2(self):
        m, n, rank, f = POLYNOMIAL_2
        N = 20

        X, Y, J = collect_information(f, N, m)
        decoupling, _, _ = cmtf_bsd(J, Y, X, rank, iterations=100)

        x = jax.random.uniform(get_random_key(), shape=m)
        truth = f(x); decoupled = decoupling(x)

        error = jnp.linalg.norm(truth - decoupled) / jnp.linalg.norm(truth)
        self.assertLess(error,  0.1)
