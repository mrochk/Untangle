import jax, jax.numpy as jnp, unittest
from untangle import algorithm
from untangle.utils import collect_information, function_error

def target(x):
    a, b = x
    return jnp.array([
        a**3 + b**2 + a*b,
        b**3 + a**2 + a*b,
    ])

class TestEasy(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.key(0)
        self.rank = 4
        self.ninits = 5
        self.X, self.Y, self.J = collect_information(target, 30, self.key)

    def _test_algorithm(self, algo):
        okay = False
        for k in jax.random.split(self.key, self.ninits):
            decoupling, _ = algo(self.X, self.Y, self.J, self.rank, key=k)
            e1, e2 = function_error(target, decoupling, self.X)
            if e1 < 10 and e2 < 10: okay = True; break
        self.assertTrue(okay)

    def test_basic(self): self._test_algorithm(algorithm.basic_decoupling)
    def test_ctd_polynomial(self): self._test_algorithm(algorithm.ctd_polynomial)
    def test_cmtf_bsd(self): self._test_algorithm(algorithm.cmtf_bsd)
    def test_cmtf_psd(self): self._test_algorithm(algorithm.cmtf_psd)
