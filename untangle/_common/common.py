import random
import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array, Float
from scipy.interpolate import BSpline, make_smoothing_spline
from functools import partial
from beartype import beartype 
from beartype.typing import Callable 

from untangle import _ops as ops

def make_log(verbose: int, prefix: str = '') -> Callable[[], None]:
    def log(*args):
        if verbose <= 0: return
        print(prefix, end='')
        print(*args, flush=True)
    return log

def get_random_key() -> Array:
    return jax.random.key(random.randint(0, int(1e10)))

def find_number_inputs(function: Callable):
    assert callable(function)
    m = 1
    while True:
        try: function(jnp.zeros(m)); return m
        except ValueError: m += 1

### factors initialization

@jaxtyped(typechecker=beartype)
def initialize(tensor: Float[Array, 'n m N'], rank: int, key: Array, with_R: bool = False):
    n, m, N = tensor.shape
    keys = jax.random.split(key, num=4)

    W = jax.random.normal(keys[0], shape=(n, rank))
    V = jax.random.normal(keys[1], shape=(m, rank))
    H = jax.random.normal(keys[2], shape=(N, rank))

    if not with_R: return (W, V, H)

    R = jax.random.normal(keys[3], shape=(N, rank))
    return (W, V, H, R)

### CP decomposition

def cpd_stopping_criterion(diff: float, tol: float, norm: float) -> bool:
    return diff < tol * norm

@jaxtyped(typechecker=beartype)
def solve_cpd_subproblem(
    unfolded: Array, 
    W: Float[Array, 'n r'], 
    V: Float[Array, 'm r'],
    H: Float[Array, 'N r'], 
    mode: int,
):
    assert 0 <= mode <= 2
    match mode:
        case 0: return ops.cpd_als_solve(unfolded, H, V)
        case 1: return ops.cpd_als_solve(unfolded, H, W)
        case 2: return ops.cpd_als_solve(unfolded, V, W)

### stuff related to fitting internals

def make_polynomial(coefs: Float[Array, 'd']) -> Callable:
    return partial(jnp.polyval, p=jnp.flip(coefs))

def make_polynomials(coefs: Float[Array, 'n d']) -> Callable:
    polynomials = [make_polynomial(c) for c in coefs]
    return (lambda x: jnp.array([f(x=xi) for f, xi in zip(polynomials, x)]))

def make_internals(internals):
    return lambda u: jnp.array([gi(ui) for gi, ui in zip(internals, u)])

def fit_internal(z_s, h_s, r_s):
    dg: BSpline = make_smoothing_spline(z_s, h_s)

    g_bias: BSpline = dg.antiderivative()
    bias = jnp.median(r_s - g_bias(z_s))

    def g(x): return g_bias(x) + bias
    return g

def fit_internals(Z, H, R):
    internals = []

    for rank in range(Z.shape[1]):
        z, h, r = Z[:, rank], H[:, rank], R[:, rank]
        idx = jnp.argsort(z)
        z_s, h_s, r_s = z[idx], h[idx], r[idx]

        g = fit_internal(z_s, h_s, r_s)

        internals.append(g)

    return internals
