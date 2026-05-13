import random
import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array, Float
from bsplx import design_matrix, design_dmatrix, bspline_inference
from functools import partial
from beartype import beartype 
from beartype.typing import Callable

from untangle import _ops as ops

def make_log(verbose: int, prefix: str = '') -> Callable[[], None]:
    def log(*args):
        if verbose <= 0: return
        print(prefix, end=' ' if prefix else '')
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

@jax.jit
def bspline_project(i, coefs, B, dB, H, R):
    H = H.at[:, i].set(dB @ coefs)
    R = R.at[:, i].set(B @ coefs)
    return H, R

def make_polynomial(coefs: Float[Array, 'd']) -> Callable:
    return partial(jnp.polyval, p=jnp.flip(coefs))

def make_polynomials(coefs: Float[Array, 'n d']) -> Callable:
    polynomials = [make_polynomial(c) for c in coefs]
    return (lambda x: jnp.array([f(x=xi) for f, xi in zip(polynomials, x)]))

def make_internals(internals):
    return lambda u: jnp.array([gi(ui) for gi, ui in zip(internals, u)])

from scipy.interpolate import make_smoothing_spline

def fit_internal(z_s, h_s, r_s):
    try:
        dss = make_smoothing_spline(z_s, h_s)
    except:
        try:
            print(z_s)
            z_s, idx = jnp.unique(z_s, True)
            h_s = h_s[idx]
            r_s = r_s[idx]
            dss = make_smoothing_spline(z_s, h_s)
        except:
            def g(x): return jnp.zeros_like(x)
            return g
        
    ss = dss.antiderivative()

    bias = jnp.median(r_s - ss(z_s))
    c = jnp.array(ss.c)
    knots = jnp.array(ss.t)
    d = ss.k

    # ensure n_basis consistency: c must have len(knots) - d - 1 coefficients
    n_basis = len(knots) - d - 1
    c = c[:n_basis]

    def g(x): return bspline_inference(x, c, knots, d) + bias
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

def get_design_matrix(z, knots: Array, degree: int):
    matrix = design_matrix(z, knots, degree)
    matrix = jnp.concatenate([jnp.ones((matrix.shape[0], 1)), matrix], axis=1)
    return matrix

def get_design_dmatrix(z, knots, degree: int):
    dmatrix = design_dmatrix(z, knots, degree)
    dmatrix = jnp.concatenate([jnp.zeros((dmatrix.shape[0], 1)), dmatrix], axis=1)
    return dmatrix

def determine_knots(z: Float[Array, 'r'], dof: int, degree: int, method: str) -> Array:
    match method:
        case 'even': return _determine_knots_even(z, dof, degree)
        case 'quantile': return _determine_knots_quantiles(z, dof, degree)
        case _: raise ValueError()

@jax.jit(static_argnames=('dof', 'degree'))
def _determine_knots_even(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    internals = dof - degree + 1

    knots = jnp.linspace(jnp.min(u), jnp.max(u), internals)

    begin = jnp.repeat(knots[0], degree)
    end   = jnp.repeat(knots[-1], degree)
    return jnp.concat([begin, knots, end])

@jax.jit(static_argnames=('dof', 'degree'))
def _determine_knots_quantiles(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    internals = dof - degree + 1

    qs = jnp.linspace(0, 1, internals)
    knots = jnp.quantile(u, qs)
    knots = jax.vmap(partial(_closest, u=u))(knots)

    begin = jnp.repeat(knots[0], degree)
    end = jnp.repeat(knots[-1], degree)
    return jnp.concat([begin, knots, end])

@jax.jit
def _closest(knot, u):
    def forloop(i, args):
        min_dist, closest_x = args
        x = u[i]
        dist = jnp.abs(x - knot)
        return jax.lax.cond(
            dist < min_dist,
            lambda: (dist, x),
            lambda: (min_dist, closest_x),
        )

    _, closest_point = jax.lax.fori_loop(0, len(u), forloop, (jnp.inf, u[0]))
    return closest_point
