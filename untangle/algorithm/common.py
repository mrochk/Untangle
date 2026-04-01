import jax, jax.numpy as jnp
from functools import partial
from scipy.interpolate import BSpline, make_interp_spline, make_smoothing_spline

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Callable

def make_internals(internals):
    return lambda u: jnp.array([gi(ui) for gi, ui in zip(internals, u)])

@jax.jit
def normalize_columns_V(W: Float[Array, 'n r'], V: Float[Array, 'm r']):
    rank = W.shape[1]

    def _(i, W_V):
        W, V = W_V
        colV, colW = V[:, i], W[:, i]
        norm = jnp.linalg.norm(colV) + 1e-12
        V = V.at[:, i].set(colV / norm)
        W = W.at[:, i].set(colW * norm)
        return W, V

    return jax.lax.fori_loop(0, rank, _, (W, V))

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

@jax.jit
def lstsq(X, Y): return jnp.linalg.lstsq(X, Y)[0].T

@jax.jit
@jaxtyped(typechecker=beartype)
def cmtf_lstsq(X1, X2, Y1, Y2, lam):
    X = jnp.concatenate([X1, lam*X2], axis=0)
    Y = jnp.concatenate([Y1, lam*Y2], axis=0)
    return jnp.linalg.lstsq(X, Y)[0].T

def make_polynomial(coefs: Float[Array, 'd']) -> Callable:
    return partial(jnp.polyval, jnp.flip(coefs))

def make_polynomials(coefs: Float[Array, 'n d']) -> Callable:
    polynomials = [make_polynomial(c) for c in coefs]
    return (lambda x: jnp.array([f(xi) for f, xi in zip(polynomials, x)]))
