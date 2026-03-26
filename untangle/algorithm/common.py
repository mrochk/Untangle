import jax, jax.numpy as jnp
from functools import partial
from scipy.interpolate import make_smoothing_spline, make_interp_spline

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Callable

def make_internals(internals):
    return lambda u: jnp.array([gi(ui) for gi, ui in zip(internals, u)])

def inference(W, V, g):
    return lambda x: W @ g(V.T @ x)

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

def prepare_spline_inputs(u_s, *ys):
    u_unique, inverse = jnp.unique(u_s, return_inverse=True)
    ys_unique = [jnp.array([y[inverse == i].mean() for i in range(len(u_unique))]) for y in ys]
    return (u_unique, *ys_unique)

def safe_smoothing_spline(u_s, y_s):
    try: return make_smoothing_spline(u_s, y_s)
    except ValueError:
        # Fallback: linear spline (always well-posed)
        return make_interp_spline(u_s, y_s, k=1)

def fit_internal(u_s, r_s):
    u_s, r_s = prepare_spline_inputs(u_s, r_s)
    return safe_smoothing_spline(u_s, r_s)

def fit_internal_derivative(u_s, h_s, r_s):
    u_s, h_s, r_s = prepare_spline_inputs(u_s, h_s, r_s)
    dg = safe_smoothing_spline(u_s, h_s)
    g_bias = dg.antiderivative()
    bias = jnp.median(r_s - g_bias(u_s))
    return lambda x: g_bias(x) + bias

def fit_internals(U, H, R, use: str = 'H'):
    internals = []

    for rank in range(U.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]
        idx = jnp.argsort(u)
        u_s, h_s, r_s = u[idx], h[idx], r[idx]

        match use:
            case 'R': g = fit_internal(u_s, r_s)
            case 'H': g = fit_internal_derivative(u_s, h_s, r_s)
            case _: raise Exception()

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
def lstsq(X, Y):
    return jnp.linalg.lstsq(X, Y)[0].T

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
