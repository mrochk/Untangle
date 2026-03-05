import jax, jax.numpy as jnp
from scipy.interpolate import make_smoothing_spline

from jaxtyping import Float, Array
from beartype.typing import Callable

from untangle.utils import make_polynomials

def inference_polynomial(
    W: Float[Array, 'n r'], 
    V: Float[Array, 'm r'], 
    coefs: Float[Array, 'r d'],
) -> Callable:
    return lambda x: W @ make_polynomials(coefs)(V.T @ x)

def make_g(internals):
    return lambda u: jnp.array([gi(ui) for gi, ui in zip(internals, u)])

def inference(W, V, g):
    return lambda x: W @ g(V.T @ x)

def normalize_columns_V(W: Float[Array, 'n r'], V: Float[Array, 'm r']):
    rank = W.shape[1]
    for i in range(rank):
        colV, colW = V[:, i], W[:, i]
        norm = jnp.linalg.norm(colV) + 1e-12
        V = V.at[:, i].set(colV / norm)
        W = W.at[:, i].set(colW * norm)
    return W, V

def fit_internal(u_s, r_s):
    return make_smoothing_spline(u_s, r_s)

def fit_internal_derivative(u_s, h_s, r_s):
    dg = make_smoothing_spline(u_s, h_s)
    g_bias = dg.antiderivative()
    c0 = (r_s - g_bias(u_s)).mean()
    return lambda x: g_bias(x) + c0

def fit_internals(U, H, R, use: str = 'R'):
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
