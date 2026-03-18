import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped, Float, Array

def get_delta_W(x):
    n = len(x)
    h = jnp.diff(x)

    delta = jnp.zeros((n-2, n))
    for i in range(n-2):
        delta = delta.at[i, i].set(1/h[i])
        delta = delta.at[i, i+1].set(-1/h[i] - 1/h[i+1])
        delta = delta.at[i, i+2].set(1/h[i+1])

    W = jnp.zeros((n-2, n-2))
    for i in range(n-2):
        W = W.at[i, i].set((h[i] + h[i+1]) / 3)
        if i > 0: W = W.at[i-1, i].set(h[i] / 6)
        if i+1 < n-2: W = W.at[i, i+1].set(h[i] / 6)

    return delta, W

def get_penalty_matrix(x):
    delta, W = get_delta_W(x)
    return delta.T @ jnp.linalg.solve(W, delta)

def finite_difference_matrix(x):
    n = x.shape[0]
    D = jnp.zeros((n, n))

    D = D.at[0, 0].set(-1 / (x[1] - x[0]))
    D = D.at[0, 1].set( 1 / (x[1] - x[0]))

    for i in range(1, n - 1):
        D = D.at[i, i - 1].set(-1 / (x[i + 1] - x[i - 1]))
        D = D.at[i, i + 1].set( 1 / (x[i + 1] - x[i - 1]))

    D = D.at[-1, -2].set(-1 / (x[-1] - x[-2]))
    D = D.at[-1, -1].set( 1 / (x[-1] - x[-2]))

    return D

@jax.jit
def gcv(lam, x, y, dy):
    n = len(x)
    D = finite_difference_matrix(x)
    A = get_penalty_matrix(x)
    I = jnp.eye(n)
    
    left = I + D.T @ D + lam * A
    right = jnp.hstack([I, D.T])
    
    A_smoother = jnp.linalg.solve(left, right)

    m = A_smoother[:, :n] @ y + A_smoother[:, n:] @ dy
    
    rss = jnp.mean((y - m)**2)
    df = jnp.trace(A_smoother[:, :n])
    denom = (1 - df / n) ** 2
    return rss / denom

def get_lambda_gcv(x, y, dy):
    lambdas = jnp.linspace(0, 1.0, 10)
    lam_idx = jnp.argmin(jnp.array([gcv(lam, x, y, dy) for lam in lambdas]))
    lam = lambdas[lam_idx]
    return lam

@jaxtyped(typechecker=beartype)
def fit_smoothing_spline(
    x: Float[Array, 'n'], 
    y: Float[Array, 'n'],
    dy: Float[Array, 'n'],
) -> Tuple[Float[Array, 'n'], Float[Array, 'n']]:

    # https://en.wikipedia.org/wiki/Smoothing_spline

    lam = get_lambda_gcv(x, y, dy)
    D = finite_difference_matrix(x)
    A = get_penalty_matrix(x)
    I = jnp.eye(len(x))

    m = jnp.linalg.solve(I + D.T@D + lam*A, y + D.T@dy)
    return m, D @ m
