import jax, jax.numpy as jnp
from scipy.interpolate import CubicSpline

def get_delta_W(x):
    n = len(x)
    h = jnp.ediff1d(x)

    delta = jnp.zeros((n-2, n))
    for i in range(n-2):
        delta = delta.at[i, i].set(1 / h[i])
        delta = delta.at[i, i+1].set(-1/h[i] - 1/h[i+1])
        delta = delta.at[i, i+2].set(1/h[i+1])

    W = jnp.zeros((n-2, n-2))
    for i in range(n-2):
        W = W.at[i, i].set((h[i] + h[i+1]) / 3)
        if i > 0: W = W.at[i-1, i].set(h[i] / 6)
        if i+1 < n-2: W = W.at[i, i+1].set(h[i] / 6)

    return delta, W

@jax.jit
def solve(delta, W, y, lam: float):
    A = delta.T @ jnp.linalg.solve(W, delta)
    m = jnp.linalg.solve(jnp.eye(A.shape[0]) + lam*A, y)
    return m

@jax.jit
def solve_with_derivatives(delta, W, D, y, dy, gamma: float, lam: float):
    A = delta.T @ jnp.linalg.solve(W, delta)
    lhs = jnp.eye(A.shape[0]) + gamma * (D.T @ D) + lam * A
    rhs = y + gamma * (D.T @ dy)
    m = jnp.linalg.solve(lhs, rhs)
    return m

@jax.jit
def gcv_score(delta, W, y, lam):
    n = len(y)
    A = delta.T @ jnp.linalg.solve(W, delta)
    H = jnp.linalg.solve(jnp.eye(n) + lam*A, jnp.eye(n))
    m = H @ y
    residuals = y - m
    dof = jnp.trace(H)
    return jnp.mean(residuals**2) / ((1 - dof/n)**2)

def find_lambda(x, y, lambdas=jnp.logspace(-3, 3, 50)):
    delta, W = get_delta_W(x)
    scores = [gcv_score(delta, W, y, lam) for lam in lambdas]
    return lambdas[jnp.argmin(jnp.array(scores))]

def finite_difference_matrix(x):
    n = len(x)
    D = jnp.zeros((n, n))
    for i in range(1, n-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        D = D.at[i, i-1].set(-h2 / (h1 * (h1 + h2)))
        D = D.at[i, i].set((h2 - h1) / (h1 * h2))
        D = D.at[i, i+1].set(h1 / (h2 * (h1 + h2)))
    return D

def error(x, y, dy, gamma, lam):
    m = fit_with_deriv(x, y, dy, gamma, lam)
    spline = CubicSpline(x, m)
    dspline = spline.derivative()

    first = float(((spline(x) - y)**2).sum())
    second = float(((dspline(x) - dy)**2).sum())

    return first + second, first, second

def fit_with_deriv(x, y, dy, gamma, lam):
    delta, W = get_delta_W(x)
    D = finite_difference_matrix(x)
    m = solve_with_derivatives(delta, W, D, y, dy, gamma, lam)
    return m
