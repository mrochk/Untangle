import jax, jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt

# WITHOUT DERIVATIVES

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

def fit(x, y, lam: float):
    idx = jnp.argsort(x)
    inv = jnp.argsort(idx)

    xs = x[idx]
    ys = y[idx]

    delta, W = get_delta_W(xs)
    m = solve(delta, W, ys, lam)

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

# WITH DERIVATIVES

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
def solve_with_derivatives(delta, W, D, y, dy, dy_smooth, gamma, lam):
    A = delta.T @ jnp.linalg.solve(W, delta)
    lhs = jnp.eye(len(y)) + lam*A + gamma*(D.T @ D)
    rhs = y + gamma*(D.T @ dy_smooth)
    return jnp.linalg.solve(lhs, rhs)

def fit_with_derivatives(x, y, dy, gamma, lam, dy_smooth):
    delta, W = get_delta_W(x)
    D = finite_difference_matrix(x)
    m = solve_with_derivatives(delta, W, D, y, dy, dy_smooth, gamma, lam)
    return m, D @ m, D

@jax.jit
def gcv_score_derivative(delta, W, D, y, dy, gamma, lam):
    n = len(y)
    A = delta.T @ jnp.linalg.solve(W, delta)
    lhs = jnp.eye(n) + lam*A + gamma*(D.T @ D)
    H = jnp.linalg.inv(lhs)
    m = H @ (y + gamma*(D.T @ dy))
    residuals = y - m
    dof = jnp.trace(H)
    return jnp.mean(residuals**2) / ((1 - dof/n)**2)

def find_lambda_with_derivatives(x, y, dy, gamma=1.0, lambdas=jnp.logspace(-3, 3, 50)):
    delta, W = get_delta_W(x)
    D = finite_difference_matrix(x)

    gcv_over_lambdas = jax.vmap(
        lambda lam: gcv_score_derivative(delta, W, D, y, dy, gamma, lam)
    )
    gcv_vals = gcv_over_lambdas(lambdas)

    best_idx = jnp.argmin(gcv_vals)
    return lambdas[best_idx]
