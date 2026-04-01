import numpy as np
from scipy.optimize import minimize_scalar
import jax, jax.numpy as jnp
import matplotlib.pyplot as plt

def _make_integration_matrix(z: np.ndarray) -> np.ndarray:
    N = len(z)
    U = np.zeros((N, N))
    for i in range(1, N):
        U[i] = U[i - 1]
        delta = z[i] - z[i - 1]
        U[i, i - 1] += delta / 2
        U[i, i]     += delta / 2
    return U

def _make_penalty_matrix(z: np.ndarray) -> np.ndarray:
    N = len(z)
    h = np.diff(z)          # h[j] = z[j+1] - z[j],  shape (N-1,)
    n = N - 2               # number of interior knots

    # C ─ second-order divided differences
    C = np.zeros((n, N))
    for j in range(n):
        C[j, j]   =  1.0 / h[j]
        C[j, j+1] = -1.0 / h[j] - 1.0 / h[j+1]
        C[j, j+2] =  1.0 / h[j+1]

    # M ─ tridiagonal mass matrix
    M = np.zeros((n, n))
    for j in range(n):
        M[j, j] = (h[j] + h[j+1]) / 3.0
    for j in range(n - 1):
        M[j,   j+1] = h[j+1] / 6.0
        M[j+1, j  ] = h[j+1] / 6.0

    K = C.T @ np.linalg.solve(M, C)
    return K

def find_lambda_gcv(z: np.ndarray, h: np.ndarray, r: np.ndarray, gamma: float) -> float:
    U   = _make_integration_matrix(z)
    K   = _make_penalty_matrix(z)
    N   = len(h)
    UtU = U.T @ U
    rhs = h + gamma * (U.T @ r)

    def gcv(log_lam: float) -> float:
        lam   = np.exp(log_lam)
        A     = np.eye(N) + gamma * UtU + lam * K

        try:
            f_hat  = np.linalg.solve(A, rhs)
            AinvK  = np.linalg.solve(A, K)
            tr_H   = N - lam * np.trace(AinvK)
        except np.linalg.LinAlgError:
            return np.inf

        res_h   = h - f_hat
        res_r   = r - U @ f_hat
        rss     = np.sum(res_h ** 2) + gamma * np.sum(res_r ** 2)

        n_total = 2 * N
        denom   = n_total - tr_H
        if denom <= 0:
            return np.inf

        return n_total * rss / denom ** 2

    result = minimize_scalar(gcv, bounds=(-10, 10), method='bounded')
    return float(np.exp(result.x))

def make_combined_smoothing_spline(z, h, r, gamma, lam):
    U = _make_integration_matrix(z)
    K = _make_penalty_matrix(z)
    N = len(h)

    A     = np.eye(N) + gamma * (U.T @ U) + lam * K
    f_hat = np.linalg.solve(A, h + gamma * U.T @ r)
    r_hat = U @ f_hat

    return f_hat, r_hat

def smoothing_splines_projection(H, R, Z, gamma: float):
    fig, ax = plt.subplots(H.shape[1], 2, figsize=(7, 7))

    for rank in range(H.shape[1]):
        z = np.array(Z[:, rank])
        h = np.array(H[:, rank])
        r = np.array(R[:, rank])

        idx = np.argsort(z)
        inv = np.argsort(idx)

        z_sorted = z[idx]
        h_sorted = h[idx]
        r_sorted = r[idx]

        lam = find_lambda_gcv(z_sorted, h_sorted, r_sorted, gamma)
        print(lam)
        dm, m = make_combined_smoothing_spline(z_sorted, h_sorted, r_sorted, gamma, lam)

        ax[rank, 0].scatter(z_sorted, h_sorted, color='red')
        ax[rank, 0].scatter(z_sorted, r_sorted, color='green')

        ax[rank, 1].scatter(z_sorted, h_sorted, color='red')
        ax[rank, 1].scatter(z_sorted, r_sorted, color='green')

        ax[rank, 1].scatter(z_sorted, dm, color='purple')
        ax[rank, 1].scatter(z_sorted, m, color='purple')

        dm = dm[inv]
        m = m[inv]

        H = H.at[:, rank].set(jnp.array(dm))
        R = R.at[:, rank].set(jnp.array(m))

    plt.savefig('plot.png')
    plt.close()

    return H, R
