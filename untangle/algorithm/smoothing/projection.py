import jax.numpy as jnp
from scipy.interpolate import make_smoothing_spline, make_splrep
from scipy.interpolate import BSpline
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
import numpy as np

import matplotlib.pyplot as plt

def smoothing_splines_projection(H, R, Z, gamma: float, iteration: int = 0):
    fig, ax = plt.subplots(H.shape[1], 2, figsize=(10, 7))

    for rank in range(H.shape[1]):
        z = Z[:, rank]
        h = H[:, rank]
        r = R[:, rank]

        idx = jnp.argsort(z)
        inv = jnp.argsort(idx)

        z_sorted = z[idx]
        h_sorted = h[idx]
        r_sorted = r[idx]

        #dss = make_splrep(z_sorted, h_sorted)
        dss, lam = make_smoothing_spline(z_sorted, h_sorted)

        dss, ss = make_combined_smoothing_spline(z_sorted, h_sorted, r_sorted, lam, gamma)
        #ss = dss.antiderivative()
        
        dm = dss(z_sorted)
        m = ss(z_sorted)
        #m = m + jnp.median(r_sorted - m)

        dm = dm[inv]
        m = m[inv]

        H = H.at[:, rank].set(dm)
        R = R.at[:, rank].set(m)

        ax[rank, 0].scatter(z, h, color='red')
        ax[rank, 0].scatter(z, r, color='blue')
        ax[rank, 1].plot(z_sorted, dm[idx], color='red')
        ax[rank, 1].plot(z_sorted, m[idx], color='blue')
        ax[rank, 1].plot(z_sorted, h_sorted, color='purple')
        ax[rank, 1].plot(z_sorted, r_sorted, color='purple')
        ax[rank, 1].scatter(z, dm, color='red')
        ax[rank, 1].scatter(z, m, color='blue')

    fig.tight_layout()
    fig.savefig(f'plots/{iteration}.png')
    plt.close(fig)

    return H, R

def make_combined_smoothing_spline(x, h, r, lam, gamma):
    k = 3
    t = np.concatenate([[x[0]] * (k + 1), x[1:-1], [x[-1]] * (k + 1)])
    m = len(t) - k - 1

    B = BSpline.design_matrix(x, t, k).toarray()

    splines = [BSpline(t, np.eye(m)[j], k) for j in range(m)]
    antiderivs = [s.antiderivative() for s in splines]
    A = np.column_stack([np.ones(len(x))] + [a(x) - a(x[0]) for a in antiderivs])

    x_q = np.linspace(x[0], x[-1], 10 * len(x))
    D2 = np.column_stack([s.derivative(2)(x_q) for s in splines])
    Omega_s = (x_q[1] - x_q[0]) * D2.T @ D2
    Omega = np.block([[np.zeros((1, 1)), np.zeros((1, m))],
                      [np.zeros((m, 1)), Omega_s]])

    B_ext = np.column_stack([np.zeros(len(x)), B])
    c = solve(gamma * A.T @ A + B_ext.T @ B_ext + lam * Omega,
              gamma * A.T @ r + B_ext.T @ h,
              assume_a='sym')

    g_prime = BSpline(t, c[1:], k)
    g_anti  = g_prime.antiderivative()
    g       = lambda u: g_anti(u) - g_anti(x[0]) + c[0]

    return g_prime, g
