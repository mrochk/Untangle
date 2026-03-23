import jax.numpy as jnp
from scipy.interpolate import make_smoothing_spline

def smoothing_splines_projection(H, R, Z):
    for rank in range(H.shape[1]):
        z = Z[:, rank]
        h = H[:, rank]
        r = R[:, rank]

        idx = jnp.argsort(z)
        inv = jnp.argsort(idx)

        z_sorted = z[idx]
        h_sorted = h[idx]
        r_sorted = r[idx]

        dspline = make_smoothing_spline(z_sorted, h_sorted)
        spline = dspline.antiderivative()

        dm = dspline(z_sorted)

        m = spline(z_sorted)
        m = m + jnp.mean(r_sorted - m)

        dm = dm[inv]
        m  = m[inv]

        H = H.at[:, rank].set(dm)
        R = R.at[:, rank].set(m)

    return H, R
