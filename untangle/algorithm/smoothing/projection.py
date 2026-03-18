import jax, jax.numpy as jnp
from scipy.interpolate import make_smoothing_spline

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Tuple

import matplotlib.pyplot as plt

def smoothing_spline(x, y):
    try: return make_smoothing_spline(x, y)
    except ValueError:
        x = jnp.unique(x)
        y = jnp.unique(y)
        return make_smoothing_spline(x, y)

@jax.jit
def sort_information(hj, rj, uj):
    idx = jnp.argsort(uj)
    inv = jnp.argsort(idx)
    u, h, r = uj[idx], hj[idx], rj[idx]
    return h, r, u, inv

@jaxtyped(typechecker=beartype)
def smoothing_splines_projection(
    H: Float[Array, 'N r'], 
    R: Float[Array, 'N r'],
    U: Float[Array, 'N r'], 
    use: str,
    gamma = None,
    plot: bool = False,
    plot_title: str = None,
    plot_path: str = None,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    rank = H.shape[1]

    if plot: fig, ax = plt.subplots(rank, 2, figsize=(10, 20))

    for j in range(rank):

        h, r, u = H[:, j], R[:, j], U[:, j]
        h_sorted, r_sorted, u_sorted, inv = sort_information(h, r, u)

        match use:

            case 'both': 
                pass

            case 'first':
                dss = smoothing_spline(u_sorted, h_sorted)
                dm = jnp.array(dss(u_sorted))
                ss = dss.antiderivative()
                m  = jnp.array(ss(u_sorted))
                m  = m + jnp.median(r_sorted - m)

            case 'zero':
                ss = smoothing_spline(u_sorted, r_sorted)
                m = jnp.array(ss(u_sorted))
                dss = ss.derivative()
                dm  = jnp.array(dss(u_sorted))

            case _: raise Exception()

        H = H.at[:, j].set(dm[inv])
        R = R.at[:, j].set(m[inv])

        if plot:
            ax[j, 0].scatter(u, r, color='blue', label='zero-th order info')
            ax[j, 0].scatter(u, h, color='red', label='first order info')
            ax[j, 1].scatter(u, m[inv], color='blue', label='m')
            ax[j, 1].scatter(u, dm[inv], color='red', label='D@m')
            ax[j, 1].plot(u_sorted, m, color='blue')
            ax[j, 1].plot(u_sorted, dm, color='red')
            ax[j, 0].set_ylim((min(r.min(), h.min())-1, max(r.max(), h.max())+1))
            ax[j, 1].set_ylim((min(r.min(), h.min())-1, max(r.max(), h.max())+1))
            ax[j, 0].legend()
            ax[j, 1].legend()
    
    if plot:
        plt.suptitle(plot_title)
        plt.tight_layout()
        fig.savefig(plot_path)
        plt.close()

    return H, R
