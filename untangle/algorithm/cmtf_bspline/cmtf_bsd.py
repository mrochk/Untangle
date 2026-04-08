import jax, jax.numpy as jnp
from scipy.interpolate import BSpline
from functools import partial
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Tuple

from untangle.algorithm import Decoupling
from untangle.utils import cpd_error
from untangle import _ops as ops
from untangle._common import (
    get_random_key,
    make_log,
    fit_internals, 
    initialize,
    make_internals, 
)

@jaxtyped(typechecker=beartype)
def cmtf_bsd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    gamma: float = 0.1,
    iterations: int = 100,
    dof: int = 12,
    degree: int = 3,
    key: Array = get_random_key(),
    verbose: int = 0,
) -> Decoupling:

    prefix = '|CMTF-BSD| ->'
    log = make_log(verbose, prefix)

    best_errors = []

    W, V, H, R = initialize(J, rank, key, with_R=True)

    J0 = ops.unfold_kolda(J, 0).T
    J1 = ops.unfold_kolda(J, 1).T
    J2 = ops.unfold_kolda(J, 2).T

    for iteration in tqdm(range(iterations), desc=prefix):
        W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, Y, gamma)
        V = ops.lstsq(ops.khatri_rao(H, W), J1)
        W, V = ops.normalize_columns_V(W, V)

        H = ops.lstsq(ops.khatri_rao(V, W), J2)
        R = ops.lstsq(W, Y.T)

        Z = X @ V
        H, R = bsplines_projection(H, R, Z, dof, degree, gamma)

        error = cpd_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {iterations}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    Z = X @ V

    internals = make_internals(fit_internals(Z, H, R))

    return Decoupling(best, internals)

def bsplines_projection(
    H: Float[Array, 'N r'],
    R: Float[Array, 'N r'],
    U: Float[Array, 'N r'],
    dof: int, 
    degree: int,
    gamma: float,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        knots = determine_knots(u, dof, degree)

        B = design_matrix(u, knots, degree)
        dB = design_dmatrix(u, knots, degree)

        coefs = ops.cmtf_lstsq(dB, B, h, r, gamma)
        H, R = project(rank, coefs, B, dB, H, R)

    return H, R

def design_matrix(u: Float[Array, 'r'], knots: Array, degree: int):
    matrix = BSpline.design_matrix(u, knots, degree).toarray()
    matrix = jnp.concatenate([jnp.ones((matrix.shape[0], 1)), matrix], axis=1)
    return matrix

def design_dmatrix(u, knots, degree: int):
    n_basis = len(knots) - degree - 1
    dmatrix = jnp.zeros((len(u), n_basis))

    for i in range(n_basis):
        c = jnp.zeros(n_basis).at[i].set(1)
        bspline = BSpline(knots, c, degree)
        dmatrix = dmatrix.at[:, i].set(bspline.derivative(nu=1)(u))

    dmatrix = jnp.concatenate([jnp.zeros((dmatrix.shape[0], 1)), dmatrix], axis=1)
    return dmatrix

@jax.jit(static_argnames=('dof', 'degree'))
def determine_knots(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    internals = dof - degree + 1

    qs = jnp.linspace(0, 1, internals)
    knots = jnp.quantile(u, qs)
    knots = jax.vmap(partial(closest, u=u))(knots)

    begin = jnp.repeat(knots[0], degree)
    end = jnp.repeat(knots[-1], degree)
    return jnp.concat([begin, knots, end])

@jax.jit
def closest(knot, u):
    def forloop(i, args):
        min_dist, closest_x = args
        x = u[i]
        dist = jnp.abs(x - knot)
        return jax.lax.cond(
            dist < min_dist,
            lambda: (dist, x),
            lambda: (min_dist, closest_x),
        )

    _, closest_point = jax.lax.fori_loop(0, len(u), forloop, (jnp.inf, u[0]))
    return closest_point

@jax.jit
def project(i, coefs, B, dB, H, R):
    H = H.at[:, i].set(dB @ coefs)
    R = R.at[:, i].set(B @ coefs)
    return H, R
