import jax, jax.numpy as jnp
from scipy.interpolate import BSpline
from jax.scipy.optimize import minimize

from functools import partial
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Tuple, Optional

from untangle.algorithm import Decoupling
from untangle.utils import cpd_error
from untangle import _ops as ops
from untangle._common import (
    get_random_key,
    make_log,
    bspline_project,
    fit_internals, 
    initialize,
    make_internals, 
)

@jaxtyped(typechecker=beartype)
def cmtf_psd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    gamma: float = 0.1,
    niters: int = 100,
    dof: int = 12,
    degree: int = 3,
    verbose: int = 0,
    key: Optional[Array] = None,
) -> Decoupling:

    if key is None: key = get_random_key()

    prefix = '|CMTF-PSD| ->'
    log = make_log(verbose, prefix)

    best_errors = []

    W, V, H, R = initialize(J, rank, key, with_R=True)

    J0 = ops.unfold_kolda(J, 0).T
    J1 = ops.unfold_kolda(J, 1).T
    J2 = ops.unfold_kolda(J, 2).T

    for iteration in tqdm(range(niters), desc=prefix):
        W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, Y, gamma)
        V = ops.lstsq(ops.khatri_rao(H, W), J1)
        W, V = ops.normalize_columns_V(W, V)

        H = ops.lstsq(ops.khatri_rao(V, W), J2)
        R = ops.lstsq(W, Y.T)

        Z = X @ V
        H, R = psplines_projection(H, R, Z, dof, degree, gamma)

        error = cpd_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {niters}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    Z = X @ V

    internals = make_internals(fit_internals(Z, H, R))

    return Decoupling(best, internals)

def psplines_projection(
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
        B  = design_matrix(u, knots, degree)
        dB = design_dmatrix(u, knots, degree)

        n_basis = B.shape[1]
        D = second_diff_matrix(n_basis)

        A = jnp.vstack([dB, gamma * B])
        y = jnp.concatenate([h, gamma * r])

        lam = gcv(A, y, D, n=len(u))

        # better than using solve(), why ?
        A = jnp.vstack([A, jnp.sqrt(lam) * D])
        y = jnp.concatenate([y, jnp.zeros(D.shape[0])])
        coefs = jnp.linalg.lstsq(A, y)[0]

        H, R = bspline_project(rank, coefs, B, dB, H, R)

    return H, R

def design_matrix(u: Float[Array, 'r'], knots: Array, degree: int):
    matrix = BSpline.design_matrix(u, knots, degree).toarray()
    return jnp.array(matrix)

def design_dmatrix(u, knots, degree: int):
    n_basis = len(knots) - degree - 1
    dmatrix = jnp.zeros((len(u), n_basis))

    for i in range(n_basis):
        c = jnp.zeros(n_basis).at[i].set(1)
        bspline = BSpline(knots, c, degree)
        dmatrix = dmatrix.at[:, i].set(bspline.derivative(nu=1)(u))

    return dmatrix

@jax.jit(static_argnames=('dof', 'degree'))
def determine_knots(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    internals = dof - degree + 1

    knots = jnp.linspace(jnp.min(u), jnp.max(u), internals)

    begin = jnp.repeat(knots[0], degree)
    end   = jnp.repeat(knots[-1], degree)
    return jnp.concat([begin, knots, end])

@jax.jit(static_argnames='n')
def second_diff_matrix(n: int) -> Array:
    D1 = jnp.diff(jnp.eye(n), axis=0)
    D2 = jnp.diff(D1, axis=0)
    return D2

@jax.jit
def gcv_score(log_lam: Array, A: Array, y: Array, D2: Array, n: int) -> Array:
    lam     = 10.0 ** log_lam
    n_basis = A.shape[1]

    AtA   = A.T @ A
    P     = lam * D2.T @ D2 + 1e-8 * jnp.eye(n_basis)
    M     = AtA + P
    M_inv = jnp.linalg.inv(M)

    coefs     = M_inv @ A.T @ y
    residuals = y - A @ coefs
    rss       = jnp.sum(residuals ** 2)

    df  = jnp.trace(M_inv @ AtA)
    gcv = (n * rss) / (n - df) ** 2   # n = actual observations, not 2N
    return gcv

@jax.jit
def gcv(A: Array, y: Array, D: Array, n: int) -> Array:
    log_lams = jnp.linspace(-8, 3, 1000)
    scores   = jax.vmap(lambda ll: gcv_score(ll, A, y, D, n))(log_lams)
    return 10.0 ** log_lams[jnp.argmin(scores)]
