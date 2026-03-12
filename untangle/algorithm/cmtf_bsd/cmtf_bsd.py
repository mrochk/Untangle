import jax, jax.numpy as jnp
from scipy.interpolate import BSpline
from tqdm import tqdm

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

from untangle.ops import unfold_kolda, khatri_rao
from untangle.utils import get_random_key, relative_error, make_log
from untangle.algorithm.common import normalize_columns_V, fit_internals, make_g, inference

@jaxtyped(typechecker=beartype)
def init_cmtf(tensor: Float[Array, 'n m N'], rank: int, key: Array):
    n, m, N = tensor.shape
    keys = jax.random.split(key, num=4)

    W = jax.random.normal(keys[0], shape=(n, rank))
    V = jax.random.normal(keys[1], shape=(m, rank))
    H = jax.random.normal(keys[2], shape=(N, rank))
    R = jax.random.normal(keys[3], shape=(N, rank))

    return W, V, H, R

@jax.jit(static_argnames=('lam',))
@jaxtyped(typechecker=beartype)
def cmtf_lstsq(X1, X2, Y1, Y2, lam: float):
    X = jnp.concatenate([X1, lam*X2], axis=0)
    Y = jnp.concatenate([Y1, lam*Y2], axis=0)
    return jnp.linalg.lstsq(X, Y)[0].T

@jaxtyped(typechecker=beartype)
def cmtf_bsd(
    J: Float[Array, 'n m N'],
    Y: Float[Array, 'N n'],
    X: Float[Array, 'N m'],
    rank: int,
    lam: float = 0.1,
    degree: int = 3,
    dof: int = 12,
    iterations: int = 20,
    random_state: Array = get_random_key(),
    verbose: int = 0,
):
    log = make_log(verbose, '|CMTF-BSD| -> ')
    best_errors = []

    W, V, H, R = init_cmtf(J, rank, random_state)

    lstsq = jax.jit(jnp.linalg.lstsq)

    for iteration in tqdm(range(iterations), desc='Computing CMTF-BSD'):
        W = cmtf_lstsq(X1=khatri_rao(H, V), X2=R, Y1=unfold_kolda(J, 0).T, Y2=Y, lam=lam)
        V = lstsq(khatri_rao(H, W), unfold_kolda(J, 1).T)[0].T
        W, V = normalize_columns_V(W, V)

        H = lstsq(khatri_rao(V, W), unfold_kolda(J, 2).T)[0].T
        R = lstsq(W, Y.T)[0].T

        H, R = bsplines_projection(H, R, X @ V, dof, degree, lam)

        error = relative_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best = (W, V, H, R)

        log(f'Iteration [{iteration+1} / {iterations}]: error = {error:.4f}, best = {best_error:.4f}')
        best_errors.append(best_error)

    log(f'Returning best result with error = {best_error:.4f}')

    W, V, H, R = best
    g = make_g(fit_internals(X @ V, H, R, use='R'))
    return inference(W, V, g), best, jnp.array(best_errors)

@jax.jit
def project(j, c, B, dB, H, R):
    H = H.at[:, j].set(dB @ c)
    R = R.at[:, j].set(B @ c)
    return H, R

def bsplines_projection(
    H: Float[Array, 'N r'],
    R: Float[Array, 'N r'],
    U: Float[Array, 'N r'],
    dof: int, 
    degree: int,
    lam: float,
):
    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        knots = determine_knots(u, dof, degree)

        B = design_matrix(u, knots, degree)
        dB = design_dmatrix(u, knots, degree)

        c = cmtf_lstsq(X1=dB, Y1=h, X2=B, Y2=r, lam=lam)
        H, R = project(rank, c, B, dB, H, R)

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

def determine_knots(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    # move quantile to actual points
    def closest(knot):
        closest_point, min_dist = None, float("inf")
        for x in u:
            if (dist := abs(x - knot)) < min_dist:
                closest_point, min_dist = x, dist
        return closest_point

    internals = dof - degree - 1 + 2

    qs = jnp.linspace(0, 1, internals)
    knots = [jnp.quantile(u, q) for q in qs]
    knots = list(map(closest, knots))

    begin = [knots[0] for _ in range(degree)]
    end = [knots[-1] for _ in range(degree)]

    return jnp.array(begin + knots + end)
