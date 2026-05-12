import jax, jax.numpy as jnp
import optax

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
    determine_knots,
    get_design_matrix,
    get_design_dmatrix,
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

    prefix = '|CMTF-PSD|'
    log = make_log(verbose, prefix)

    best_errors = []

    W, V, H, R = initialize(J, rank, key, with_R=True)

    J0 = ops.unfold_kolda(J, 0).T
    J1 = ops.unfold_kolda(J, 1).T
    J2 = ops.unfold_kolda(J, 2).T

    log_lam_init = jnp.zeros((rank,))

    bar = tqdm(range(niters), desc=prefix)

    for iteration in bar:
        W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, Y, gamma)
        V = ops.lstsq(ops.khatri_rao(H, W), J1)
        W, V = ops.normalize_columns_V(W, V)

        H = ops.lstsq(ops.khatri_rao(V, W), J2)
        R = ops.lstsq(W, Y.T)

        Z = X @ V
        H, R, log_lam_init = psplines_projection(H, R, Z, dof, degree, gamma, log_lam_init)

        error = cpd_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best_error = error
            best_iter = iteration
            best = (W, V, H, R)

        bar.set_postfix_str(f'error={error:.4f}, best={best_error:.4f} ({best_iter})')
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
    log_lam_init,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    log_lam_out = []

    for rank in range(H.shape[1]):
        u, h, r = U[:, rank], H[:, rank], R[:, rank]

        knots = determine_knots(u, dof, degree, 'even')
        B  = get_design_matrix(u, knots, degree)
        dB = get_design_dmatrix(u, knots, degree)

        n_basis = B.shape[1]
        D = second_diff_matrix(n_basis)

        A = jnp.vstack([dB, gamma * B])
        y = jnp.concatenate([h, gamma * r])

        log_lam = gcv_lbfgs(A, y, D, n=len(u), log_lam_init=log_lam_init[rank])

        lam = 10**log_lam

        log_lam_out.append(log_lam)

        A = jnp.vstack([A, lam * D])
        y = jnp.concatenate([y, jnp.zeros(D.shape[0])])
        coefs = jnp.linalg.lstsq(A, y)[0]

        H, R = bspline_project(rank, coefs, B, dB, H, R)

    return H, R, jnp.array(log_lam_out)

@jax.jit(static_argnames='n')
def second_diff_matrix(n: int) -> Array:
    D1 = jnp.diff(jnp.eye(n), axis=0)
    D2 = jnp.diff(D1, axis=0)
    return D2

@jax.jit
def gcv_score(log_lam: Array, A: Array, y: Array, D: Array, n: int) -> Array:
    lam = 10.0 ** log_lam

    A_aug = jnp.vstack([A, lam * D])
    y_aug = jnp.concatenate([y, jnp.zeros(D.shape[0])])

    coefs     = jnp.linalg.lstsq(A_aug, y_aug)[0]
    residuals = y_aug[:len(y)] - A @ coefs
    rss       = jnp.sum(residuals ** 2)

    Q, _ = jnp.linalg.qr(A_aug)
    df   = jnp.trace(Q[:len(y)] @ Q[:len(y)].T)

    gcv = (n * rss) / (n - df) ** 2
    return gcv

@jax.jit
def gcv(A: Array, y: Array, D: Array, n: int) -> Array:
    log_lams_coarse = jnp.linspace(-10, 10, 500)
    scores   = jax.vmap(lambda ll: gcv_score(ll, A, y, D, n))(log_lams_coarse)
    best = log_lams_coarse[jnp.argmin(scores)]

    log_lams_fine = jnp.linspace(best-1, best+1, 500)
    scores = jax.vmap(lambda ll: gcv_score(ll, A, y, D, n))(log_lams_fine)
    best = log_lams_fine[jnp.argmin(scores)]

    return 10**best

def gcv_lbfgs(
    A: Array,
    y: Array,
    D: Array,
    n: int,
    log_lam_init,
    lbfgs_iters: int = 10,
    n_restarts: int = 5,
) -> Array:

    log_lam_inits = jnp.linspace(-6, 4, n_restarts) if log_lam_init == 0 else [log_lam_init]

    @jax.jit
    def run_lbfgs(log_lam_init: Array) -> Tuple[Array, Array]:
        optimizer = optax.lbfgs()

        params = {'log_lam': log_lam_init}

        opt_state = optimizer.init(params)

        loss_and_grad = jax.value_and_grad(
            lambda p: gcv_score(p['log_lam'], A, y, D, n)
        )

        def step(carry, _):
            params, opt_state = carry
            loss, grads = loss_and_grad(params)
            updates, opt_state = optimizer.update(
                grads, opt_state, params,
                value=loss,
                grad=grads,
                value_fn=lambda p: gcv_score(p['log_lam'], A, y, D, n),
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params_final, _), losses = jax.lax.scan(
            step, (params, opt_state), None, length=lbfgs_iters
        )

        return params_final['log_lam'], losses[-1]

    best_log_lam = log_lam_inits[0]
    best_score   = jnp.inf

    for log_lam_init in log_lam_inits:
        log_lam, score = run_lbfgs(log_lam_init)
        if jnp.isnan(log_lam): log_lam = 0.0
        print(log_lam)
        if score < best_score:
            best_score   = score
            best_log_lam = log_lam

    return best_log_lam
