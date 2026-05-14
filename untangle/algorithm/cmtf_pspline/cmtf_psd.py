import jax, jax.numpy as jnp
import optax

from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Tuple, Optional
from beartype import beartype

from untangle.algorithm._cmtf import cmtf
from untangle.algorithm import Decoupling
from untangle import _ops as ops 
from untangle._common import *

@jaxtyped(typechecker=beartype)
def cmtf_psd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    niters: int = 100,
    gamma: float = 0.1,
    dof: Optional[int] = None,
    degree: int = 3,
    key: Optional[Array] = None,
) -> Decoupling:

    N = X.shape[0]
    if dof is None: dof = default_dof(N)
    if key is None: key = get_random_key()

    projection_params = {'dof': dof, 'degree': degree, 'gamma': gamma}

    factors, out_proj = cmtf(
        X, Y, J, rank, niters, gamma,
        pspl_projection, projection_params,
        key, '|CMTF-PSD|',
    )

    _, coefs, knots = out_proj
    internals = make_internals(fit_internals_with_best_coefs(coefs, knots, degree))
    return Decoupling(factors, internals)

def pspl_projection(
    H: Float[Array, 'N r'],
    R: Float[Array, 'N r'],
    Z: Float[Array, 'N r'],
    out, dof: int, degree: int, gamma: float,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    rank = H.shape[1]

    lam_in = jnp.zeros(rank) if out is None else out[0]
    lam_out = []

    coefs_out = []
    knots_out = []

    for rank in range(H.shape[1]):
        z, h, r = Z[:, rank], H[:, rank], R[:, rank]

        is_degenerate = (jnp.max(z) - jnp.min(z)) < 1e-6

        if is_degenerate:
            print(f'{rank} is degenerate')
            H = H.at[:, rank].set(jnp.zeros_like(H[:, rank]))
            R = R.at[:, rank].set(jnp.zeros_like(R[:, rank]))
            lam_out.append(lam_in[rank])
            coefs_out.append(None)
            knots_out.append(None)
            continue

        knots = determine_knots(z, dof, degree, 'even')
        B  = get_design_matrix(z, knots, degree)
        dB = get_design_dmatrix(z, knots, degree)

        n_basis = dB.shape[1]
        D = ops.second_diff_matrix(n_basis)

        A = jnp.vstack([dB, gamma * B])
        y = jnp.concatenate([h, gamma * r])

        log_lam = gcv_lbfgs(A, y, D, n=len(z), log_lam_init=lam_in[rank])

        lam = 10**log_lam

        lam_out.append(log_lam)

        A = jnp.vstack([A, lam * D])
        y = jnp.concatenate([y, jnp.zeros(D.shape[0])])
        coefs = jnp.linalg.lstsq(A, y)[0]

        H, R = bspline_project(rank, coefs, B, dB, H, R)

        g_coefs = jnp.linalg.lstsq(B, R[:, rank])[0]
        coefs_out.append(g_coefs)
        knots_out.append(knots)

    out = (jnp.array(lam_out), coefs_out, knots_out)
    return H, R, out

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

    log_lam_inits = jnp.linspace(-6, 4, n_restarts) if float(log_lam_init) == 0.0 else [log_lam_init]

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
        if score < best_score:
            best_score = score
            best_log_lam = log_lam

    return best_log_lam
