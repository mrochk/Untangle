from tqdm import tqdm
from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Callable

from untangle import _ops as ops
from untangle.utils import cpd_error
from untangle._common import initialize

@jaxtyped(typechecker=beartype)
def cmtf(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    niters: int,
    gamma: float,
    projection: Callable,
    projection_params: dict,
    key: Array,
    prefix: str,
):

    best_errors = []

    W, V, H, R = initialize(J, rank, key, with_R=True)

    J0 = ops.unfold_kolda(J, 0).T
    J1 = ops.unfold_kolda(J, 1).T
    J2 = ops.unfold_kolda(J, 2).T

    bar = tqdm(range(niters), desc=f'{prefix} (rank={rank})')

    out_proj = None

    for iteration in bar:
        W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, Y, gamma)
        V = ops.lstsq(ops.khatri_rao(H, W), J1)
        W, V = ops.normalize_columns_V(W, V)

        H = ops.lstsq(ops.khatri_rao(V, W), J2)
        R = ops.lstsq(W, Y.T)

        H, R, out_proj = projection(H, R, X @ V, out_proj, **projection_params)

        error = cpd_error(J, (W, V, H))

        if iteration == 0 or error < best_error:
            best = (W, V, H, R)
            best_iter = iteration
            best_error = error
            best_out_proj = out_proj

        bar.set_postfix_str(f'error={error:.4f}, best={best_error:.4f} ({best_iter})')
        best_errors.append(best_error)

    return best, best_out_proj
