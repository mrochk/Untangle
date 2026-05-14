import jax, jax.numpy as jnp

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Tuple, Optional

from untangle.algorithm._cmtf import cmtf
from untangle.algorithm import Decoupling
from untangle import _ops as ops
from untangle import _common as c

@jaxtyped(typechecker=beartype)
def cmtf_bsd(
    X: Float[Array, 'N m'],
    Y: Float[Array, 'N n'],
    J: Float[Array, 'n m N'],
    rank: int,
    niters: int = 100,
    dof: Optional[int] = None,
    gamma: float = 0.1,
    degree: int = 3,
    key: Optional[Array] = None,
) -> Decoupling:

    N = X.shape[0]
    if dof is None: dof = c.default_dof(N)
    if key is None: key = c.get_random_key() 

    factors, _ = cmtf(
        X, Y, J, rank, niters, gamma, bspl_projection, 
        {'dof': dof, 'degree': degree, 'gamma': gamma},
        key, '|CMTF-BSD|',
    )

    _, V, H, R = factors
    internals = c.make_internals(c.fit_internals_with_smoothing_spline(X@V, H, R))
    return Decoupling(factors, internals)

def bspl_projection(
    H: Float[Array, 'N r'],
    R: Float[Array, 'N r'],
    Z: Float[Array, 'N r'],
    out,
    dof: int, 
    degree: int,
    gamma: float,
) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

    coefs_out, knots_out = [], []

    for rank in range(H.shape[1]):
        z, h, r = Z[:, rank], H[:, rank], R[:, rank]

        knots = c.determine_knots(z, dof, degree, 'quantile')

        B = c.get_design_matrix(z, knots, degree)
        dB = c.get_design_dmatrix(z, knots, degree)

        coefs = ops.cmtf_lstsq(dB, B, h, r, gamma)
        H, R = c.bspline_project(rank, coefs, B, dB, H, R)

        g_coefs = jnp.linalg.lstsq(B, R[:, rank])[0]
        coefs_out.append(g_coefs)
        knots_out.append(knots)

    out = (coefs_out, knots_out)

    return H, R, out
