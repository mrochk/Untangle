import jax, jax.numpy as jnp

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype

#@jaxtyped(typechecker=beartype)
#def decoupling_basic_constrained(
    #J: Float[Array, 'n m N'], 
    #Y: Float[Array, 'N n'], 
    #X: Float[Array, 'N m'], 
    #rank: int,
    #degree: int = 3,
    #dof: int = 12,
    ##n_init: int = mp.cpu_count(),
    #verbose: int = 0,
#):