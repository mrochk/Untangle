import jax.numpy as jnp

def polynomial1(u):
    u1, u2, u3 = u
    return jnp.array([
        -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
        2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
        -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
    ])

POLYNOMIAL_1 = (3, 3, 4, polynomial1)

def polynomial2(u):
    u1, u2, u3 = u
    return jnp.array([
        -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
        2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
        -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
        6 * u1 * u2**2 - 2 * u3**2 - 10,
        u3**3 - 3*u2,
    ])

POLYNOMIAL_2 = (3, 5, 5, polynomial2)
