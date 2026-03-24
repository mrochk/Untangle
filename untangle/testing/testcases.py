import jax.numpy as jnp
from dataclasses import dataclass
from beartype.typing import Callable

@dataclass
class TestCase:
    function: Callable
    n_inputs: int
    n_outputs: int
    family: str = ''
    rank: int = None

    def unpack(self): return (
        self.function,
        self.n_inputs,
        self.n_outputs,
        self.family,
        self.rank,
    )

def polynomial1(u):
    u1, u2, u3 = u
    return jnp.array([
        -40000 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
        0.2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
        -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
    ])
 
POLYNOMIAL_1 = TestCase(polynomial1, 3, 3, 'polynomial1', 4)
 
def polynomial2(u):
    u1, u2, u3 = u
    return jnp.array([
        -4 * u1**2 + 8 * u1 * u3 + 6 * u1 - 3 * u3**2 - 8 * u3 - 6,
        2 * u1**2 - 4 * u1 * u3 - 3 * u1 + u2**3 + 6 * u2**2 * u3 + 12 * u2 * u3**2 - u2 + 8 * u3**3 + 2 * u3**2 + u3 + 3,
        -2 * u1**2 + 4 * u1 * u3 + 4 * u1 - 2 * u3**2 - 3 * u3 - u2 - 8,
        6 * u1 * u2**2 - 2 * u3**2 - 10,
        u3**3 - 3 * u2,
    ])
 
POLYNOMIAL_2 = TestCase(polynomial2, 3, 5, 'polynomial2', 5)
 
def polynomial3(u):
    """Underdetermined: 2 inputs → 4 outputs, degree-3 cross terms."""
    u1, u2 = u
    return jnp.array([
        u1**3 - 3 * u1 * u2**2 + u2,          # real part of (u1+iu2)^3 + u2
        3 * u1**2 * u2 - u2**3 - u1,           # imag part
        u1**2 + u2**2 - 1,                      # unit-circle residual
        u1**2 * u2 - u1 * u2**2 + u1 - u2,     # asymmetric bilinear
    ])
 
POLYNOMIAL_3 = TestCase(polynomial3, 2, 4, 'polynomial3')
 
def polynomial4(u):
    """Square 4×4 system with mixed degree-1/2/3 terms."""
    u1, u2, u3, u4 = u
    return jnp.array([
        u1**2 * u2 - u3 * u4 + u1 - 2,
        u2**3 - 3 * u1 * u3 + u4**2 - 1,
        u1 * u2 * u3 - u4**3 + u2 - u3 + 3,
        u3**2 * u4 - u1**2 + 2 * u2 * u4 - u1 + 4,
    ])
 
POLYNOMIAL_4 = TestCase(polynomial4, 4, 4, 'polynomial4')
 
def polynomial5(u):
    """High-degree (degree 5), low-rank structure: outputs depend only on
    the two linear combinations v1 = u1+u2 and v2 = u3-u4."""
    u1, u2, u3, u4 = u
    v1 = u1 + u2
    v2 = u3 - u4
    return jnp.array([
        v1**5 - 3 * v1**3 * v2 + v2**2,
        v1**4 * v2 - v2**5 + v1,
        v1**3 - 2 * v2**3 + v1 * v2,
    ])
 
POLYNOMIAL_5 = TestCase(polynomial5, 4, 3, 'polynomial5')
 
def periodic1(u):
    return jnp.array([
        jnp.cos(4 * jnp.pi * u),
        -jnp.sin(3 * jnp.pi * u),
    ])
 
PERIODIC_1 = TestCase(periodic1, 1, 2, 'periodic1', 3)
 
def periodic2(u):
    u1, u2, u3, u4 = u
    return jnp.array([
        jnp.cos(4 * jnp.pi * u1 * u2) + (u3 * u4)**2,
        jnp.sin(4 * jnp.pi * u3 * u4) + (u1 * u2)**2,
    ])
 
PERIODIC_2 = TestCase(periodic2, 4, 2, 'periodic2')
 
def periodic3(u):
    """Multi-frequency: mixes sin/cos at different harmonics with polynomials."""
    u1, u2, u3 = u
    return jnp.array([
        jnp.sin(2 * jnp.pi * u1) + jnp.cos(4 * jnp.pi * u2) + u3,
        jnp.sin(6 * jnp.pi * u2) * jnp.cos(2 * jnp.pi * u3) - u1**2,
        jnp.cos(2 * jnp.pi * u1 * u3) + jnp.sin(4 * jnp.pi * u2) - 0.5,
        jnp.sin(2 * jnp.pi * (u1 + u2 + u3)),
    ])
 
PERIODIC_3 = TestCase(periodic3, 3, 4, 'periodic3')
 
def periodic4(u):
    """Low-rank: all outputs depend on a single phase φ = 2π(u1+u2)."""
    u1, u2, u3, u4 = u
    phi = 2 * jnp.pi * (u1 + u2)
    return jnp.array([
        jnp.sin(phi) + u3 * u4,
        jnp.cos(phi) - u3**2,
        jnp.sin(2 * phi) + u4,
        jnp.cos(2 * phi) - u3 + u4**2,
    ])
 
PERIODIC_4 = TestCase(periodic4, 4, 4, 'periodic4')
 
def periodic5(u):
    """Coupled oscillators: 6 inputs, 3 outputs, rank-3."""
    u1, u2, u3, u4, u5, u6 = u
    return jnp.array([
        jnp.sin(2 * jnp.pi * u1) * jnp.cos(2 * jnp.pi * u2) + u5 * u6,
        jnp.sin(2 * jnp.pi * u3) * jnp.cos(2 * jnp.pi * u4) + u1 * u2,
        jnp.cos(2 * jnp.pi * (u1 + u3 + u5)) + jnp.sin(2 * jnp.pi * (u2 + u4 + u6)),
    ])
 
PERIODIC_5 = TestCase(periodic5, 6, 3, 'periodic5', 3)
 
def nonlinear1(u):
    """Exponential / Gaussian-like residuals, 2→2."""
    u1, u2 = u
    return jnp.array([
        jnp.exp(u1 + u2) - u1 * u2 - 2,
        jnp.exp(-u1**2 - u2**2) - 0.5,
    ])
 
NONLINEAR_1 = TestCase(nonlinear1, 2, 2, 'nonlinear1')
 
def nonlinear2(u):
    """Logarithmic / rational mix, 3→3. Assumes u > 0."""
    u1, u2, u3 = u
    return jnp.array([
        jnp.log(u1 + u2 + 1) - u3**2 + 1,
        u1 / (1 + u2**2) + jnp.log(u3 + 1) - 0.5,
        jnp.log(u1 * u2 + 1) + u3 / (1 + u1**2) - 1,
    ])
 
NONLINEAR_2 = TestCase(nonlinear2, 3, 3, 'nonlinear2')
 
def nonlinear3(u):
    """Sigmoid / tanh activations — resembles a shallow neural network, 4→3."""
    u1, u2, u3, u4 = u
    def sigma(x): return jnp.tanh(x)
    return jnp.array([
        sigma(u1 + 2 * u2) + sigma(u3 - u4) - 0.5,
        sigma(u1 * u3) - sigma(u2 + u4) + u1 - u2,
        sigma(u1 - u2 + u3 - u4) + sigma(u1 + u2) * sigma(u3 + u4),
    ])
 
NONLINEAR_3 = TestCase(nonlinear3, 4, 3, 'nonlinear3')
 
def nonlinear4(u):
    """Softmax-style competition + polynomial tail, 3→4."""
    u1, u2, u3 = u
    s = jnp.exp(u1) + jnp.exp(u2) + jnp.exp(u3)
    return jnp.array([
        100*jnp.exp(u1) / s - u2 * u3,
        0.01*jnp.exp(u2) / s + u1**2 - 0.5,
        jnp.exp(u3) / s - u1 * u2 + u3,
        u1**3 + u2**3 + u3**3 - 1,
    ])
 
NONLINEAR_4 = TestCase(nonlinear4, 3, 4, 'nonlinear4')
 
def nonlinear5(u):
    """Low-rank: 6 inputs, 2 outputs depending on u1+u2+u3 and u4*u5*u6."""
    u1, u2, u3, u4, u5, u6 = u
    s = u1 + u2 + u3
    p = u4 * u5 * u6
    return jnp.array([
        jnp.exp(-s**2) + p**2 - 0.5,
        jnp.tanh(s) * jnp.exp(-p**2) + s * p - 1,
    ])
 
NONLINEAR_5 = TestCase(nonlinear5, 6, 2, 'nonlinear5')
 
def nonlinear6(u):
    """Power-law + trig mix, 3→3."""
    u1, u2, u3 = u
    return jnp.array([
        jnp.abs(u1)**1.5 * jnp.sign(u1) - jnp.sin(u2 * u3),
        jnp.abs(u2)**2.5 * jnp.sign(u2) + jnp.cos(u1 + u3) - 1,
        jnp.abs(u3)**0.5 * jnp.sign(u3) - jnp.exp(-u1**2) + u2,
    ])
 
NONLINEAR_6 = TestCase(nonlinear6, 3, 3, 'nonlinear6')
 
ALL_TESTS = [
    # Polynomials
    POLYNOMIAL_1,
    POLYNOMIAL_2,
    POLYNOMIAL_3,
    POLYNOMIAL_4,
    POLYNOMIAL_5,
    # Periodic
    PERIODIC_1,
    PERIODIC_2,
    PERIODIC_3,
    PERIODIC_4,
    PERIODIC_5,
    # Nonlinear
    NONLINEAR_1,
    NONLINEAR_2,
    NONLINEAR_3,
    NONLINEAR_4,
    NONLINEAR_5,
    NONLINEAR_6,
]

EASY = [
    POLYNOMIAL_1,
    POLYNOMIAL_2,
    PERIODIC_1,
]