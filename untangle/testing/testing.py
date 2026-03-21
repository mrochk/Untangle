import jax, jax.numpy as jnp

from untangle.utils import get_random_key, collect_information, output_errors
from untangle.decomposition import search_rank
from untangle.scaler import StdScaler

def test_algorithm(algorithm, test_case, N: int = 20, **algorithm_kwargs):
    '''Function given in example (4) of the paper.'''

    f, m, n, family, rank = test_case.unpack()

    scaler = StdScaler(f, m)
    f_scaled = scaler.scale()

    X, Y, J = collect_information(f_scaled, N, m)

    if rank is None: 
        rank, _ = search_rank(J)
        print(f'found rank = {rank}')

    learned_scaled, _, _ = algorithm(J, Y, X, rank, **algorithm_kwargs)
    learned = scaler.unscale(learned_scaled)

    return output_errors(f, learned, X)
