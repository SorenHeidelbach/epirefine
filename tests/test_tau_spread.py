import numpy as np

from pbinner.grid import Grid
from pbinner.model import learn_departures, transition_spec


def _table(g, k, n):
    return (k * np.log(g.x) + (n - k) * np.log1p(-g.x)).astype(np.float32)


def test_learned_departures_separate_clean_and_variable_groups():
    g = Grid(64)
    rng = np.random.default_rng(1)
    B, C, N = 4, 40, 400
    L = np.zeros((B * C, 2, g.size), np.float32)
    chunk_bin = np.repeat(np.arange(B), C)
    for b in range(B):
        for c in range(C):
            k0 = rng.binomial(N, 0.95 if b % 2 else 0.02)                  # group 0: on/off, never departs
            lvl = 0.25 if c < 4 else 0.5                                    # group 1: 10% of chunks depart
            L[b * C + c, 0] = _table(g, k0, N)
            L[b * C + c, 1] = _table(g, rng.binomial(N, lvl), N)
    eps, tw, tab = learn_departures(L, chunk_bin, B, ["m", "m"], g, log=lambda *a: None)
    assert eps[0] < eps[1]
    assert 0.02 <= eps[1] <= 0.2


def test_outlier_kernel_bounds_the_cost_of_one_shifted_group():
    g = Grid(64)
    T = transition_spec(g, float("inf"), outlier=(np.array([0.01]), 10.0))[0][1]
    assert np.allclose(T.sum(1), 1.0)
    i = np.argmin(np.abs(g.x - 0.5)); j = np.argmin(np.abs(g.x - 0.3))
    assert 1e-4 < T[i, j] < 0.01 and T[i, i] > 0.98
