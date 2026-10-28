import numpy as np
import polars as pl

from pbinner.calibrate import outlier_rates
from pbinner.grid import Grid
from pbinner.model import transition_spec


def test_outlier_rates_count_rare_departures_not_contaminants():
    rng = np.random.default_rng(0)
    rows = []
    for b in range(20):
        for c in range(10):
            cid = b * 20 + c
            N = 2000
            p0 = 0.9 if b % 2 else 0.02
            rows.append((b, cid, 0, N, int(rng.binomial(N, p0))))                      # group 0: never departs
            level = 0.6 if (b, c) in {(1, 1), (3, 2), (5, 3), (7, 4)} else 0.4         # group 1: 4 of 200 contigs depart by 0.2
            rows.append((b, cid, 1, N, int(rng.binomial(N, level))))
        rows.append((b, b * 20 + 19, 0, 2000, 1800 if b % 2 == 0 else 20))             # contaminant: a jump, not a departure
    st = pl.DataFrame(rows, schema={"bin": pl.Int32, "contig": pl.Int32, "group": pl.Int32, "N": pl.Int64, "K": pl.Int64}, orient="row")
    eps, eps0, tab = outlier_rates(st, 2)
    assert eps[0] < eps[1]
    assert 0.01 <= eps[1] <= 0.05


def test_outlier_kernel_bounds_the_cost_of_one_shifted_group():
    g = Grid(64)
    Ts = transition_spec(g, float("inf"), outlier=(np.array([0.01]), 10.0))
    T = Ts[0][1]
    assert np.allclose(T.sum(1), 1.0)
    i = np.argmin(np.abs(g.x - 0.5))
    j = np.argmin(np.abs(g.x - 0.3))
    # a unit at 0.3 against a host at 0.5: possible (the outlier part), but far less likely than staying
    assert 1e-4 < T[i, j] < 0.01 and T[i, i] > 0.98
