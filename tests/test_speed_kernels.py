"""The restructured model kernels (threads over row slices, work buffers, cached leave-out profiles) return
the bits of the plain expressions they replace."""
import numpy as np
import pytest
from scipy.special import expit

import epirefine.model as M
from epirefine.grid import Grid


@pytest.fixture(autouse=True)
def numpy_code(monkeypatch):
    """These tests are about the numpy code: the compiled kernels (tests/test_rs_model.py) stay off."""
    monkeypatch.setenv("EPIREFINE_RS", "0")


def _tables(rng, U=37, Mg=11, G=64, dtype=np.float32):
    L = -rng.gamma(2.0, 40.0, (U, Mg, G))
    L[rng.random((U, Mg)) < 0.2] = 0.0  # groups without sites
    return L.astype(dtype)


def _marginalize_plain(L, T, batch=2048):
    out = np.empty_like(L)
    parts = [(slice(None), T)] if isinstance(T, np.ndarray) else T
    tiny = np.finfo(L.dtype).tiny
    for ms, Tm in parts:
        Tt = Tm.T.astype(L.dtype)
        for s in range(0, L.shape[0], batch):
            x = L[s:s + batch, ms]
            mx = x.max(-1, keepdims=True)
            out[s:s + batch, ms] = np.log(np.maximum(np.exp(x - mx) @ Tt, tiny)) + mx
    return out


@pytest.mark.parametrize("threads", [1, 3, 16])
@pytest.mark.parametrize("tau", [3.0, 100.0, float("inf"), "per-group"])
def test_marginalize_unit_bits(monkeypatch, threads, tau):
    monkeypatch.setattr(M, "_n_threads", lambda: threads)
    rng = np.random.default_rng(1)
    grid = Grid(64)
    L = _tables(rng, U=45)
    if tau == "per-group":
        tau = rng.choice([10.0, 300.0, float("inf")], L.shape[1])
    T = M.transition_spec(grid, tau)
    for batch in (2048, 7):
        assert np.array_equal(M.marginalize_unit(L, T, batch=batch), _marginalize_plain(L, T, batch=batch))


def test_logsumexp_work_buffer_bits():
    rng = np.random.default_rng(2)
    a = -rng.gamma(2.0, 40.0, (9, 5, 64))
    a[0, 0] = -np.inf; a[1, 1, :3] = a[1, 1].max()
    ref = M.logsumexp(a.copy(), -1)
    keep = a.copy()
    got = M.logsumexp(a, -1, work=np.empty_like(a))
    assert np.array_equal(ref, got, equal_nan=True) and np.array_equal(a, keep)


def test_bin_loglik_threads_bits(monkeypatch):
    rng = np.random.default_rng(3)
    Lp = _tables(rng, U=60)
    cb = rng.integers(0, 7, 60); cb[cb == 4] = 5  # an empty bin
    w = rng.random(60)
    monkeypatch.setattr(M, "_n_threads", lambda: 1)
    ref = M.bin_loglik(Lp, cb, 7, w)
    for h in range(7):
        idx = np.flatnonzero(cb == h)
        plain = np.tensordot(w[idx], Lp[idx].astype(np.float64), 1) if len(idx) else 0.0
        assert np.array_equal(ref[h], np.zeros(Lp.shape[1:]) + plain)
    monkeypatch.setattr(M, "_n_threads", lambda: 8)
    assert np.array_equal(M.bin_loglik(Lp, cb, 7, w), ref)
    monkeypatch.setattr(M, "BIN_BYTES", 1)  # no room for more than one bin at a time
    assert np.array_equal(M.bin_loglik(Lp, cb, 7, w), ref)


def test_outlier_responsibility_bits():
    rng = np.random.default_rng(4)
    grid = Grid(64)
    L = _tables(rng, U=80, Mg=6)
    cb = rng.integers(0, 4, 80)
    m = M.FittedModel(L, cb, 4, ["a"] * 6, grid, 100.0, robust_iters=1)
    eps = 0.05
    Lp = m.Lp_chunks.astype(np.float64)
    s_out = M.logsumexp(Lp + m.log_pi[None], -1).sum(-1)
    ell_loo = m.ell[cb] - m.w[:, None, None] * Lp
    s_in = M.logsumexp(Lp + M.host_profiles(ell_loo, m.log_pi), -1).sum(-1)
    ref = expit(np.log(eps) - np.log1p(-eps) + s_out - s_in)
    m._s_out = None
    for batch in (None, 7):
        assert np.array_equal(m._outlier_responsibility(eps, batch=batch), ref)


def _npmle_plain(ell, iters=300, tol=1e-6, init=None):
    B, Mg, G = ell.shape
    lik = np.exp(ell - ell.max(-1, keepdims=True)).transpose(1, 0, 2).copy()
    pi = np.full((Mg, G), 1.0 / G) if init is None else init.copy()
    for _ in range(iters):
        denom = np.maximum(np.matmul(lik, pi[:, :, None])[:, :, 0], 1e-300)
        new = np.maximum(pi * np.matmul((1.0 / denom)[:, None, :], lik)[:, 0, :] / B, 1e-300)
        delta = np.abs(new - pi).max()
        pi = new
        if delta < tol:
            break
    return pi


@pytest.mark.parametrize("tol", [1e-6, 1e-3, 1e-2])
def test_npmle_blocks_bits(monkeypatch, tol):
    """Blocks of groups on threads, with the stop taken over all groups: the bits of the single loop, whether
    the fit stops inside a round, at its end, or at the iteration cap."""
    rng = np.random.default_rng(5)
    ell = -rng.gamma(2.0, 3.0, (100, 37, 64)) * rng.integers(1, 40, (100, 37, 1))
    monkeypatch.setattr(M, "_n_threads", lambda: 4)
    for iters in (300, 7):
        for kw in (dict(blocks=37), dict(blocks=5, round_iters=3), dict(blocks=4), dict()):
            ref = _npmle_plain(ell, iters, tol)
            assert np.array_equal(M.npmle(ell, iters, tol, **kw), ref)
            assert np.array_equal(M.npmle(ell * 1.1, iters, tol, init=ref, **kw), _npmle_plain(ell * 1.1, iters, tol, init=ref))


@pytest.mark.parametrize("U,Mg", [(700, 11), (40, 300), (5, 1), (500, 1)])
def test_stack_product_bits(U, Mg):
    """The three ways _stack_product takes (all units in the comparison; one product over all rows; per-unit
    products on threads) return numpy's stacked product."""
    rng = np.random.default_rng(6)
    e = np.exp(-rng.gamma(2.0, 5.0, (U, Mg, 64))).astype(np.float32)
    Tt = Grid(64).transition(30.0).T.astype(np.float32)
    assert np.array_equal(M._stack_product(e, Tt), e @ Tt)


def _expected_tables_plain(sites, n_units, n_groups, det, grid, n_cap=500, dtype=np.float64):
    """expected_tables as it was: one (group, n) combination at a time, tuple keys, one block."""
    import polars as pl
    from scipy.special import gammaln
    from epirefine.detection import bb_logpmf
    s = sites.with_columns(pl.col("n").clip(upper_bound=n_cap)).group_by("unit", "group", "n").agg(pl.len().alias("w"))
    table = {}
    for g, n in s.select("group", "n").unique().iter_rows():
        p = det[g]
        k = np.arange(n + 1, dtype=float)
        logc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        l1, l0 = bb_logpmf(k, n, p.mu1, p.kappa1), bb_logpmf(k, n, p.mu0, p.kappa0)
        p1, p0 = np.exp(logc + l1), np.exp(logc + l0)
        term = np.logaddexp(l1[:, None] + grid.logx, l0[:, None] + grid.log1mx)
        table[(g, n)] = (p1 / p1.sum()) @ term, (p0 / p0.sum()) @ term
    H1 = np.zeros((n_units, n_groups, grid.size), dtype); H0 = np.zeros_like(H1)
    keys = sorted(set(zip(s["group"].to_list(), s["n"].to_list())))
    kid = {k: i for i, k in enumerate(keys)}
    T1 = np.stack([table[k][0] for k in keys]); T0 = np.stack([table[k][1] for k in keys])
    s = s.with_columns((pl.col("unit").cast(pl.Int64) * n_groups + pl.col("group")).alias("flat")).sort("flat")
    flat, w = s["flat"].to_numpy(), s["w"].to_numpy().astype(float)
    r = np.array([kid[k] for k in zip(s["group"].to_list(), s["n"].to_list())])
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    for T, out in ((T1, H1), (T0, H0)):
        out.reshape(-1, grid.size)[flat[starts]] = np.add.reduceat(T[r] * w[:, None], starts, axis=0)
    return H1, H0


@pytest.mark.parametrize("threads", [1, 5])
def test_expected_tables_bits(monkeypatch, threads):
    import polars as pl
    from epirefine import assignability as A
    from epirefine.detection import DetectionParams
    monkeypatch.setattr(A, "_n_threads", lambda: threads)
    rng = np.random.default_rng(7)
    det = {g: DetectionParams(rng.uniform(0.001, 0.1), rng.uniform(5, 200), rng.uniform(0.6, 0.99), rng.uniform(5, 50))
           for g in range(5)}
    n = 30_000
    sites = pl.DataFrame({"unit": rng.integers(0, 60, n), "group": rng.integers(0, 5, n),
                          "n": np.minimum(rng.geometric(0.02, n), 700)})
    grid = Grid(64)
    ref = _expected_tables_plain(sites, 60, 5, det, grid, dtype=np.float32)
    for chunk_rows in (200_000, 300):
        got = A.expected_tables(sites, 60, 5, det, grid, dtype=np.float32, chunk_rows=chunk_rows)
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])


def test_asimov_threads_bits():
    from epirefine.assignability import asimov
    rng = np.random.default_rng(8)
    H1, H0 = (-rng.gamma(2.0, 3.0, (50, 400, 64)).astype(np.float32) for _ in range(2))
    phi = rng.random((50, 400))
    assert np.array_equal(asimov(H1, H0, phi), phi[..., None] * H1 + (1 - phi[..., None]) * H0)


def test_thread_count_follows_the_environment(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False); monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    full = M._n_threads()
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    assert M._n_threads() == 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "x"); monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert M._n_threads() == min(3, full)
