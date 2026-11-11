"""The restructured model kernels (threads over row slices, work buffers, cached leave-out profiles) return
the bits of the plain expressions they replace."""
import numpy as np
import pytest
from scipy.special import expit

import epirefine.model as M
from epirefine.grid import Grid


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
