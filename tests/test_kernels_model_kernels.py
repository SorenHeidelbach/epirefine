"""Model kernels against the numpy expressions of epirefine.model that they replace (written out here, so the
tests need no EpiRefine): float64 results equal to rounding (relative 1e-12), independent of the thread count."""
import numpy as np
import pytest
from scipy.special import logsumexp

rs = pytest.importorskip("epirefine._kernels", reason="no compiled kernels in this install")

pytestmark = pytest.mark.skipif(not rs.supports("model_kernels"), reason="build without model kernels")


def tables(rng, U=70, M=9, G=64):
    L = -rng.gamma(2.0, 40.0, (U, M, G))
    L[rng.random((U, M)) < 0.2] = 0.0
    return L.astype(np.float32)


def test_bin_loglik():
    rng = np.random.default_rng(1)
    lp, cb, w = tables(rng), rng.integers(0, 5, 70), rng.random(70)
    cb[cb == 1] = 0
    want = np.zeros((5, 9, 64))
    for h in range(5):
        idx = np.flatnonzero(cb == h)
        if len(idx):
            want[h] = np.tensordot(w[idx], lp[idx].astype(np.float64), 1)
    got = rs.bin_loglik(lp, cb, 5, w, threads=3)
    assert got.shape == want.shape and np.allclose(got, want, rtol=1e-12, atol=0)
    assert np.array_equal(got, rs.bin_loglik(lp, cb, 5, w, threads=1))
    with pytest.raises(ValueError):
        rs.bin_loglik(lp, cb, 2, w)
    with pytest.raises(ValueError):
        rs.bin_loglik(lp.astype(np.float64), cb, 5, w)


def test_community_and_leave_chunk_out_terms():
    rng = np.random.default_rng(2)
    lp, cb, w = tables(rng), rng.integers(0, 5, 70), rng.random(70)
    lp64 = lp.astype(np.float64)
    ell = np.stack([np.tensordot(w[cb == h], lp64[cb == h], 1) for h in range(5)])
    log_pi = np.log(rng.dirichlet(np.ones(64), 9))
    want = logsumexp(lp64 + log_pi[None], axis=-1).sum(-1)
    got = rs.community_terms(lp, log_pi, threads=4)
    assert np.allclose(got, want, rtol=1e-12, atol=0) and np.array_equal(got, rs.community_terms(lp, log_pi, threads=1))
    a = ell[cb] - w[:, None, None] * lp64 + log_pi[None]
    prof = a - logsumexp(a, axis=-1, keepdims=True)
    want = logsumexp(lp64 + prof, axis=-1).sum(-1)
    got = rs.leave_chunk_out_terms(lp, cb, w, ell, log_pi, threads=4)
    assert np.allclose(got, want, rtol=1e-12, atol=0)
    assert np.array_equal(got, rs.leave_chunk_out_terms(lp, cb, w, ell, log_pi, threads=1))


def test_logsumexp_edge_cases():
    lp = np.zeros((3, 2, 8), np.float32)
    lp[1, 0, :] = -np.inf
    lp[2, 1, 3:] = -50.0
    log_pi = np.log(np.full((2, 8), 1 / 8))
    with np.errstate(all="ignore"):
        want = logsumexp(lp.astype(np.float64) + log_pi[None], axis=-1).sum(-1)
    got = rs.community_terms(lp, log_pi)
    assert got[1] == -np.inf and np.allclose(got[[0, 2]], want[[0, 2]], rtol=1e-14, atol=0)


def npmle(lik, pi, iters, tol):
    B = lik.shape[1]
    for _ in range(iters):
        denom = np.maximum(np.matmul(lik, pi[:, :, None])[:, :, 0], 1e-300)
        new = np.maximum(pi * np.matmul((1.0 / denom)[:, None, :], lik)[:, 0, :] / B, 1e-300)
        delta = np.abs(new - pi).max()
        pi = new
        if delta < tol:
            break
    return pi


@pytest.mark.parametrize("tol,iters", [(1e-6, 300), (1e-3, 300), (1e-2, 40), (0.0, 17), (0.0, 16), (1.0, 5)])
def test_npmle(tol, iters):
    rng = np.random.default_rng(3)
    ell = -rng.gamma(2.0, 3.0, (41, 30, 64)) * rng.integers(1, 40, (41, 30, 1))  # [M, B, G]
    lik = np.exp(ell - ell.max(-1, keepdims=True))
    pi0 = np.full((41, 64), 1 / 64)
    want = npmle(lik, pi0, iters, tol)
    got = rs.npmle(lik, pi0, iters, tol, threads=4)
    assert got.shape == want.shape and np.allclose(got, want, rtol=1e-9, atol=1e-300)
    assert np.array_equal(got, rs.npmle(lik, pi0, iters, tol, threads=1))


@pytest.mark.parametrize("threads", [1, 4])
def test_cell_sums_bits(threads):
    rng = np.random.default_rng(4)
    table = -rng.gamma(2.0, 3.0, (500, 64))
    table[3] = 0.0
    r, w = rng.integers(0, 500, 20_000), rng.integers(1, 9, 20_000).astype(float)
    starts = np.unique(np.r_[0, rng.integers(1, 12_000, 3_000), 12_000, 12_009, 12_150, 12_500])  # short and long cells
    want = np.add.reduceat(table[r] * w[:, None], starts, axis=0)
    got = rs.cell_sums(table, r, w, starts, threads=threads)
    assert got.dtype == np.float64 and np.array_equal(got, want)
    with pytest.raises(ValueError):
        rs.cell_sums(table, r, w, np.array([0, 5, 5]))
    with pytest.raises(ValueError):
        rs.cell_sums(table, np.array([500]), np.array([1.0]), np.array([0]))


@pytest.mark.parametrize("U", [1, 33, 70])
def test_score_sums(U):
    rng = np.random.default_rng(5)
    M_, G, H = 9, 64, 7
    lp = tables(rng, U=U, M=M_)
    prof = np.log(rng.dirichlet(np.ones(G) * 0.2, (H, M_)))
    am = lp.max(-1); bm = prof.max(-1)
    e = np.exp(lp - am[..., None]).astype(np.float64); eb = np.exp(prof - bm[..., None])
    A = np.log(np.maximum(np.einsum("umg,hmg->uhm", e, eb), 1e-300)) + am[:, None, :] + bm[None]
    au = rng.normal(-50, 20, (U, M_)); le1, le0 = np.log1p(-0.05), np.log(0.05) + au
    S, R = rs.score_sums(e, am, eb.transpose(1, 2, 0), bm.T, le1, le0, threads=3)
    assert np.allclose(S, A.sum(-1), rtol=1e-12, atol=0)
    assert np.allclose(R, np.logaddexp(A + le1, le0[:, None, :]).sum(-1), rtol=1e-12, atol=0)
    S1, none = rs.score_sums(e, am, eb.transpose(1, 2, 0), bm.T, threads=1)
    assert none is None and np.array_equal(S1, S)
