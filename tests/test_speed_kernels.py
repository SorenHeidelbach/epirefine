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
    """expected_tables written plainly: one (group, n) combination at a time, tuple keys, one reduction (the rows
    of a cell in ascending n)."""
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
    s = s.with_columns((pl.col("unit").cast(pl.Int64) * n_groups + pl.col("group")).alias("flat")).sort("flat", "n")
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
    # in batches of units: the same tables (units 60..63 have no sites)
    parts = list(A.expected_tables(sites, 64, 5, det, grid, dtype=np.float32, unit_batch=7))
    assert [p[0] for p in parts] == list(range(0, 64, 7))
    H1, H0 = (np.concatenate([p[i] for p in parts]) for i in (1, 2))
    assert np.array_equal(H1[:60], ref[0]) and np.array_equal(H0[:60], ref[1]) and not H1[60:].any()


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


def _unit_loglik_plain(unit, group, n, k, n_units, n_groups, det, grid):
    """The definition: per (unit, group) cell the reduceat sum of count x term over its (n, k) in ascending order."""
    import polars as pl
    agg = (pl.DataFrame({"unit": unit, "group": group, "n": n, "k": k}).group_by("unit", "group", "n", "k").len()
           .sort("unit", "group", "n", "k"))
    table = M.site_terms(agg.select("group", "n", "k"), det, grid)
    flat = agg["unit"].to_numpy().astype(np.int64) * n_groups + agg["group"].to_numpy()
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    out = np.zeros((n_units * n_groups, grid.size), np.float32)
    out[flat[starts]] = np.add.reduceat(table * agg["len"].to_numpy().astype(float)[:, None], starts, axis=0)
    return out.reshape(n_units, n_groups, grid.size)


def test_unit_loglik_is_defined_by_its_cells():
    """The same table whatever the order of the site rows, the size of the unit batches and of the reduction
    blocks, and equal to the plain definition (so it is the same in every run)."""
    from epirefine.detection import DetectionParams
    rng = np.random.default_rng(9)
    det = {g: DetectionParams(rng.uniform(0.001, 0.1), rng.uniform(5, 200), rng.uniform(0.6, 0.99), rng.uniform(5, 50))
           for g in range(4)}
    N = 60_000
    unit, group = np.sort(rng.integers(0, 50, N)), rng.integers(0, 4, N).astype(np.int32)
    n = np.minimum(rng.geometric(0.05, N), 300).astype(np.int32)
    k = rng.binomial(n, rng.random(N)).astype(np.int32)
    grid = Grid(64)
    ref = _unit_loglik_plain(unit, group, n, k, 50, 4, det, grid)
    for kw in (dict(), dict(batch_rows=5_000), dict(batch_rows=700, chunk_rows=50)):
        assert np.array_equal(M.unit_loglik_arrays(unit, group, n, k, 50, 4, det, grid, dtype=np.float32, **kw), ref)
    p = rng.permutation(N)  # rows not unit by unit: the batches are taken through an index
    got = M.unit_loglik_arrays(unit[p], group[p], n[p], k[p], 50, 4, det, grid, dtype=np.float32, batch_rows=5_000)
    assert np.array_equal(got, ref)
    import polars as pl
    df = pl.DataFrame({"unit": unit[p], "group": group[p], "n": n[p], "k": k[p]})
    assert np.array_equal(M.unit_loglik(df, 50, 4, det, grid, dtype=np.float32), ref)


def test_batched_summary_threads_bits(monkeypatch):
    import epirefine.pipeline as P
    rng = np.random.default_rng(10)
    grid = Grid(64)
    L = _tables(rng, U=300, Mg=40)
    whole = grid.summarize(L.astype(np.float64))
    for threads in (1, 7):
        monkeypatch.setattr(P, "_n_threads", lambda: threads)
        got = P._batched_summary(grid, lambda sl: L[sl].astype(np.float64), len(L), batch=50)
        assert all(np.array_equal(a, b) for a, b in zip(got, whole))


def test_lengths_and_sample_from_the_cached_assembly(community, base_run, tmp_path):
    """With a site cache written from the same assembly file, contig lengths and the grouping sample come from
    its encoding: the same lengths in the same order, the same motif records, and no cache for another file."""
    import shutil
    import epirefine.pipeline as P
    from epirefine import io as pio
    from epirefine.motifs import build_records
    args = community.args("unused")
    asm = P._cached_assembly(base_run.out / "cache", args["assembly"])
    assert asm is not None
    lengths = pio.fasta_lengths(args["assembly"])
    assert list(zip(asm.names, asm.lengths.tolist())) == list(lengths.items())
    names = list(lengths)[::2]
    motifs = pio.read_motifs(args["motifs_path"])
    assert build_records(P._decoded(asm, names), motifs).equals(build_records(pio.fasta_subset(args["assembly"], names), motifs))
    other = shutil.copy(args["assembly"], tmp_path / "copy.fasta")          # another path: not this cache's file
    assert P._cached_assembly(base_run.out / "cache", other) is None
    assert P._cached_assembly(tmp_path / "no_cache", args["assembly"]) is None


@pytest.mark.parametrize("threads", ["1", "4"])
def test_own_bin_scores_and_margin_vs_unknown(monkeypatch, threads):
    """own_scores = score_units at the unit's own bin, margin_vs_unknown = the vs-unknown margin of
    margin_vs_competitors, to rounding (the per-unit dot product sums in another order than the matrix product)."""
    import epirefine.assignability as A
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    rng = np.random.default_rng(5)
    grid = Grid(64)
    U, Mg, H = 300, 13, 9
    model = object.__new__(M.FittedModel)
    lp = np.log(rng.dirichlet(np.ones(64) * 0.3, (H, Mg)))
    model.grid, model.n_bins, model.log_prof = grid, H, lp
    model.log_pi = np.log(np.exp(lp).mean(0))
    model.T = M.transition_spec(grid, 100.0)
    Lp = _tables(rng, U, Mg)
    t = rng.integers(0, H, U)
    full = A.score_units(Lp, lp)
    np.testing.assert_allclose(A.own_scores(Lp, lp, t), full[np.arange(U), t], rtol=1e-13)
    H1, H0 = (-rng.gamma(2.0, 3.0, (U, Mg, 64)).astype(np.float32) for _ in range(2))
    H1[rng.random((U, Mg)) < 0.3] = 0.0
    _, _, ref = A.margin_vs_competitors(model, H1, H0, t, 1.7, batch=128)
    np.testing.assert_allclose(A.margin_vs_unknown(model, H1, H0, t, 1.7, batch=128), ref, rtol=1e-12, atol=1e-9)


def _two_state_share_plain(model, min_chunks=5, screen=0.05, max_groups=20, batch=500):
    """two_state_share as it was: one loop over chunk batches, then one bin after another."""
    H, (U, Mg, G) = model.n_bins, model.Lp_chunks.shape
    on_w, off_w, on_n, off_n = (np.zeros((H, Mg)) for _ in range(4))
    for s in range(0, U, batch):
        Lp = model.Lp_chunks[s:s + batch].astype(np.float64)
        _, lo, hi, _ = model.grid.summarize(Lp)
        info = Lp.max(-1) - Lp.min(-1)
        b = model.chunk_bin[s:s + batch]
        on, off = (lo >= 0.5) & (info > 0), (hi < 0.5) & (info > 0)
        np.add.at(on_w, b, info * on); np.add.at(off_w, b, info * off)
        np.add.at(on_n, b, on); np.add.at(off_n, b, off)
    tot = on_w + off_w
    minor = np.where((on_n + off_n >= min_chunks) & (np.minimum(on_n, off_n) >= 2) & (tot > 0),
                     np.minimum(on_w, off_w) / np.maximum(tot, 1e-300), 0.0)
    share, group = np.zeros(H), np.full(H, -1)
    upper = model.grid.x >= 0.5
    order = np.argsort(model.chunk_bin, kind="stable")
    bounds = np.searchsorted(model.chunk_bin[order], np.arange(H + 1))
    for h in range(H):
        ms = np.flatnonzero(minor[h] >= screen)
        if not len(ms):
            continue
        ms = ms[np.argsort(-minor[h, ms], kind="stable")[:max_groups]]
        idx = order[bounds[h]:bounds[h + 1]]
        ell = model.Lp_chunks[idx][:, ms, :].astype(np.float64)
        ell = ell - ell.max(-1, keepdims=True)
        rho = M.npmle(ell, iters=200, tol=1e-5)
        two = np.log(np.maximum(np.einsum("umg,mg->um", np.exp(ell), rho), 1e-300)).sum(0)
        one = ell.sum(0).max(-1)
        ok = two - one > np.log(len(idx))
        sh = np.where(ok, np.minimum(rho[:, upper].sum(1), rho[:, ~upper].sum(1)), 0.0)
        k = int(sh.argmax())
        if sh[k] > 0:
            share[h], group[h] = sh[k], ms[k]
    return share, group


@pytest.mark.parametrize("threads", ["1", "4"])
def test_two_state_share_threads_bits(monkeypatch, threads):
    """two_state_share with chunk rows and bins on threads = the plain loops, bit for bit."""
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    rng = np.random.default_rng(6)
    grid = Grid(64)
    U, Mg, H = 700, 9, 6
    model = object.__new__(M.FittedModel)
    model.grid, model.n_bins = grid, H
    model.chunk_bin = rng.integers(0, H, U)
    lvl = np.where(rng.random((U, Mg)) < 0.5, 0.1, 0.9)  # two states in every bin: the NPMLE runs
    k = rng.binomial(30, lvl)[..., None]
    x = grid.x[None, None, :]
    model.Lp_chunks = (k * np.log(x) + (30 - k) * np.log1p(-x)).astype(np.float32)
    model.Lp_chunks[rng.random((U, Mg)) < 0.2] = 0.0
    ref = _two_state_share_plain(model)
    got = model.two_state_share()
    assert (ref[1] >= 0).sum() >= 3
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])


@pytest.mark.parametrize("threads", ["1", "4"])
def test_summed_scores_row_form(monkeypatch, threads):
    """The row form of summed_scores (one-thread BLAS products on unit slices, one logarithm for the relatives)
    agrees with the group form to rounding, does not depend on the thread count, and leaves BLAS's thread count
    as it found it."""
    monkeypatch.setenv("OMP_NUM_THREADS", threads)
    rng = np.random.default_rng(7)
    U, Mg, H = 300, 11, 7
    Lp = _tables(rng, U, Mg)
    lp = np.log(rng.dirichlet(np.ones(64) * 0.05, (H, Mg)))  # sparse profiles: many cells far below the maximum
    lp[0, 0, :10] = -900.0
    Au = rng.normal(-30, 5, (U, Mg))
    Au = np.minimum(Au, Lp.max(-1))  # Au <= max of the unit's table, as for a community draw
    api = M._blas_thread_api()
    before = api[1]() if api else None
    S0, R0 = M.summed_scores(Lp, lp, rel=(0.05, Au), form="groups")
    S1, R1 = M.summed_scores(Lp, lp, rel=(0.05, Au), form="rows")
    np.testing.assert_allclose(S1, S0, rtol=1e-12)
    np.testing.assert_allclose(R1, R0, rtol=1e-12)
    assert (api[1]() if api else None) == before
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    S2, R2 = M.summed_scores(Lp, lp, rel=(0.05, Au), form="rows")
    assert np.array_equal(S1, S2) and np.array_equal(R1, R2)
    # units scored in batches of a multiple of ROWS_STEP: the same bits
    b = M.ROWS_STEP
    S3 = np.concatenate([M.summed_scores(Lp[s:s + b], lp, form="rows") for s in range(0, U, b)])
    assert np.array_equal(S3, M.summed_scores(Lp, lp, form="rows"))


def test_profile_exp_flush_keeps_products():
    """Products with the flushed profile exponentials equal those with the full ones (terms < 1e-250 vanish)."""
    rng = np.random.default_rng(8)
    lp = np.log(rng.dirichlet(np.ones(64) * 0.02, (50, 9)))
    a = rng.uniform(-80, 0, (200, 64)).astype(np.float32)
    x = np.exp(a - a.max(1, keepdims=True)).astype(np.float64)
    for h in range(9):
        b = lp[:, h, :]
        full = np.exp(b - b.max(1, keepdims=True))
        assert (full < M._TINY).any()
        np.testing.assert_array_equal(x @ M._profile_exp(b).T, x @ full.T)


def test_host_profiles_threads_bits(monkeypatch):
    """host_profiles on row slices on threads = the one-call expression, bit for bit."""
    from scipy.special import logsumexp
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    rng = np.random.default_rng(9)
    ell = -rng.gamma(2.0, 50.0, (40, 410, 64))
    log_pi = np.log(rng.dirichlet(np.ones(64), 410))
    lp = ell + log_pi[None]
    assert np.array_equal(M.host_profiles(ell, log_pi), lp - logsumexp(lp, axis=-1, keepdims=True))


def test_mix_log_is_logaddexp():
    rng = np.random.default_rng(10)
    grid = Grid(64)
    la, lb = rng.normal(-20, 15, 5000), rng.normal(-20, 15, 5000)
    la[:5], lb[3:8] = -np.inf, -np.inf          # one side impossible, and both
    lb[10:20] = la[10:20]                       # equal
    ref = np.logaddexp(la[:, None] + grid.logx, lb[:, None] + grid.log1mx)
    got = M.mix_log(la, lb, grid, block=777)
    assert np.array_equal(np.isneginf(ref), np.isneginf(got))
    f = np.isfinite(ref)
    np.testing.assert_allclose(got[f], ref[f], rtol=1e-13, atol=1e-13)


def test_unit_tables_by_keys_equal_the_group_by(monkeypatch):
    """unit_loglik_arrays through packed integer keys (numpy) = through polars' group-by, bit for bit."""
    from epirefine.detection import DetectionParams
    rng = np.random.default_rng(11)
    det = {g: DetectionParams(rng.uniform(0.001, 0.1), rng.uniform(5, 200), rng.uniform(0.6, 0.99), rng.uniform(5, 50))
           for g in range(7)}
    N = 50_000
    unit, group = rng.integers(0, 90, N), rng.integers(0, 7, N)
    n = np.minimum(rng.geometric(0.05, N), 3000)
    k = rng.binomial(n, 0.4)
    grid = Grid(64)
    got = M.unit_loglik_arrays(unit, group, n, k, 90, 7, det, grid, dtype=np.float32, batch_rows=20_000)
    monkeypatch.setattr(M, "_cells_by_keys", lambda *a: False)
    ref = M.unit_loglik_arrays(unit, group, n, k, 90, 7, det, grid, dtype=np.float32, batch_rows=20_000)
    assert np.array_equal(got, ref)


def test_spill_keeps_values(tmp_path, monkeypatch):
    import epirefine.pipeline as P
    a = np.random.default_rng(12).random((50, 7, 64)).astype(np.float32)
    assert P._spill(a, tmp_path) is a  # small: kept in memory
    monkeypatch.setattr(P, "SPILL_MIN_BYTES", 1)
    m = P._spill(a, tmp_path)
    assert isinstance(m, np.memmap) and np.array_equal(m, a) and np.array_equal(m[[3, 9]], a[[3, 9]])
    assert not list(tmp_path.iterdir())  # unlinked temporary file


def test_spilled_chunk_tables_same_outputs(community, base_run, tmp_path, monkeypatch):
    """A run whose chunk tables go to a temporary file after the final fit writes the same tables."""
    import filecmp
    import epirefine.pipeline as P
    from helpers import copy_cache, run_pipeline
    monkeypatch.setattr(P, "SPILL_MIN_BYTES", 1)
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")))
    for f in sorted(base_run.out.glob("*.tsv")):
        assert filecmp.cmp(f, run.out / f.name, shallow=False), f.name


@pytest.mark.parametrize("rs", ["0", "1"])
def test_motif_scan_in_pieces_same_records(monkeypatch, rs):
    """find_many with each batch cut into pieces at contig starts, scanned on threads: the records of one scan,
    in the same order."""
    monkeypatch.setenv("EPIREFINE_RS", rs)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    from epirefine.motifs import Motif, MotifMatcher, encode
    rng = np.random.default_rng(13)
    motifs = [Motif("GATC", 1, "a"), Motif("CCWGG", 1, "m"), Motif("GANTC", 1, "a"), Motif("TCGA", 3, "a"),
              Motif("CAGNNNNNNCTG", 1, "a"), Motif("RGATCY", 2, "a")]
    contigs = {f"c{i}": "".join(rng.choice(list("ACGT"), int(rng.integers(5, 3000)))) for i in range(300)}
    contigs["c7"] = "GATC" * 50 + "N" * 7 + "GATC"
    m = MotifMatcher(motifs)
    sep = int(m.length.max())
    seqs = list(contigs.values())
    codes = encode(("N" * sep).join(seqs))
    offsets = np.cumsum([0] + [len(x) + sep for x in seqs])[:-1]
    scanner = m._scanner()
    whole = (m.find if scanner is None else scanner.find)(codes)
    got = m._find_pieces(codes, offsets, min_piece=5_000)
    for k in whole:
        assert np.array_equal(whole[k], got[k]), k


def test_threads_follow_cpu_affinity(monkeypatch):
    """Thread and worker counts follow the cores the process may run on, not the machine's count."""
    import os
    from epirefine import runinfo
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert M._cores() == 3 and M._n_threads() == 3 and runinfo.n_workers(None) == 3
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(40)), raising=False)
    assert M._n_threads() == 16


def test_prefetch_cache_reads_parts(tmp_path):
    import json
    import epirefine.pipeline as P
    assert P._prefetch_cache(tmp_path) is None  # no manifest
    (tmp_path / "sites").mkdir()
    for i in range(3):
        (tmp_path / "sites" / f"p{i}.parquet").write_bytes(b"x" * 1000)
    (tmp_path / "manifest.json").write_text(json.dumps({"parts": [f"elsewhere/sites/p{i}.parquet" for i in range(3)]}))
    th = P._prefetch_cache(tmp_path)
    th.join(10)
    assert not th.is_alive()
