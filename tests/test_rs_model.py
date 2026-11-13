"""The optional compiled model kernels (package epirefine-rs, feature model_kernels): each returns what the numpy
code returns to the last bits of exp, log and sums, a whole run takes the same decisions, and EPIREFINE_RS=0
switches them off. Skipped when the package is not installed (or is a build without these kernels)."""
import numpy as np
import pytest

import epirefine.model as M
from epirefine.grid import Grid
from helpers import assert_same_outputs_to_rounding, copy_cache, run_pipeline

rs = pytest.importorskip("epirefine_rs")
pytestmark = pytest.mark.skipif(not rs.supports("model_kernels"), reason="epirefine-rs without model kernels")


def _both(monkeypatch, fn):
    """fn() with the numpy code, then with the kernels."""
    monkeypatch.setenv("EPIREFINE_RS", "0")
    assert M._rs() is None
    ref = fn()
    monkeypatch.delenv("EPIREFINE_RS")
    assert M._rs() is not None
    return ref, fn()


def _tables(rng, U=90, Mg=13, G=64):
    L = -rng.gamma(2.0, 40.0, (U, Mg, G))
    L[rng.random((U, Mg)) < 0.2] = 0.0
    return L.astype(np.float32)


def test_bin_loglik(monkeypatch):
    rng = np.random.default_rng(1)
    Lp, cb, w = _tables(rng), rng.integers(0, 6, 90), rng.random(90)
    cb[cb == 2] = 3  # an empty bin
    ref, got = _both(monkeypatch, lambda: M.bin_loglik(Lp, cb, 6, w))
    assert got.shape == ref.shape and np.allclose(got, ref, rtol=1e-12, atol=0)
    ref, got = _both(monkeypatch, lambda: M.bin_loglik(Lp, cb, 6))
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("tol,iters", [(1e-6, 300), (1e-3, 300), (1e-2, 300), (0.0, 25)])
def test_npmle(monkeypatch, tol, iters):
    rng = np.random.default_rng(2)
    ell = -rng.gamma(2.0, 3.0, (60, 41, 64)) * rng.integers(1, 40, (60, 41, 1))
    ref, got = _both(monkeypatch, lambda: M.npmle(ell, iters, tol))
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-300)
    start = ref
    ref, got = _both(monkeypatch, lambda: M.npmle(ell * 1.2, iters, tol, init=start))
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-300)


def test_outlier_responsibility_and_model(monkeypatch):
    rng = np.random.default_rng(3)
    grid = Grid(64)
    L, cb = _tables(rng, U=120, Mg=7), rng.integers(0, 4, 120)
    L[5] = 0.0  # a chunk without information

    def fit():
        m = M.FittedModel(L, cb, 4, ["a"] * 4 + ["m"] * 3, grid, 100.0)
        return m.w, m.log_prof, m.log_pi, m._outlier_responsibility(0.05)
    ref, got = _both(monkeypatch, fit)
    for a, b in zip(ref, got):
        assert np.allclose(b, a, rtol=1e-9, atol=1e-12)
    # ties and -inf in the logsumexp, as the numpy code treats them
    lp = np.zeros((3, 2, 64), np.float32); lp[1, 0, :] = -np.inf; lp[2, 1, 3:] = -50.0
    log_pi = np.log(np.full((2, 64), 1 / 64))
    want = M.logsumexp(lp.astype(np.float64) + log_pi[None], -1).sum(-1)
    with np.errstate(all="ignore"):
        assert np.array_equal(rs.community_terms(lp, log_pi), want) or np.allclose(rs.community_terms(lp, log_pi), want, rtol=1e-14)


def test_run_takes_the_same_decisions(monkeypatch, community, base_run, tmp_path):
    """A whole run with the kernels against one without, from the same site cache: every output table equal as
    written (four decimals), the numbers of params.json (written in full) to 1e-9."""
    monkeypatch.setenv("EPIREFINE_RS", "0")
    py = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "py")))
    monkeypatch.delenv("EPIREFINE_RS")
    k = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "rs")))
    assert_same_outputs_to_rounding(py.out, k.out)


def test_cell_sums_returns_numpys_bits(monkeypatch):
    if not rs.supports("cell_sums"):
        pytest.skip("epirefine-rs without cell_sums")
    rng = np.random.default_rng(4)
    table = -rng.gamma(2.0, 3.0, (300, 64))
    r, w = rng.integers(0, 300, 150_000), rng.integers(1, 9, 150_000).astype(float)
    starts = np.unique(np.r_[0, rng.integers(1, 100_000, 9_000), 100_000, 100_300])
    ref, got = _both(monkeypatch, lambda: M._cell_sums(table, r, w, starts))
    assert np.array_equal(got, ref) and np.array_equal(ref, np.add.reduceat(table[r] * w[:, None], starts, axis=0))


def test_motif_grouping_scan_gives_the_same_records(monkeypatch, community):
    """build_records (the scan behind the motif groups) with the compiled scanner: the same table, row for row."""
    if not rs.supports("motif_scan"):
        pytest.skip("epirefine-rs without the motif scan")
    from epirefine import io as pio
    from epirefine.motifs import build_records
    args = community.args("unused")
    motifs = pio.read_motifs(args["motifs_path"])
    contigs = dict(pio.iter_fasta(args["assembly"]))
    monkeypatch.setenv("EPIREFINE_RS", "0")
    ref = build_records(contigs, motifs)
    monkeypatch.delenv("EPIREFINE_RS")
    got = build_records(contigs, motifs)
    assert ref.height > 0 and ref.schema == got.schema and ref.equals(got)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_summed_scores(monkeypatch, dtype):
    """The scoring sums with the kernel: equal to the numpy sums to rounding (the exponentials are numpy's in both)."""
    if not rs.supports("score_sums"):
        pytest.skip("epirefine-rs without score_sums")
    rng = np.random.default_rng(6)
    grid = Grid(64)
    L, cb = _tables(rng, U=300, Mg=9), rng.integers(0, 5, 300)
    monkeypatch.setenv("EPIREFINE_RS", "0")
    m = M.FittedModel(L, cb, 5, ["a"] * 9, grid, 100.0)
    Lp = m.marginal(_tables(rng, U=130, Mg=9)).astype(dtype)
    Au = M.group_scores(Lp, m.log_pi[None])[:, 0, :]
    ref, got = _both(monkeypatch, lambda: M.summed_scores(Lp, m.log_prof, rel=(0.05, Au)))
    assert np.allclose(got[0], ref[0], rtol=1e-12, atol=0) and np.allclose(got[1], ref[1], rtol=1e-12, atol=0)
    ref, got = _both(monkeypatch, lambda: M.summed_scores(Lp, m.log_prof))
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_score_in_batches_is_score(monkeypatch):
    """Scoring in unit batches (kernel only) returns what scoring all units at once returns with the kernel."""
    if not rs.supports("score_sums"):
        pytest.skip("epirefine-rs without score_sums")
    rng = np.random.default_rng(7)
    grid = Grid(64)
    L, cb = _tables(rng, U=200, Mg=9), rng.integers(0, 5, 200)
    m = M.FittedModel(L, cb, 5, ["a"] * 9, grid, 100.0)
    m.set_unknown_params(0.05, 0.9)
    Lt = _tables(rng, U=8192, Mg=9)  # (batches large enough for summed_scores to use the kernel)
    rows = np.sort(rng.choice(8192, 700, replace=False))
    whole = m.score(Lt, lp_rows=rows)
    monkeypatch.setattr(M.FittedModel, "SCORE_BATCH", 4096)
    parts = m.score(Lt, lp_rows=rows)
    for f in ("S", "R", "Lp", "lp_rows"):
        assert np.array_equal(getattr(parts, f), getattr(whole, f)), f
    # Au comes from a BLAS product (group_scores); with several BLAS threads a product of 4,096 rows may sum in
    # another order than one of 8,192 rows: one ulp of a log-sum near 1 (4.4e-16 absolute), which a purely
    # relative test fails on cells whose log is near 0. These are log values: compare them absolutely too.
    for f in ("Au", "Su", "Su_comm"):
        assert np.allclose(getattr(parts, f), getattr(whole, f), rtol=1e-12, atol=1e-12), f
