import numpy as np
import polars as pl

from epirefine.assignability import expected_tables, margin_vs_competitors
from epirefine.detection import DetectionParams
from epirefine.grid import Grid
from epirefine.model import FittedModel
from epirefine.motifs import Motif, assign_sites, build_records, revcomp

G = Grid(48)
DET = {g: DetectionParams(0.04, 20, 0.88, 20) for g in range(3)}


def _sim_L(phi, n_sites, cov=20, rng=None):
    """L[m, g] for one unit with true phi per group, using the model's own site likelihood."""
    from epirefine.model import unit_loglik
    rng = rng or np.random.default_rng(0)
    rows = []
    for m, p in enumerate(phi):
        for _ in range(n_sites):
            n = rng.poisson(cov) + 1
            mu = 0.88 if rng.random() < p else 0.04
            rows.append((0, m, n, rng.binomial(n, rng.beta(mu * 20, (1 - mu) * 20))))
    df = pl.DataFrame(rows, schema=["unit", "group", "n", "k"], orient="row")
    return unit_loglik(df, 1, len(phi), DET, G)[0]


def _model(profiles, chunks_per_bin=6, n_sites=60, contaminant=None, rng=None):
    """Bins with given phi per group; optional (bin, phi) contaminating chunk."""
    rng = rng or np.random.default_rng(1)
    L, cb = [], []
    for b, phi in enumerate(profiles):
        for _ in range(chunks_per_bin):
            L.append(_sim_L(phi, n_sites, rng=rng)); cb.append(b)
    if contaminant is not None:
        L.append(_sim_L(contaminant[1], n_sites, rng=rng)); cb.append(contaminant[0])
    return FittedModel(np.stack(L), np.array(cb), len(profiles), ["a"] * len(profiles[0]), G, float("inf"))


def test_transition_delta_shifts_mean():
    T = G.transition(100.0, 0.8)
    i = np.argmin(abs(G.x - 0.9))
    assert abs((T[i] * G.x).sum() - 0.72) < 0.02
    Ti = G.transition(float("inf"), 0.8)
    assert Ti.sum(1).max() == 1 and abs(G.x[Ti[i].argmax()] - 0.72) < 0.05


def test_cross_group_record_goes_to_most_specific_group():
    m = [Motif("TCGA", 3, "a"), Motif("GANTC", 1, "a")]
    seq = "TTTCGACTCTT" + "GAATC" * 3  # TCGA at 2 (A at 5) and GACTC at 4 (A at 5): shared record
    recs = build_records({"c": seq}, m)
    sites = assign_sites(recs, [0, 1])  # separate groups (in a real genome overlap is ~1/16)
    shared = sites.filter((pl.col("pos") == 5) & (pl.col("strand") == "+"))
    assert shared.height == 1 and shared["group"][0] == 0  # TCGA has fewer records: more specific


def test_robust_profiles_flag_contaminant():
    model = _model([[0.95, 0.01, 0.01], [0.01, 0.95, 0.01]], contaminant=(1, [0.95, 0.01, 0.01]))
    assert model.w[-1] < 0.05 and model.w[:-1].min() > 0.9
    phi = (np.exp(model.log_prof) * G.x).sum(-1)
    assert phi[1, 0] < 0.05  # contaminant did not leak into bin 1's profile


def test_expected_margin_zero_for_lookalikes_and_grows_with_sites():
    model = _model([[0.95, 0.01, 0.01], [0.95, 0.01, 0.01], [0.01, 0.95, 0.01]])
    rows = [(u, m, 20) for u, ns in enumerate([5, 50]) for m in range(3) for _ in range(ns)]
    H1, H0 = expected_tables(pl.DataFrame(rows, schema=["unit", "group", "n"], orient="row"), 2, 3, DET, G)
    m0, near0, _ = margin_vs_competitors(model, H1, H0, np.array([0, 0]), 1.0)
    assert abs(m0).max() < 0.5 and set(near0) == {1}
    m2, _, _ = margin_vs_competitors(model, H1, H0, np.array([2, 2]), 1.0)
    assert 0 < m2[0] < m2[1]


def test_summed_scores_match_full_tensor():
    from epirefine.model import group_scores, summed_scores
    rng = np.random.default_rng(3)
    Lp = rng.normal(-5, 3, (7, 5, G.size))
    prof = rng.normal(-4, 2, (40, 5, G.size))
    Au = rng.normal(-5, 3, (7, 5))
    A = group_scores(Lp, prof)
    assert np.allclose(summed_scores(Lp, prof), A.sum(-1))
    S, R = summed_scores(Lp, prof, rel=(0.2, Au))
    assert np.allclose(S, A.sum(-1))
    assert np.allclose(R, np.logaddexp(np.log(0.8) + A, np.log(0.2) + Au[:, None, :]).sum(-1))


def test_npmle_vectorised_matches_per_group_em():
    from scipy.special import logsumexp as lse
    from epirefine.model import npmle
    rng = np.random.default_rng(4)
    ell = -rng.gamma(2, 20, (30, 4, G.size))
    got = npmle(ell, iters=200, tol=0)
    for m in range(4):
        pi = np.full(G.size, 1 / G.size)
        for _ in range(200):
            lp = np.log(pi)[None] + ell[:, m]
            pi = np.maximum(np.exp(lp - lse(lp, 1, keepdims=True)).mean(0), 1e-300)
        assert np.allclose(got[m], pi, atol=1e-10)


def test_motif_matcher_matches_regex_with_iupac_and_n():
    import re
    from epirefine.motifs import IUPAC, MotifMatcher, encode
    rng = np.random.default_rng(5)
    seq = "".join(rng.choice(list("ACGT"), 5000)) + "NNN" + "".join(rng.choice(list("ACGT"), 3000))
    motifs = [Motif("RGATCY", 2, "a"), Motif("GANTC", 1, "a"), Motif("CCWGG", 1, "m"),
              Motif("GAGNNNNNGTCA", 1, "a"), Motif("TGACNNNNNCTC", 2, "a")]
    rec = MotifMatcher(motifs).find(encode(seq))
    got = sorted(zip(rec["motif"].tolist(), rec["pos"].tolist(), rec["strand"].tolist()))
    want = []
    for mi, m in enumerate(motifs):
        for strand, pat in ((0, m.seq), (1, revcomp(m.seq))):
            rx = re.compile("(?=" + "".join(IUPAC[c] for c in pat) + ")")
            for hit in rx.finditer(seq):
                want.append((mi, hit.start() + (m.mod_pos if strand == 0 else len(m.seq) - 1 - m.mod_pos), strand))
    assert got == sorted(want)


def test_fragment_selection_is_reproducible():
    from epirefine.calibrate import select_fragments
    hc = pl.DataFrame({"contig": list(range(40)), "bin_idx": [i % 7 for i in range(40)],
                       "length": [50_000 + 1000 * i for i in range(40)]})
    a = [select_fragments(hc.sample(fraction=1.0, shuffle=True, seed=s), (2000, 10000), 3, np.random.default_rng(1))
         for s in range(3)]
    assert all(x.equals(a[0]) for x in a)


def test_refine_flags_contaminant_includes_member_and_leaves_orphan():
    from epirefine.refine import em_weight, refine
    # 3 bins; contig likelihoods strongly favour one explanation each
    S = np.array([[0., -50, -50],    # binned in 0, fits 0: keep
                  [-50, 0., -50],    # binned in 0, fits 1: contaminant, move to 1
                  [-50, -50, -50],   # binned in 2, fits nothing (unknown): contaminant, stays out
                  [-50, -50, 0.],    # unbinned, fits 2: include
                  [-50, -50, -50],   # unbinned, fits nothing: stay unbinned
                  [0., 0., 0.]])     # unbinned, no information: stay unbinned
    Su = np.array([-40., -40, 0, -40, 0, 0])
    given = np.array([0, 0, 2, -1, -1, -1])
    r = refine(S, Su, given, given >= 0, temperature=1.0)
    assert r.contaminant.tolist() == [False, True, True, False, False, False]
    assert r.include_bin.tolist() == [-1, 1, -1, 2, -1, -1]
    w, resp = em_weight(np.zeros(100), np.r_[np.full(20, 10.0), np.full(80, -10.0)])
    assert abs(w - 0.2) < 0.01


def test_caller_fit_by_bin_recovers_rare_methylated_component():
    """One motif methylated in 2 of 40 bins: the pooled fit degenerates, the bin-labelled fit doesn't."""
    from epirefine.detection import fit_detection, fit_detection_by_bin
    rng = np.random.default_rng(7)
    rows = []
    for b in range(40):
        meth = b < 2
        n = rng.poisson(25, 300) + 1
        mu = np.where(meth & (rng.random(300) < 0.95), 0.85, 0.02)
        k = rng.binomial(n, rng.beta(mu * 12, (1 - mu) * 12))
        rows.append(pl.DataFrame({"group": 0, "bin": b, "n": n, "k": k}))
    sites = pl.concat(rows)
    counts = sites.group_by("group", "bin", "n", "k").agg(pl.len().alias("w"))
    by_bin = fit_detection_by_bin(counts, {0: "a"})[0]
    assert abs(by_bin.mu1 - 0.85) < 0.05 and abs(by_bin.mu0 - 0.02) < 0.01


def test_contamination_needs_a_methylome_difference():
    from epirefine.refine import refine
    S = np.array([[-50., -50, -50], [-50., -50, -50]])  # both fit their bin (0) badly, nothing else well
    Su = np.array([0., 0.])
    given = np.array([0, 0])
    small_shift, real = 0.15, 0.9
    r = refine(S, Su, given, given >= 0, 1.0, methylome_difference=np.array([small_shift, real]), min_difference=0.5)
    assert r.contaminant.tolist() == [False, True]


def test_redundancy_containment():
    from epirefine.motifs import encode
    from epirefine.redundancy import containment, sketch
    rng = np.random.default_rng(11)
    genome = "".join(rng.choice(list("ACGT"), 200_000))
    variant = list(genome[50_000:80_000])
    for p in rng.choice(len(variant), int(0.01 * len(variant)), replace=False):  # 99% identity
        variant[p] = "ACGT"[("ACGT".index(variant[p]) + 1) % 4]
    other = "".join(rng.choice(list("ACGT"), 30_000))
    g = sketch(encode(genome))
    assert containment(sketch(encode("".join(variant))), g) > 0.6
    assert containment(sketch(encode(other)), g) < 0.05
    rc = genome[120_000:150_000][::-1].translate(str.maketrans("ACGT", "TGCA"))
    assert containment(sketch(encode(rc)), g) > 0.95  # canonical k-mers: strand doesn't matter


def test_logit_kernel_spreads_intermediate_levels_more_than_extremes():
    g = Grid(64, kernel="logit")
    T = g.transition(0.5)
    assert np.allclose(T.sum(1), 1)
    sd = lambda i: np.sqrt((T[i] * (g.x - (T[i] * g.x).sum()) ** 2).sum())
    lo, mid = np.argmin(abs(g.x - 0.003)), np.argmin(abs(g.x - 0.5))
    assert sd(lo) < 0.005 < 0.08 < sd(mid)
    assert np.allclose(g.transition(0.0), np.eye(64))  # sigma 0 = no variation


def test_calibration_leaves_the_whole_source_contig_out():
    from epirefine.calibrate import contig_chunks
    chunk_contig = np.array([7, 7, 3, 7, 3, 9])
    got = contig_chunks(np.array([7, 3, 5]), chunk_contig)
    assert got[0].tolist() == [0, 1, 3] and got[1].tolist() == [2, 4] and got[2].tolist() == []


def test_bin_evidence_status():
    from epirefine.refine import bin_evidence_table
    mean = np.array([[0.95, 0.02], [0.02, 0.01], [0.4, 0.3]])
    lo = np.array([[0.9, 0.0], [0.0, 0.0], [0.05, 0.02]])
    hi = np.array([[0.99, 0.05], [0.05, 0.03], [0.9, 0.8]])
    nsg = np.array([[10, 10], [10, 10], [2, 2]])
    t = bin_evidence_table(["a", "b", "c"], (mean, lo, hi, None), nsg, np.array([0, 1, 2]))
    assert t["status"].to_list() == ["methylated", "no_methylation_detected", "no_methylation_evidence"]


def test_low_evidence_contigs_get_their_own_status(tmp_path):
    from epirefine.refine import refine, write_outputs
    S = np.array([[0., -50, -50], [0., -50, -50], [-50, -50, 0.], [-1., -1, -1]])
    Su = np.array([-40., -40, -40, -1])
    given = np.array([0, 0, -1, -1])
    r = refine(S, Su, given, given >= 0, temperature=1.0)
    P = np.full((4, 4), 0.25)
    write_outputs(tmp_path, ["a", "b", "c", "d"], {"a": 10, "b": 10, "c": 10, "d": 10}, given, ["x", "y", "z"], r,
                  np.array([100, 2, 100, 1]), set(), P, evidence=np.array([20.0, 0.5, 20.0, np.nan]))
    st = dict(pl.read_csv(tmp_path / "contig_decisions.tsv", separator="\t").select("contig", "status").iter_rows())
    assert st == {"a": "kept", "b": "no_methylation_evidence", "c": "included", "d": "no_methylation_evidence"}


def test_low_coverage_binned_contigs_are_not_judged(tmp_path):
    from epirefine.refine import refine, write_outputs
    S = np.array([[0., -50, -50], [0., -50, -50], [-50, -50, 0.]])
    given = np.array([0, 0, -1])
    low = np.array([False, True, False])
    r = refine(S, np.array([-40., -40, -40]), given, (given >= 0) & ~low, temperature=1.0)
    write_outputs(tmp_path, ["a", "b", "c"], {"a": 10, "b": 10, "c": 10}, given, ["x", "y", "z"], r,
                  np.array([100, 100, 100]), set(), np.full((3, 4), 0.25), evidence=np.array([20.0, 20.0, 20.0]),
                  coverage=np.array([12.0, 2.0, 3.0]), low_coverage=low)
    d = pl.read_csv(tmp_path / "contig_decisions.tsv", separator="\t")
    assert dict(d.select("contig", "status").iter_rows()) == {"a": "kept", "b": "low_coverage", "c": "included"}
    assert d["coverage"].to_list() == [12.0, 2.0, 3.0]


def test_no_decision_below_coverage_cutoff():
    from epirefine.refine import refine
    S = np.array([[0., -50, -50], [-50, 0., -50], [-50, -50, 0.], [-50, -50, 0.]])
    given = np.array([0, 0, -1, -1])
    dec = np.array([True, False, True, False])  # contigs 1 and 3: coverage too low
    r = refine(S, np.array([-40., -40, -40, -40]), given, given >= 0, temperature=1.0, decidable=dec)
    assert r.contaminant.tolist() == [False, False, False, False]  # contig 1 fits bin 1 but isn't judged
    assert r.include_bin.tolist() == [-1, -1, 2, -1]              # contig 3 fits bin 2 but isn't included


def test_short_contigs_get_status_too_short(tmp_path):
    from epirefine.refine import refine, write_outputs
    S = np.array([[0., -50, -50], [-50, -50, 0.], [-50, -50, 0.]])
    given = np.array([0, -1, -1])
    short = np.array([False, False, True])  # contig c fits bin z but is shorter than the cutoff
    r = refine(S, np.array([-40., -40, -40]), given, given >= 0, temperature=1.0, decidable=~short)
    write_outputs(tmp_path, ["a", "b", "c"], {"a": 5000, "b": 5000, "c": 800}, given, ["x", "y", "z"], r,
                  np.array([100, 100, 100]), set(), np.full((3, 4), 0.25), evidence=np.array([20.0, 20.0, 20.0]),
                  coverage=np.array([12.0, 12.0, 12.0]), low_coverage=np.zeros(3, bool), too_short=short)
    d = pl.read_csv(tmp_path / "contig_decisions.tsv", separator="\t")
    assert dict(d.select("contig", "status").iter_rows()) == {"a": "kept", "b": "included", "c": "too_short"}


def test_inclusion_into_unmethylated_bin_needs_higher_posterior():
    from epirefine.refine import refine
    # contig 0 fits bin 1 (unmethylated) with ~0.93 posterior, contig 1 fits bin 0 (methylated) alike
    S = np.array([[-2.6, 0.0, -50.0], [0.0, -2.6, -50.0]])
    given = np.array([-1, -1])
    r = refine(S, np.array([-40., -40.]), given, given >= 0, temperature=1.0, p_unknown=0.01,
               unmethylated_bins=np.array([False, True, False]), include_threshold_unmethylated=0.95)
    assert r.include_bin.tolist() == [-1, 0]


def test_fragment_rows_match_the_join():
    """The range lookup returns the (unit, row) pairs of the join on contig filtered on position, in its order."""
    from epirefine.sites import _fragment_rows
    rng = np.random.default_rng(3)
    cid = np.sort(rng.integers(0, 6, 4000)).astype(np.int32)
    pos = np.concatenate([np.sort(rng.integers(0, 30_000, int((cid == c).sum()))) for c in range(6)]).astype(np.int64)
    fr = pl.DataFrame({"unit": np.arange(40), "contig": rng.integers(0, 8, 40).astype(np.int32),
                       "start": rng.integers(0, 6, 40) * 5000}).with_columns((pl.col("start") + 5000).alias("end"))
    fr = pl.concat([fr, fr.head(5).with_columns(pl.col("unit") + 40, pl.lit(0, pl.Int64).alias("start"),
                                                  pl.lit(20_000, pl.Int64).alias("end"))])

    def join(cid, pos):
        f = (pl.DataFrame({"row": np.arange(len(cid)), "contig": cid, "pos": pos}).join(fr, on="contig", how="inner")
             .filter((pl.col("pos") >= pl.col("start")) & (pl.col("pos") < pl.col("end"))))
        return f["unit"].to_numpy(), f["row"].to_numpy()
    u, r = _fragment_rows(cid, pos, fr)
    ju, jr = join(cid, pos)
    assert len(r) > 500 and np.array_equal(u, ju) and np.array_equal(r, jr)
    # a part that is not sorted takes the join itself
    p = rng.permutation(len(cid))
    u, r = _fragment_rows(cid[p], pos[p], fr)
    ju, jr = join(cid[p], pos[p])
    assert sorted(zip(u.tolist(), r.tolist())) == sorted(zip(ju.tolist(), jr.tolist()))
    u, r = _fragment_rows(cid, pos, fr.filter(pl.col("contig") > 6))
    assert len(u) == 0 and len(r) == 0


def test_site_terms_blocks_are_bit_identical():
    from epirefine.detection import DetectionParams
    from epirefine.grid import Grid
    from epirefine.model import site_terms
    rng = np.random.default_rng(5)
    grid = Grid(64)
    det = {0: DetectionParams(0.03, 20.0, 0.85, 8.0, 0.5, 100), 1: DetectionParams(0.1, 5.0, 0.6, 3.0, 0.5, 100)}
    n = rng.integers(1, 300, 1000)
    keys = pl.DataFrame({"group": rng.integers(0, 2, 1000), "n": n, "k": rng.integers(0, n + 1)})
    grp, nf, kf = keys["group"].to_numpy(), n.astype(float), keys["k"].to_numpy().astype(float)
    from epirefine.detection import bb_logpmf
    mu0, k0, mu1, k1 = (np.array([getattr(det[g], f) for g in (0, 1)])[grp] for f in ("mu0", "kappa0", "mu1", "kappa1"))
    whole = np.logaddexp(bb_logpmf(kf, nf, mu1, k1)[:, None] + grid.logx[None, :],
                         bb_logpmf(kf, nf, mu0, k0)[:, None] + grid.log1mx[None, :])
    # (one logarithm per cell instead of logaddexp since 0.11: equal to rounding; blocks do not change a bit)
    first = site_terms(keys, det, grid, block=1)
    np.testing.assert_allclose(first, whole, rtol=1e-13)
    for block in (7, 256, 65_536):
        assert np.array_equal(site_terms(keys, det, grid, block=block), first)


def test_summed_scores_same_bits_whatever_the_memory_budget(monkeypatch):
    monkeypatch.setenv("EPIREFINE_RS", "0")  # the numpy sums (the kernel of epirefine-rs: tests/test_rs_model.py)
    """Scores don't depend on how many group blocks run at once, and equal the block-wise sums they replace."""
    from epirefine import model as md
    rng = np.random.default_rng(11)
    U, M, G, H = 400, 37, 64, 30
    Lp = (-rng.gamma(2.0, 20.0, (U, M, G))).astype(np.float32)
    prof = np.log(rng.dirichlet(np.ones(G) * 0.3, (H, M)))
    Au = -rng.gamma(2.0, 5.0, (U, M))
    eta = 0.05
    # the sums as they were formed before: one accumulator per block of groups, then the blocks added up
    A = md.group_scores(Lp, prof)                                   # [U, H, M]
    RA = np.logaddexp(A + np.log1p(-eta), (np.log(eta) + Au)[:, None, :])
    blocks = [b for b in np.array_split(np.arange(M), min(M, md._n_threads())) if len(b)]

    def blockwise(X):
        parts = []
        for b in blocks:
            acc = np.zeros((U, H))
            for m in b:
                acc += X[:, :, m]
            parts.append(acc)
        return sum(parts)
    S0, R0 = blockwise(A), blockwise(RA)
    per_block = 3 * 8 * U * H
    for budget in (1, per_block, 3 * per_block, 7 * per_block, 1 << 40, None):
        S, R = md.summed_scores(Lp, prof, rel=(eta, Au), max_bytes=budget)
        assert np.array_equal(S, S0) and np.array_equal(R, R0), budget
        assert np.array_equal(md.summed_scores(Lp, prof, max_bytes=budget), S0), budget
    assert np.allclose(S0, A.sum(-1), rtol=1e-12, atol=0)


def test_scores_keep_lp_rows_of_chosen_units_only():
    """score(lp_rows=...) gives the same scores, and leave-one-out rescoring of a kept unit the same bits."""
    import pytest
    rng = np.random.default_rng(4)
    model = _model([[0.95, 0.01, 0.01], [0.01, 0.95, 0.01], [0.01, 0.01, 0.95]], rng=rng)
    model.set_unknown_params(0.05, 0.9)
    L = np.stack([_sim_L(p, 40, rng=rng) for p in ([0.95, 0.01, 0.01], [0.01, 0.95, 0.01], [0.5, 0.5, 0.5],
                                                 [0.01, 0.01, 0.95], [0.95, 0.95, 0.01])])
    full, part = model.score(L), model.score(L, lp_rows=[1, 3])
    assert part.Lp.shape[0] == 2 and np.array_equal(part.Lp, full.Lp[[1, 3]])
    for f in ("S", "Su", "Au", "Su_comm", "R"):
        assert np.array_equal(getattr(full, f), getattr(part, f))
    for i, b in ((1, 1), (3, 2)):
        a = model.loo_rescore(full, i, b, np.array([6 * b]))
        assert np.array_equal(a, model.loo_rescore(part, i, b, np.array([6 * b])))
    assert np.array_equal(full.S, part.S) and np.array_equal(full.R, part.R) and np.array_equal(full.Su, part.Su)
    assert np.array_equal(model.host_terms(full, np.array([3, 1]), np.array([0, 2])),
                          model.host_terms(part, np.array([3, 1]), np.array([0, 2])))
    for i in (0, 2, 4):
        with pytest.raises(KeyError):
            part.lp(i)


def test_expected_tables_chunks_are_bit_identical():
    """Reducing T[r] * w in blocks of whole (unit, group) cells gives the bits of the single reduction."""
    rng = np.random.default_rng(8)
    n = 6000
    sites = pl.DataFrame({"unit": rng.integers(0, 40, n), "group": rng.integers(0, 3, n), "n": rng.integers(1, 80, n)})
    table: dict = {}
    whole = expected_tables(sites, 41, 3, DET, G, dtype=np.float32, table=table, chunk_rows=10 ** 9)
    assert np.abs(whole[0]).sum() > 0 and not whole[0][40].any()
    for chunk_rows in (1, 2, 17, 100, 1000):
        part = expected_tables(sites, 41, 3, DET, G, dtype=np.float32, table=table, chunk_rows=chunk_rows)
        assert np.array_equal(part[0], whole[0]) and np.array_equal(part[1], whole[1]), chunk_rows


def test_outlier_responsibility_does_not_depend_on_the_batch():
    model = _model([[0.95, 0.01, 0.01], [0.01, 0.95, 0.01]], contaminant=(1, [0.95, 0.01, 0.01]))
    ref = model._outlier_responsibility(0.05, batch=200)
    for batch in (1, 3, 5, None):  # (the cached community term _s_out is kept: it is from the first robust round)
        assert np.array_equal(model._outlier_responsibility(0.05, batch=batch), ref)


def test_batched_summary_does_not_depend_on_the_batch():
    from epirefine.pipeline import _batched_summary
    L = -np.random.default_rng(2).gamma(2.0, 10.0, (57, 5, G.size))
    ref = G.summarize(L)
    for batch in (1, 4, 50, 2000):
        got = _batched_summary(G, lambda sl: L[sl], len(L), batch=batch)
        assert all(np.array_equal(a, b) for a, b in zip(got, ref))


def test_lean_logsumexp_has_the_bits_of_scipy():
    import scipy
    from scipy.special import logsumexp as ref
    from epirefine.model import logsumexp as lse
    rng = np.random.default_rng(0)
    same = np.array_equal if tuple(int(v) for v in scipy.__version__.split(".")[:2]) >= (1, 15) else np.allclose
    for shape, axis, dt in (((300, 40, 64), -1, np.float64), ((300, 50), 1, np.float64), ((64,), -1, np.float64),
                            ((100, 7, 64), -1, np.float32), ((1, 495), 1, np.float64)):
        a = (-rng.gamma(2, 30, shape)).astype(dt)
        a.reshape(-1)[::17] = -np.inf
        if a.ndim > 1 and len(a) > 2:
            a[0] = -np.inf          # nothing in the row
            a[1] = a[1].max()       # every element tied for the maximum
            a[2, ..., :2] = 0.0     # two tied maxima
        for kd in (False, True):
            x, y = ref(a, axis=axis, keepdims=kd), lse(a, axis=axis, keepdims=kd)
            assert np.shape(x) == np.shape(y) and x.dtype == y.dtype and same(x, y, equal_nan=True)
    odd = np.array([[np.inf, 1.0], [np.nan, 1.0], [-np.inf, -np.inf]])
    assert np.array_equal(ref(odd, axis=1), lse(odd, axis=1), equal_nan=True)
