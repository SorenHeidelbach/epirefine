import numpy as np
import polars as pl

from pbinner.assignability import expected_tables, margin_vs_competitors
from pbinner.detection import DetectionParams
from pbinner.grid import Grid
from pbinner.model import FittedModel
from pbinner.motifs import Motif, assign_sites, build_records, revcomp

G = Grid(48)
DET = {g: DetectionParams(0.04, 20, 0.88, 20) for g in range(3)}


def _sim_L(phi, n_sites, cov=20, rng=None):
    """L[m, g] for one unit with true phi per group, using the model's own site likelihood."""
    from pbinner.model import unit_loglik
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


def test_cross_group_records_dropped():
    m = [Motif("TCGA", 3, "a"), Motif("GANTC", 1, "a")]
    seq = "TTTCGACTCTT"  # TCGA at 2 (A at 5) and GACTC at 4 (A at 5): shared record
    recs = build_records({"c": seq}, m)
    sites = assign_sites(recs, [0, 1])  # separate groups (in a real genome overlap is ~1/16)
    assert not ((sites["pos"] == 5) & (sites["strand"] == "+")).any()


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
    from pbinner.model import group_scores, summed_scores
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
    from pbinner.model import npmle
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
    from pbinner.motifs import IUPAC, MotifMatcher, encode
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
    from pbinner.calibrate import select_fragments
    hc = pl.DataFrame({"contig": list(range(40)), "bin_idx": [i % 7 for i in range(40)],
                       "length": [50_000 + 1000 * i for i in range(40)]})
    a = [select_fragments(hc.sample(fraction=1.0, shuffle=True, seed=s), (2000, 10000), 3, np.random.default_rng(1))
         for s in range(3)]
    assert all(x.equals(a[0]) for x in a)


def test_refine_flags_contaminant_includes_member_and_leaves_orphan():
    from pbinner.refine import em_weight, refine
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
    from pbinner.detection import fit_detection, fit_detection_by_bin
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
    from pbinner.refine import refine
    S = np.array([[-50., -50, -50], [-50., -50, -50]])  # both fit their bin (0) badly, nothing else well
    Su = np.array([0., 0.])
    given = np.array([0, 0])
    small_shift, real = 0.15, 0.9
    r = refine(S, Su, given, given >= 0, 1.0, methylome_difference=np.array([small_shift, real]), min_difference=0.5)
    assert r.contaminant.tolist() == [False, True]


def test_redundancy_containment():
    from pbinner.motifs import encode
    from pbinner.redundancy import containment, sketch
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
    from pbinner.calibrate import contig_chunks
    chunk_contig = np.array([7, 7, 3, 7, 3, 9])
    got = contig_chunks(np.array([7, 3, 5]), chunk_contig)
    assert got[0].tolist() == [0, 1, 3] and got[1].tolist() == [2, 4] and got[2].tolist() == []


def test_bin_evidence_status():
    from pbinner.refine import bin_evidence_table
    mean = np.array([[0.95, 0.02], [0.02, 0.01], [0.4, 0.3]])
    lo = np.array([[0.9, 0.0], [0.0, 0.0], [0.05, 0.02]])
    hi = np.array([[0.99, 0.05], [0.05, 0.03], [0.9, 0.8]])
    nsg = np.array([[10, 10], [10, 10], [2, 2]])
    t = bin_evidence_table(["a", "b", "c"], (mean, lo, hi, None), nsg, np.array([0, 1, 2]))
    assert t["status"].to_list() == ["methylated", "no_methylation_detected", "no_methylation_evidence"]


def test_low_evidence_contigs_get_their_own_status(tmp_path):
    from pbinner.refine import refine, write_outputs
    S = np.array([[0., -50, -50], [0., -50, -50], [-50, -50, 0.], [-1., -1, -1]])
    Su = np.array([-40., -40, -40, -1])
    given = np.array([0, 0, -1, -1])
    r = refine(S, Su, given, given >= 0, temperature=1.0)
    P = np.full((4, 4), 0.25)
    write_outputs(tmp_path, ["a", "b", "c", "d"], {"a": 10, "b": 10, "c": 10, "d": 10}, given, ["x", "y", "z"], r,
                  np.array([100, 2, 100, 1]), set(), P, evidence=np.array([20.0, 0.5, 20.0, np.nan]))
    st = dict(pl.read_csv(tmp_path / "contig_decisions.tsv", separator="\t").select("contig", "status").iter_rows())
    assert st == {"a": "kept", "b": "no_methylation_evidence", "c": "included", "d": "no_methylation_evidence"}
