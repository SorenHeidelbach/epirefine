import numpy as np
import polars as pl
import pytest

from epirefine.detection import DetectionParams, fit_mixture
from epirefine.grid import Grid
from epirefine.io import _parse_pileup_block, read_pileup_at_sites
from epirefine.model import marginalize_unit, posterior, score_units
from epirefine.motifs import Motif, assign_sites, build_records, find_records, group_motifs, revcomp


def test_revcomp_iupac():
    assert revcomp("GATC") == "GATC"
    assert revcomp("GAGNNNNNGTCA") == "TGACNNNNNCTC"
    assert revcomp("RGATCY") == "RGATCY"


def test_palindrome_records_share_site():
    seq = "TTTGATCTTT"
    pos, strand, occ = find_records(seq, Motif("GATC", 1, "a"))
    assert sorted(zip(pos.tolist(), strand.tolist())) == [(4, 0), (5, 1)]  # A on +, A (T on +) on -
    assert len(set(occ.tolist())) == 1


def test_non_palindrome_both_strands():
    seq = "GAAGACTTTT" + revcomp("GAAGAC")
    pos, strand, occ = find_records(seq, Motif("GAAGAC", 4, "a"))
    assert sorted(zip(pos.tolist(), strand.tolist())) == [(4, 0), (11, 1)]
    assert seq[11] == "T"  # A on the minus strand
    assert len(set(occ.tolist())) == 2


def test_complement_motifs_merged_and_share_sites():
    m1, m2 = Motif("GAGNNNNNGTCA", 1, "a"), Motif("TGACNNNNNCTC", 2, "a")
    seq = "TTGAGACGTAGTCATT"
    recs = build_records({"c": seq}, [m1, m2])
    groups = group_motifs(recs, [m1, m2])
    assert groups[0] == groups[1]
    sites = assign_sites(recs, groups)
    assert sites.height == 2 and sites["site"].n_unique() == 1  # one occurrence, two methylated As


def test_nested_motifs_grouped_and_deduplicated():
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 20000))
    motifs = [Motif("GATC", 1, "a"), Motif("RGATCY", 2, "a"), Motif("CCWGG", 1, "m")]
    recs = build_records({"c": seq}, motifs)
    groups = group_motifs(recs, motifs)
    # RGATCY is a quarter of GATC's sites: nested, not the same motif, so not merged
    assert len({groups[0], groups[1], groups[2]}) == 3
    sites = assign_sites(recs, groups)
    assert sites.unique(["contig", "pos", "strand", "mod_type"]).height == sites.height
    # every RGATCY position goes to the more specific RGATCY group, none is dropped
    n_rgatcy = recs.filter(pl.col("motif") == 1).height
    assert sites.filter(pl.col("group") == groups[1]).height == n_rgatcy
    assert sites.height == recs.unique(["contig", "pos", "strand", "mod_type"]).height


@pytest.mark.parametrize("sep", [" ", "\t"])
def test_pileup_both_formats(tmp_path, sep):
    line = sep.join(["10", "50.00", "5", "5", "0", "0", "0", "0", "0"])
    text = f"c1\t4\t5\ta\t10\t+\t4\t5\t255,0,0\t{line}\nc1\t9\t10\tm\t3\t-\t9\t10\t255,0,0\t{line}\n"
    df = _parse_pileup_block(text.encode())
    assert df["n_valid"].to_list() == [10, 10] and df["n_mod"].to_list() == [5, 5]
    p = tmp_path / "p.bed"
    p.write_text(text)
    recs = pl.DataFrame({"contig": ["c1"], "pos": [4], "strand": ["+"], "mod_type": ["a"], "group": [0]})
    out = read_pileup_at_sites(p, recs, chunk_bytes=30)  # tiny chunks exercise line splitting
    assert out.height == 1 and out["n_mod"][0] == 5


def test_detection_mixture_recovers_parameters():
    rng = np.random.default_rng(1)
    n = rng.poisson(30, 20000) + 1
    meth = rng.random(20000) < 0.3
    mu = np.where(meth, 0.85, 0.05)
    k = rng.binomial(n, rng.beta(mu * 20, (1 - mu) * 20))
    fit = fit_mixture(n, k, np.ones_like(n), DetectionParams(0.05, 10, 0.85, 10), 3.0)
    assert abs(fit.mu0 - 0.05) < 0.01 and abs(fit.mu1 - 0.85) < 0.02 and abs(fit.pi1 - 0.3) < 0.02
    assert 12 < fit.kappa1 < 30


def test_marginalize_identity_and_empty_units():
    g = Grid(32)
    L = np.zeros((2, 1, 32))
    L[0, 0] = np.linspace(-5, 0, 32)
    assert np.allclose(marginalize_unit(L, g.transition(None)), L)
    assert np.allclose(marginalize_unit(L, g.transition(30.0))[1], 0.0)  # no data -> no evidence


def test_score_prefers_matching_host_and_unknown_is_fallback():
    g = Grid(32)
    hi, lo = g.beta_mass(50, 2), g.beta_mass(2, 50)
    prof = np.log(np.stack([np.stack([hi, lo]), np.stack([lo, hi])]))  # host0: m0 on; host1: m1 on
    pi = np.log(np.stack([(hi + lo) / 2, (hi + lo) / 2]))
    L = np.zeros((1, 2, 32))
    L[0, 0] = 40 * g.logx  # unit: m0 methylated
    L[0, 1] = 40 * g.log1mx  # m1 unmethylated
    S = score_units(L, prof)
    Su = score_units(L, pi[None])[:, 0]
    P = posterior(S, Su, 0.1)
    assert P[0].argmax() == 0 and P[0, 0] > 0.9
    P_masked = posterior(S, Su, 0.1, host_mask=np.array([[False, True]]))
    assert P_masked[0].argmax() == 2  # true host removed -> unknown, not the wrong host
