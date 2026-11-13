"""Rules revisited after the 2026-11-10 reviews: robust fit convergence report, flagged share, regional withdrawal,
contamination rate after the gate, redundancy among inclusions. (The damped robust fit and the merged-bin rule
--max-flagged-share were removed in 0.12, U12.)"""
import numpy as np
import polars as pl

from epirefine.detection import DetectionParams
from epirefine.grid import Grid
from epirefine.model import FittedModel, unbinned_state_frequencies, unit_loglik
from epirefine.redundancy import inclusion_redundancy
from epirefine.refine import flagged_share_per_bin, refine, regional_pattern, withdraw_as_mixed


def two_genome_bin(share_major, sites=60, n_chunks=40, n_other=10, M=8, seed=0, reads=20):
    """Bin 0: n_chunks chunks, share_major of them genome A, the rest genome B (4 of 8 groups differ);
    n_other further bins of 10 chunks each."""
    rng = np.random.default_rng(seed)
    det = {m: DetectionParams(0.03, 20, 0.9, 20) for m in range(M)}
    prof = (rng.random((n_other + 2, M)) < 0.3).astype(float)
    prof[0] = 0; prof[0, :M // 2] = 1
    prof[1] = prof[0]; prof[1, :4] = 1 - prof[0, :4]
    n_major = int(round(share_major * n_chunks))
    genome = np.array([0] * n_major + [1] * (n_chunks - n_major) + [g for g in range(2, n_other + 2) for _ in range(10)])
    chunk_bin = np.array([0] * n_chunks + [g - 1 for g in range(2, n_other + 2) for _ in range(10)])
    rows = []
    for u, g in enumerate(genome):
        for m in range(M):
            on = rng.random(sites) < (0.97 if prof[g, m] > 0.5 else 0.01)
            rows.append(pl.DataFrame({"unit": u, "group": m, "n": reads, "k": rng.binomial(reads, np.where(on, 0.9, 0.03))}))
    df = pl.concat(rows).with_columns(pl.col("unit", "group", "n", "k").cast(pl.Int64))
    grid = Grid(64)
    return unit_loglik(df, len(genome), M, det, grid).astype(np.float32), chunk_bin, n_other + 1, ["a"] * M, grid, genome


def test_alternating_robust_fit_reports_a_bin_that_did_not_settle():
    L, cb, H, mt, grid, genome = two_genome_bin(0.6)
    old = FittedModel(L, cb, H, mt, grid, float("inf"))
    # the 2-cycle: no chunk of the bin is down-weighted, and the fit reports that it did not settle
    assert (1 - old.w[genome == 1]).mean() < 0.5
    assert old.robust_unconverged()[0] and not old.robust_unconverged()[1:].any()
    light = FittedModel(*two_genome_bin(0.95)[:5], float("inf"))
    assert not light.robust_unconverged().any()


def test_flag_not_withdrawn_when_the_chunks_say_member():
    post, _, _ = regional_pattern([np.full(6, -10.0)])  # every chunk fits the bin
    assert post[0].argmax() == 0
    assert withdraw_as_mixed(post, "vs_foreign")[0]      # the 0.10.x rule: mixed > foreign is enough
    assert not withdraw_as_mixed(post, "largest")[0]
    post, _, _ = regional_pattern([np.r_[np.full(4, -10.0), np.full(4, 10.0)], np.full(8, -10.0), np.full(8, 10.0)])
    assert withdraw_as_mixed(post, "largest").tolist() == [True, False, False]


def test_flagged_share_per_bin():
    given = np.array([0, 0, 0, 1, 1, -1])
    flagged = np.array([True, True, False, True, False, False])
    judged = np.array([True, True, True, True, True, False])
    share = flagged_share_per_bin(given, flagged, judged, np.array([30, 30, 40, 10, 90, 5]), 2)
    assert np.allclose(share, [0.6, 0.1])


def test_contamination_rate_counts_only_contigs_that_pass_the_gate():
    # 100 binned contigs strongly against their bin by likelihood, 10 of them with a methylome difference
    N, H = 100, 3
    S = np.zeros((N, H)); S[:, 0] = -20.0
    Su = np.full(N, -30.0)
    given = np.zeros(N, int)
    diff = np.r_[np.full(10, 0.9), np.full(90, 0.1)]
    kw = dict(given=given, testable=np.ones(N, bool), temperature=1.0, methylome_difference=diff, p_unknown=0.5)
    a = refine(S, Su, **kw)
    b = refine(S, Su, eps_gated=True, **kw)
    assert a.eps > 0.8 and 0.05 < b.eps < 0.15
    assert a.contaminant.sum() == b.contaminant.sum() == 10


class _Asm:
    def __init__(self, seqs):
        self.seqs = seqs

    def seq(self, i):
        return self.seqs[i]


def test_redundancy_among_inclusions():
    rng = np.random.default_rng(0)
    member, new = rng.integers(0, 4, 20_000).astype(np.uint8), rng.integers(0, 4, 20_000).astype(np.uint8)
    asm = _Asm([member, new, new.copy(), member.copy()])
    cands = [(1, 0), (2, 0), (3, 0)]  # two copies of a new sequence, and a copy of the member
    indep = inclusion_redundancy(asm, cands, {0: [0]})
    assert indep[0] < 0.05 and indep[1] < 0.05 and indep[2] > 0.95
    seq = inclusion_redundancy(asm, cands, {0: [0]}, priority=np.array([0.99, 0.95, 0.9]), max_redundancy=0.3)
    assert seq[0] < 0.05 and seq[1] > 0.95 and seq[2] > 0.95


def test_two_state_share_sees_a_second_methylome():
    L, cb, H, mt, grid, genome = two_genome_bin(0.7)
    share, group = FittedModel(L, cb, H, mt, grid, float("inf")).two_state_share()
    assert 0.2 < share[0] < 0.4 and group[0] in (0, 1, 2, 3)   # 30% of the bin's chunks are the other genome
    assert (share[1:] < 0.05).all()


def test_unbinned_state_frequencies():
    rng = np.random.default_rng(0)
    n, M = 400, 3
    truth = np.array([0.0, 0.3, 0.02])                       # share of unbinned contigs methylated per group
    state = rng.random((n, M)) < truth
    on = np.where(state, 0.0, -30.0); off = np.where(state, -30.0, 0.0)
    on[:100, 2] = off[:100, 2] = 0.0                         # contigs without sites in group 2 do not count
    c = unbinned_state_frequencies(off, on, q=np.array([0.2, 0.2, 0.2]), strength=5.0)
    assert c[0] < 0.01 and abs(c[1] - state[:, 1].mean()) < 0.01 and abs(c[2] - state[100:, 2].mean()) < 0.01
    few = unbinned_state_frequencies(off[:2], on[:2], q=np.array([0.2, 0.2, 0.2]), strength=5.0)
    assert (np.abs(few - 0.2) < 0.15).all()                  # two contigs barely move the bins' frequency


def test_no_inclusion_into_a_bin_without_sites():
    # bin 1 has no site: every contig scores under it like a community draw; with p_unknown near 0 its prior
    # share alone passes the inclusion threshold
    N, H = 5, 2
    S = np.zeros((N, H)); S[:, 0] = -50.0; S[:, 1] = -10.0
    Su = np.full(N, -10.0)
    kw = dict(given=np.full(N, -1), testable=np.zeros(N, bool), temperature=1.0, p_unknown=0.001)
    assert (refine(S, Su, **kw).include_bin == 1).all()
    assert (refine(S, Su, empty_bins=np.array([False, True]), **kw).include_bin == -1).all()
