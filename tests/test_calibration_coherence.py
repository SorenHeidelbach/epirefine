"""Calibration coherence (dev log proposal 2026-11-10): order of tempering and mixing, the prior the
calibration is fitted under, tau compared on common fragments, the EM priors and the false-flag diagnostic."""
import numpy as np
import pytest
from scipy.special import logsumexp

from epirefine.calibrate import (COHERENT, FEATURES, FragmentFit, TemperatureCurve, isotonic_increasing,
                                 loglik_on_common, parse_features)
from epirefine.model import posterior, unknown_logit, unknown_score
from epirefine.refine import em_weight, refine


def _perfect_fit(H, M=20, eta=0.05, gap=15.0):
    """A contig that fits bin 0 in every group and no other bin; 1 / H of genomes carry each level."""
    S = np.full((1, H), -gap * M)
    S[0, 0] = 0.0
    Sc = np.array([-np.log(H) * M])
    R = np.array([[M * np.logaddexp(np.log1p(-eta) + (0.0 if h == 0 else -gap), np.log(eta) - np.log(H)) for h in range(H)]])
    return S, Sc, R


def test_unknown_gains_nothing_from_the_number_of_bins_when_tempered_before_mixing():
    eta, lam, T = 0.05, 0.5, 5.0
    odds = {}
    for H in (7, 490):
        S, Sc, R = _perfect_fit(H, eta=eta)
        for order in (False, True):
            zu = unknown_logit(R, Sc, eta, lam, T, temper_then_mix=order)[0]
            # log odds unknown : bin 0 at p_unknown = 0.5 (bin prior 1 / H)
            odds[H, order] = zu - (S[0, 0] / T - np.log(H))
    # mixing first: the weight 1 / H of 'relative of bin 0' is tempered too, log(H) (1 - 1 / T) in favour of unknown
    assert odds[490, False] - odds[7, False] == pytest.approx(np.log(490 / 7) * (1 - 1 / T), abs=0.05)
    assert odds[490, False] > 3          # unknown wins although the contig fits its bin perfectly
    # tempering first: the same odds whatever the number of bins, and the bin wins
    assert odds[490, True] == pytest.approx(odds[7, True], abs=0.05)
    assert odds[490, True] < 0


def test_both_orders_agree_without_tempering_and_without_relatives():
    S, Sc, R = _perfect_fit(30)
    for T, lam in ((1.0, 0.5), (3.0, 1.0)):
        a = unknown_logit(R, Sc, 0.05, lam, T, temper_then_mix=True)
        b = unknown_logit(R, Sc, 0.05, lam, T, temper_then_mix=False)
        assert a == pytest.approx(b)
    assert unknown_logit(R, Sc, 0.05, 0.5, 2.0, temper_then_mix=False) == pytest.approx(unknown_score(R, Sc, 0.05, 0.5) / 2.0)


def test_unknown_logit_excludes_a_bin_and_takes_a_temperature_per_unit():
    rng = np.random.default_rng(0)
    R, Sc = rng.normal(-20, 5, (4, 6)), rng.normal(-25, 5, 4)
    T = np.array([1.0, 2.0, 3.0, 4.0])
    z = unknown_logit(R, Sc, 0.05, 0.3, T, exclude=np.array([0, -1, 2, 5]))
    for i, (t, ex) in enumerate(zip(T, (0, -1, 2, 5))):
        r = np.delete(R[i], ex) if ex >= 0 else R[i]
        want = np.logaddexp(np.log(0.3) + Sc[i] / t, np.log(0.7) + logsumexp(r / t) - np.log(len(r)))
        assert z[i] == pytest.approx(want)


def test_posterior_with_a_tempered_unknown_score_matches_the_scalar_form():
    rng = np.random.default_rng(1)
    S, Su = rng.normal(-30, 6, (5, 4)), rng.normal(-30, 6, 5)
    assert posterior(S, None, 0.3, 2.0, Zu=Su / 2.0) == pytest.approx(posterior(S, Su, 0.3, 2.0))
    T = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
    P = posterior(S, None, 0.3, T, Zu=Su / T)
    for i in range(5):
        assert P[i] == pytest.approx(posterior(S[i:i + 1], Su[i:i + 1], 0.3, float(T[i]))[0])


def test_p_unknown_prior_keeps_one_contig_from_deciding_the_prior():
    # one unbinned contig that prefers 'unknown' by 0.1 nats: flat prior runs to the clip, Beta(2, 2) stays near 0.5
    flat, _ = em_weight(np.array([0.0]), np.array([0.1]), 1.0, 1.0)
    weak, _ = em_weight(np.array([0.0]), np.array([0.1]), 2.0, 2.0)
    assert flat > 0.999
    assert 0.4 < weak < 0.6
    # hundreds of contigs: the prior does not matter
    rng = np.random.default_rng(0)
    d = np.where(rng.random(400) < 0.4, 8.0, -8.0)
    a, _ = em_weight(np.zeros(400), d, 1.0, 1.0)
    b, _ = em_weight(np.zeros(400), d, 2.0, 2.0)
    assert abs(a - b) < 0.002


def test_contamination_rate_counts_only_contigs_that_pass_the_gate():
    # 100 binned contigs in 2 bins: 10 real contaminants (large difference), 20 discordant members whose
    # likelihood says foreign but whose methylome difference is below the gate, 70 clean members
    N, H = 100, 2
    S = np.zeros((N, H))
    S[:, 1] = -30.0                      # own bin (0) fits, the other does not
    Su = np.full(N, -30.0)
    S[:30, 0] = -40.0                    # contaminants and discordant members: own bin is bad
    S[:10, 1] = 0.0                      # contaminants: the other bin fits
    Su[10:30] = -10.0                    # discordant members: "unknown" fits better than the bin
    given = np.zeros(N, int)
    diff = np.r_[np.full(10, 0.9), np.full(20, 0.1), np.zeros(70)]
    kw = dict(given=given, testable=np.ones(N, bool), temperature=1.0, methylome_difference=diff, p_unknown=0.5)
    before = refine(S, Su, **kw)
    after = refine(S, Su, eps_after_gate=True, **kw)
    assert before.contaminant.sum() == after.contaminant.sum() == 10
    assert before.eps == pytest.approx(30 / 118, abs=0.01)   # MAP with Beta(1, 19): (30 + 0) / (100 + 18)
    assert after.eps == pytest.approx(10 / 118, abs=0.01)
    # the reported probabilities stay likelihood-based for every contig
    assert (after.p_foreign[10:30] > 0.99).all()


def test_taus_are_compared_on_the_fragments_every_fit_keeps():
    # fit 1 drops its worst fragment and looks better on its own kept set; on the common set fit 0 wins
    own = [np.array([-1.0, -1.0, -50.0]), np.array([-2.0, -2.0, -60.0])]
    keeps = [np.array([True, True, True]), np.array([True, True, False])]
    assert own[1][keeps[1]].sum() > own[0][keeps[0]].sum()
    common, lls = loglik_on_common(own, keeps)
    assert common.tolist() == [True, True, False]
    assert lls[0] > lls[1]


def test_isotonic_and_temperature_curve():
    assert isotonic_increasing(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4)).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert isotonic_increasing(np.array([3.0, 1.0]), np.array([1.0, 3.0])).tolist() == [1.5, 1.5]
    c = TemperatureCurve(np.log([100.0, 1000.0]), np.log([1.0, 2.0]))
    assert c([10, 100, 1000, 1e6]) == pytest.approx([1.0, 1.0, 2.0, 2.0])   # constant outside the calibrated range
    assert 1.0 < c([316])[0] < 2.0


def test_parse_features():
    assert parse_features("legacy") == frozenset()
    assert parse_features("coherent") == frozenset(COHERENT)
    assert parse_features("coherent,-common-tau") == frozenset(COHERENT) - {"common-tau"}
    assert parse_features("temper-then-mix, em-priors") == {"temper-then-mix", "em-priors"}
    assert set(COHERENT) <= set(FEATURES)
    with pytest.raises(ValueError):
        parse_features("nonsense")
    with pytest.raises(ValueError):
        parse_features("prior-averaged,prior-matched")


def _gaussian_fragments(rng, H=20, d=4, s2=4.0, n=4000):
    """Exact likelihoods, so the true temperature is 1: fragments of bin h see x ~ N(mu_h, I), bins'
    mu_h ~ N(0, s2 I), and a genome without a bin is one more draw: x ~ N(0, (1 + s2) I)."""
    mu = rng.normal(0, np.sqrt(s2), (H, d))
    fb = rng.integers(0, H, n)
    x = mu[fb] + rng.normal(0, 1, (n, d))
    S = -0.5 * ((x[:, None, :] - mu[None]) ** 2).sum(-1) - 0.5 * d * np.log(2 * np.pi)
    Sc = -0.5 * (x ** 2).sum(-1) / (1 + s2) - 0.5 * d * np.log(2 * np.pi * (1 + s2))
    return S, fb, Sc


def _fit(S, fb, Sc, features, **kw):
    n = len(fb)
    base = dict(S=S, fbin=fb, Su_comm=Sc, R={}, kept=np.ones(n, bool), distinct=np.ones(n, bool),
                length=np.full(n, 1000), n_sites=np.full(n, 10.0), difference=np.zeros(n),
                features=frozenset(features), temps=tuple(np.round(np.exp(np.linspace(np.log(0.3), np.log(5), 41)), 3)),
                allow_sharpening=True)
    base.update(kw)
    return FragmentFit(**base)


def test_fragment_losses_equal_the_posterior_based_ones():
    S, fb, Sc = _gaussian_fragments(np.random.default_rng(3), H=6, n=200)
    ff = _fit(S, fb, Sc, {"temper-then-mix"})
    la, lb = ff.losses(1.7, None, 1.0, 0.3)
    Pa = posterior(S, Sc, 0.3, 1.7)
    assert la == pytest.approx(-np.log(Pa[np.arange(200), fb]))
    # host absent: the remaining H - 1 bins share 1 - p
    Z = S / 1.7
    Z[np.arange(200), fb] = -np.inf
    lg = np.c_[Z + np.log(0.7 / 5), Sc / 1.7 + np.log(0.3)]
    assert lb == pytest.approx(-(lg[:, -1] - logsumexp(lg, axis=1)))


def test_matched_prior_recovers_temperature_one_where_the_legacy_objective_does_not():
    S, fb, Sc = _gaussian_fragments(np.random.default_rng(0))
    ff = _fit(S, fb, Sc, {"temper-then-mix", "prior-averaged", "unit-temperature"})
    # the calibration of 0.10: posteriors at prior 0.1, tests A and B weighted 1 : 1
    legacy = min(ff.temps, key=lambda T: sum(x.mean() for x in ff.losses(T, None, 1.0, 0.1)))
    assert legacy > 1.4                          # inflated by the prior mismatch alone
    assert ff.fit(0.5, log=lambda *_: None)["fitted_temperature"] == pytest.approx(1.0, abs=0.07)
    matched = _fit(S, fb, Sc, {"temper-then-mix", "prior-matched", "unit-temperature"})
    for p in (0.1, 0.5, 0.9):
        assert matched.fit(p, log=lambda *_: None)["fitted_temperature"] == pytest.approx(1.0, abs=0.08)


def test_never_sharpen_clips_at_exactly_one_with_unit_temperature():
    S, fb, Sc = _gaussian_fragments(np.random.default_rng(1), n=1500)
    # under-confident scores (half the true log-likelihoods: the fitted temperature is 0.5)
    kw = dict(temps=(0.5, 0.928, 1.082, 2.0), allow_sharpening=False)
    assert _fit(0.5 * S, fb, 0.5 * Sc, {"prior-averaged"}, **kw).fit(0.5, log=lambda *_: None)["temperature"] == 1.082
    assert _fit(0.5 * S, fb, 0.5 * Sc, {"prior-averaged", "unit-temperature"}, **kw).fit(0.5, log=lambda *_: None)["temperature"] == 1.0


def test_false_flag_rate_of_member_fragments():
    S, fb, Sc = _gaussian_fragments(np.random.default_rng(2), H=6, n=600)
    diff = np.zeros(600)
    diff[:300] = 0.9
    ff = _fit(S, fb, Sc, {"temper-then-mix"}, difference=diff, length=np.r_[np.full(300, 2000), np.full(300, 50_000)])
    fit = {"temperature": 1.0, "eta": None, "lam": 1.0, "curve": None}
    rows = {(r["length"], r["fragments"]): r for r in ff.false_flags(fit, p_unknown=0.5, eps=0.05)}
    # members are rarely called foreign, and never flagged where the gate is not passed
    assert rows["50000", "all"]["false_flag_rate"] == 0.0
    assert rows["2000", "all"]["false_flag_rate"] == rows["2000", "all"]["rate_p_foreign_above_threshold"] < 0.1
    assert rows["all", "all"]["n"] == 600
    # a higher contamination prior can only raise it
    more = {(r["length"], r["fragments"]): r for r in ff.false_flags(fit, p_unknown=0.5, eps=0.5)}
    assert more["2000", "all"]["false_flag_rate"] >= rows["2000", "all"]["false_flag_rate"]
    rel = ff.deployed_reliability(fit, 0.5, 0.05)
    assert {r["test"] for r in rel} == {"A_any_candidate", "A_map", "B_unknown"}
