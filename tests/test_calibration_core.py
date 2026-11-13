"""Calibration core options (mission WP3; dev log proposals/2026-11-12/wp3_designs): each is off by default, and
switched on it changes only what it names."""
import json

import numpy as np
import pytest

from helpers import copy_cache, run_pipeline
from epirefine.model import FittedModel, npmle


def _params(run):
    return json.loads((run.out / "params.json").read_text())


@pytest.fixture(scope="module")
def tau_inf_run(community, base_run, tmp_path_factory):
    out = copy_cache(base_run, tmp_path_factory.mktemp("tau_inf") / "out")
    return run_pipeline(**community.args(out), cache_dir=str(out / "cache"), tau_select="inf")


def test_tau_select_inf_fits_one_tau_inside_the_calibration(tau_inf_run):
    rows = tau_inf_run.table("calibration/tau_fit.tsv")
    assert rows.height == 1 and [str(x) for x in rows["label"].to_list()] == ["inf"]
    taus = tau_inf_run.table("calibration/tau_groups.tsv")["tau"].to_numpy()
    assert np.isinf(taus).all()
    assert _params(tau_inf_run)["tau"] in ("inf", float("inf"))


def test_tau_select_rejects_an_unknown_value(community, base_run, tmp_path):
    with pytest.raises(ValueError, match="tau_select"):
        run_pipeline(**community.args(tmp_path / "out"), cache_dir=str(base_run.out / "cache"), tau_select="common")


def test_cold_prior_start_ignores_the_warm_start():
    """A warm start that put no mass on the cells the data need cannot be revived by the EM; a cold fit can."""
    rng = np.random.default_rng(0)
    G, B, M = 16, 40, 2
    ell = np.full((B, M, G), -50.0)
    ell[:, :, 12] = 0.0                       # every bin sits at grid cell 12
    ell += rng.normal(0, 0.01, ell.shape)
    bad = np.full((M, G), 1e-300); bad[:, 2] = 1.0; bad /= bad.sum(1, keepdims=True)
    warm = npmle(ell, init=bad)
    cold = npmle(ell)
    assert warm[:, 12].max() < 1e-3 < cold[:, 12].min()


def test_fitted_model_prior_start_is_validated():
    with pytest.raises(ValueError, match="prior_start"):
        FittedModel(np.zeros((1, 1, 4), np.float32), np.zeros(1, int), 1, ["a"], None, float("inf"), prior_start="x")


def test_exclusion_off_keeps_every_fragment(community, base_run, tmp_path):
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), calibration_exclusion=False)
    off = run.table("calibration/fragments.tsv")["foreign_excluded"].to_numpy()
    on = base_run.table("calibration/fragments.tsv")["foreign_excluded"].to_numpy()
    # only fragments without an independent profile (their contig is the whole bin) are still left out
    assert len(off) == len(on) and not (off & ~on).any() and off.sum() < on.sum()


# ---------------------------------------------------------------------------------------- the calibration function
from epirefine.calibrate import (ContigScores, CoreOptions, SitesLaw, calibrate_decisions, fit_sites_law,  # noqa: E402
                                 piece_losses)
from epirefine.refine import refine  # noqa: E402
from test_unknown_choice import _fragments  # noqa: E402


def _contigs(H=6, n=60, seed=1):
    rng = np.random.default_rng(seed)
    S = rng.normal(-12.0, 2.0, (n, H))
    given = np.where(np.arange(n) < n // 2, np.arange(n) % H, -1)
    best = np.where(given >= 0, given, rng.integers(0, H, n))
    S[np.arange(n), best] = rng.normal(0.0, 1.0, n)
    Su = rng.normal(-6.0, 2.0, n)
    return ContigScores(S=S, Su_comm=Su, R={0.05: S - 0.3}, given=given, testable=given >= 0,
                        decidable=np.ones(n, bool), n_sites=rng.uniform(200, 30000, n))


def test_piece_losses_equal_the_fragment_fit_at_one_temperature():
    ff = _fragments()
    for T, eta, lam in ((1.3, 0.05, 1.0), (4.0, 0.05, 0.5)):
        a0, b0 = ff.losses(T, eta, lam, 0.1)
        a1, b1 = piece_losses(ff, T, eta, lam, 0.1)
        assert np.allclose(a0, a1, rtol=0, atol=1e-12) and np.allclose(b0, b1, rtol=0, atol=1e-12)


def test_sites_law_is_floored_and_monotone():
    law = SitesLaw(a=np.log(3.0), b=0.33)
    T = law([10, 300, 1000, 3000, 1e5])
    assert T[0] == pytest.approx(1.082) and np.all(np.diff(T) >= 0) and law(1000) == pytest.approx(3.0)


def test_sites_law_fit_finds_no_slope_when_pieces_have_one_length():
    ff = _fragments()
    law = fit_sites_law(ff, 0.05, 1.0, ff.priors_for(0.1), 2.0, log=lambda *a: None)
    best_grid = min(ff.temps, key=lambda T: ff.objective(T, 0.05, 1.0, ff.priors_for(0.1)) if T >= 1 else np.inf)
    # one site count: b is not identified; the law's temperature must do at least as well as the grid's
    T = float(law(300))
    assert ff.objective(T, 0.05, 1.0, ff.priors_for(0.1)) <= ff.objective(best_grid, 0.05, 1.0, ff.priors_for(0.1)) + 1e-6


def test_default_core_is_the_refine_call_of_0_11():
    c = _contigs()
    fit = {"temperature": 2.0, "eta": 0.05, "lam": 1.0, "curve": None}
    rules = dict(contamination_threshold=0.5, include_threshold=0.9)
    res = calibrate_decisions(None, c, rules, CoreOptions(), fit=fit, log=lambda *a: None)
    ref = refine(c.S, c.Su_comm, c.given, c.testable, 2.0, 0.5, 0.9, decidable=c.decidable)
    assert res.p_unknown == ref.p_unknown and np.array_equal(res.refinement.include_bin, ref.include_bin)
    assert np.array_equal(res.refinement.p_foreign, ref.p_foreign, equal_nan=True)


def test_unknown_share_t1_does_not_move_with_the_temperature():
    c = _contigs()
    rules = dict(contamination_threshold=0.5, include_threshold=0.9)
    ps = [calibrate_decisions(None, c, rules, CoreOptions(unknown_share="t1"), log=lambda *a: None,
                              fit={"temperature": T, "eta": 0.05, "lam": 1.0, "curve": None}).p_unknown for T in (1.0, 3.0, 8.0)]
    dep = [calibrate_decisions(None, c, rules, CoreOptions(), log=lambda *a: None,
                               fit={"temperature": T, "eta": 0.05, "lam": 1.0, "curve": None}).p_unknown for T in (1.0, 3.0, 8.0)]
    assert max(ps) - min(ps) < 1e-12
    assert dep[2] > dep[0] + 0.05   # the deployed EM drifts with T through the untempered 1/H prior
    assert ps[0] == pytest.approx(dep[0], abs=1e-9)


def test_core_options_are_validated():
    with pytest.raises(ValueError, match="temperature_law"):
        CoreOptions(temperature_law="power")


@pytest.mark.parametrize("opts", [dict(unknown_share="t1"), dict(temperature_law="sites", evidence_temperature="contig")])
def test_core_options_run_end_to_end(community, base_run, tmp_path, opts):
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), **opts)
    par = _params(run)
    assert run.table("contig_decisions.tsv").height == base_run.table("contig_decisions.tsv").height
    if "unknown_share" in opts:
        assert par["p_unknown_t1"] == pytest.approx(par["p_unknown"])
    else:
        assert par["temperature_by_sites"][0]["law"] == "sites" and "temperature law" in run.log


# ---------------------------------------------------------------------------------------- per-bin relatives weights
from epirefine.calibrate import FragmentFit, RelativesWeights, fit_relatives_weights  # noqa: E402


def _sibling_pieces(H=20, per_bin=15, siblings=((0, 1), (1, 0)), seed=3):
    """Test-B pieces: a piece fits the community draw better than any relative, except the pieces of bins with a
    sibling, which fit the sibling's relative far better."""
    rng = np.random.default_rng(seed)
    n = H * per_bin
    fbin = np.repeat(np.arange(H), per_bin)
    S = rng.normal(-30.0, 1.0, (n, H))
    S[np.arange(n), fbin] = 0.0
    R = S - 2.0
    Su = np.full(n, -10.0)
    for a, b in siblings:   # the sibling bin fits a's pieces almost as well, its unbinned relative better still
        S[fbin == a, b] = -3.0
        R[fbin == a, b] = -1.0
    ff = FragmentFit(S=S, fbin=fbin, Su_comm=Su, R={0.05: R}, kept=np.ones(n, bool), distinct=np.ones(n, bool),
                     length=np.full(n, 5000), n_sites=np.full(n, 300.0), difference=np.zeros(n), contig=np.arange(n))
    return ff


def test_relatives_weights_concentrate_on_sibling_bins():
    ff = _sibling_pieces()
    w = fit_relatives_weights(ff, log=lambda *a: None)
    H = len(w.w)
    assert w.lam + w.w.sum() == pytest.approx(1.0)
    assert set(np.argsort(w.w)[-2:]) == {0, 1}           # the two sibling bins
    assert w.summary()["share_of_relatives_on_top10_bins"] > 0.9
    assert (H * w.w[2:] < 0.05).all()
    assert 0.8 < w.lam < 0.95                            # 2 of 20 bins' pieces are relatives' pieces


def test_community_only_removes_the_switch():
    """--relatives community-only: where the switch picks relatives, the fit becomes the community-only row the 2-SE
    rule falls back to; the decisions are those of that fit (lam = 1)."""
    ff = _sibling_pieces()
    fit0 = ff.fit(0.1, log=lambda *a: None)
    assert fit0["lam"] < 1                                   # the switch picks relatives on these pieces
    rows = [r for r in fit0["temperature_fit"] if r["lam"] >= 1 and r["temperature"] >= 1.0]
    best = min(rows, key=lambda r: r["log_loss"])
    c = _contigs(H=ff.H)
    rules = dict(contamination_threshold=0.5, include_threshold=0.9)
    opts = CoreOptions(relatives="community-only", unknown_share="t1")
    res = calibrate_decisions(ff, c, rules, opts, log=lambda *a: None)
    assert res.fit["lam"] == 1.0 and res.fit["temperature"] == best["temperature"]
    assert res.fit["switch_choice"]["lam"] == fit0["lam"]
    same = calibrate_decisions(None, c, rules, CoreOptions(unknown_share="t1"), log=lambda *a: None,
                               fit={"temperature": best["temperature"], "eta": best["eta"], "lam": 1.0, "curve": None})
    assert res.p_unknown == same.p_unknown and np.array_equal(res.refinement.include_bin, same.refinement.include_bin)
    with pytest.raises(ValueError, match="community-only"):
        calibrate_decisions(ff, c, rules, opts, fit=dict(fit0), log=lambda *a: None)


def test_community_only_runs_end_to_end(community, base_run, tmp_path, monkeypatch):
    """On the test community the switch chooses community only: forcing it changes no decision. With a calibration
    made to report relatives, the pipeline replaces them by the community-only row and decides as before."""
    import epirefine.pipeline as P
    out2 = copy_cache(base_run, tmp_path / "out2")
    ref = run_pipeline(**community.args(out2), cache_dir=str(out2 / "cache"), unknown_share="t1")
    assert _params(ref)["unknown_lam"] == 1.0
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), relatives="community-only",
                       unknown_share="t1")
    par = _params(run)
    assert par["relatives"] == "community-only" and par["unknown_lam"] == 1.0 and "unknown_model_forced" not in par
    assert run.table("contig_decisions.tsv").equals(ref.table("contig_decisions.tsv"))

    real = P.calibrate

    def relatives_chosen(*a, **k):   # as if the switch had picked relatives at another temperature
        cal = real(*a, **k)
        T = cal["temperature"] * 3
        cal["fit"] = dict(cal["fit"], eta=0.05, lam=0.1, temperature=T)
        cal.update(eta=0.05, lam=0.1, temperature=T)
        return cal
    monkeypatch.setattr(P, "calibrate", relatives_chosen)
    out3 = copy_cache(base_run, tmp_path / "out3")
    forced = run_pipeline(**community.args(out3), cache_dir=str(out3 / "cache"), relatives="community-only",
                          unknown_share="t1")
    par3 = _params(forced)
    assert par3["unknown_model_forced"]["switch_lam"] == 0.1 and par3["unknown_lam"] == 1.0
    assert "unknown model forced to community only" in forced.log
    assert forced.table("contig_decisions.tsv").equals(ref.table("contig_decisions.tsv"))


def test_relatives_logit_renormalises_without_the_own_bin():
    w = RelativesWeights(lam=0.5, w=np.array([0.25, 0.25]), alpha=1.0)
    R = np.array([[0.0, 0.0]]); Su = np.array([0.0])
    assert w.logit(R, Su, 1.0)[0] == pytest.approx(0.0)
    assert w.logit(R, Su, 1.0, exclude=np.array([0]))[0] == pytest.approx(0.0)


def test_per_bin_relatives_run_end_to_end(community, base_run, tmp_path):
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), relatives="per-bin", unknown_share="t1")
    hr = run.table("hidden_relative.tsv")
    assert hr.height == run.table("contig_decisions.tsv").height
    p = hr["p_hidden_relative"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    assert "relatives per bin" in run.log and "relatives_per_bin" in _params(run)


def test_core_temper_then_mix_uses_the_coherent_unknown_term():
    from epirefine.model import unknown_logit
    c = _contigs()
    fit = {"temperature": 3.0, "eta": 0.05, "lam": 0.5, "curve": None}
    rules = dict(contamination_threshold=0.5, include_threshold=0.9)
    res = calibrate_decisions(None, c, rules, CoreOptions(temper_then_mix=True), fit=fit, log=lambda *a: None)
    zu = unknown_logit(c.R[0.05], c.Su_comm, 0.05, 0.5, 3.0, temper_then_mix=True)
    ref = refine(c.S, None, c.given, c.testable, 3.0, 0.5, 0.9, decidable=c.decidable, Zu=zu)
    assert res.p_unknown == ref.p_unknown and np.array_equal(res.refinement.include_bin, ref.include_bin)


def test_core_options_with_the_temper_then_mix_feature_run(community, base_run, tmp_path):
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), unknown_share="t1",
                       calibration="default,temper-then-mix")
    assert "unknown share by the EM at T = 1" in run.log


# ---------------------------------------------------------------------------------------- slot model (candidate 3)
from epirefine.calibrate import SlotModel, fit_slot_model, slot_decide, slot_piece_terms  # noqa: E402


def test_slot_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    n, H = 40, 7
    S = rng.normal(-5, 3, (n, H)); R = S + rng.normal(-0.5, 0.5, (n, H)); Sc = rng.normal(-6, 2, n)
    fb = rng.integers(0, H, n); dist = rng.random(n) < 0.7
    T = rng.uniform(1.1, 4.0, n); u = rng.normal(-2, 1, H)
    la, lb, g = slot_piece_terms(S, R, Sc, fb, dist, T, u, 0.1)
    f = lambda v: (lambda a, b, _: a.mean() + b.mean())(*slot_piece_terms(S, R, Sc, fb, dist, T, v, 0.1, need_grad=False))
    num = np.array([(f(u + 1e-6 * np.eye(H)[h]) - f(u - 1e-6 * np.eye(H)[h])) / 2e-6 for h in range(H)])
    assert np.allclose(g, num, atol=1e-6)


def test_slot_unknown_share_does_not_see_the_bin_relative_split():
    """Design C's test R0: members of bins whose relative fits almost as well (R = S + log 0.8). The EM over 'community'
    against 'some slot' gives the same p_c whatever pi; the 0.11 parametrisation (relatives inside the unknown share)
    runs to p_u near 1 at lam 0.5."""
    from epirefine.refine import em_weight
    from epirefine.model import unknown_score
    from scipy.special import logsumexp as lse
    rng = np.random.default_rng(0)
    H, n = 490, 4000
    best = rng.integers(0, H, n)
    S = rng.normal(-40, 3, (n, H)); S[np.arange(n), best] = 0.0
    R = S + np.log(0.8)
    comm = rng.random(n) < 0.4
    Sc = np.where(comm, 5.0, -15.0)    # 40% fit the community draw better than any slot
    c = ContigScores(S=S, Su_comm=Sc, R={0.05: R}, given=np.full(n, -1), testable=np.zeros(n, bool),
                     decidable=np.ones(n, bool), n_sites=np.full(n, 1000.0))
    ps = []
    for pi in (0.0, 0.5, 0.9):
        m = SlotModel(pi=np.full(H, max(pi, 1e-9)), pi0=pi, kappa=np.inf, law=None, T=1.0)
        _, p_c, _, _ = slot_decide(c, m, {}, 0.5, 0.9, log=lambda *a: None)
        ps.append(p_c)
    assert max(abs(p - 0.4) for p in ps) < 0.02
    # the 0.11 parametrisation follows p_u = min(1, f_c / (1 - r)), r = (1 - lam) c (calibration_coherence R1.1)
    old = lambda lam: em_weight(lse(S, axis=1) - np.log(H), unknown_score(R, Sc, 0.05, lam))[0]
    assert old(0.5) == pytest.approx(0.4 / (1 - 0.5 * 0.8), abs=0.05)
    assert old(0.1) >= 0.99


def test_slot_model_fit_raises_pi_only_for_sibling_bins():
    ff = _sibling_pieces()
    m = fit_slot_model(ff, law_mode=False, kappa=10.0, log=lambda *a: None)
    assert set(np.argsort(m.pi)[-2:]) == {0, 1}
    assert min(m.pi[0], m.pi[1]) > 2 * np.median(m.pi)


def test_slot_averaged_objective_gradient_matches_finite_differences():
    from epirefine.calibrate import PRIOR_GRID, _slot_objective
    rng = np.random.default_rng(6)
    n, H = 30, 6
    S = rng.normal(-5, 3, (n, H)); R = S + rng.normal(-0.5, 0.5, (n, H)); Sc = rng.normal(-6, 2, n)
    arr = (S, R, Sc, rng.integers(0, H, n), rng.random(n) < 0.6)
    pri = tuple((p, (1 - p) / 5, p / 5) for p in PRIOR_GRID)
    u = rng.normal(-1.5, 1, H)
    _, g = _slot_objective(arr, 2.0, u, pri)
    f = lambda v: _slot_objective(arr, 2.0, v, pri, need_grad=False)[0]
    num = np.array([(f(u + 1e-6 * np.eye(H)[h]) - f(u - 1e-6 * np.eye(H)[h])) / 2e-6 for h in range(H)])
    assert np.allclose(g, num, atol=1e-6)


def test_slot_relatives_run_end_to_end(community, base_run, tmp_path):
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), relatives="slot", unknown_share="t1",
                       temperature_law="sites")
    assert "slot model" in run.log and "relatives_slot" in _params(run)
    p = run.table("hidden_relative.tsv")["p_hidden_relative"].to_numpy()
    assert np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError, match="unknown-share t1"):
        run_pipeline(**community.args(tmp_path / "o2"), cache_dir=str(out / "cache"), relatives="slot")


def test_core_options_are_recorded_and_rerun(community, base_run, tmp_path):
    """Every WP3 core option away from its default, with a calibration seed, comes back from params.json: the CLI
    rerun repeats the pieces and the decisions (the WP1 run record covers the core options)."""
    from epirefine.cli import main
    from helpers import DECISION_FILES, assert_same_outputs, logged
    first = copy_cache(base_run, tmp_path / "first")
    core = {"tau_select": "inf", "npmle_start": "cold", "unknown_share": "t1", "temperature_law": "sites",
            "evidence_temperature": "contig", "relatives": "slot", "slot_objective": "averaged", "slot_test_b": "all",
            "slot_kappa": 2.0}
    opts = [x for k, v in core.items() for x in ("--" + k.replace("_", "-"), v)]
    logged(main, community.cli_args(first, "--workers", 1, "--cache-dir", first / "cache", "--calibration-seed", 2,
                                    "--calibration-exclusion", "off", *opts))
    rec = json.loads((first / "params.json").read_text())
    assert {k: rec[k] for k in core} == core and rec["calibration_exclusion"] is False and rec["calibration_seed"] == 2
    log = logged(main, ["rerun", "--params", str(first / "params.json"), "--out", str(tmp_path / "again")])
    assert "reusing site cache" in log and "slot model" in log
    assert_same_outputs(first, tmp_path / "again",
                        DECISION_FILES + ["calibration/fragments.tsv", "hidden_relative.tsv",
                                          "calibration/relatives_by_bin.tsv"])
    again = json.loads((tmp_path / "again" / "params.json").read_text())
    assert {k: again[k] for k in core} == core and again["calibration_exclusion"] is False
