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
    for a, b in siblings:
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
