"""Calibration core options (mission WP3; dev log proposals/2026-11-12/wp3_designs): switched on, each changes only
what it names. Since 0.12 --relatives community-only and --unknown-share t1 are the default (option E,
tests/test_option_e.py); the temperature law is off by default. The designs that lost (--tau-select, --npmle-start,
--relatives per-bin and slot, --calibration-exclusion) were removed in 0.12 (U12)."""
import json

import numpy as np
import pytest

from helpers import copy_cache, run_pipeline
from epirefine.model import npmle


def _params(run):
    return json.loads((run.out / "params.json").read_text())


def test_npmle_cannot_revive_cells_a_warm_start_emptied():
    """A warm start that put no mass on the cells the data need cannot be revived by the EM; a uniform start can
    (why model_diagnostics.md section 1 found warm-started fits off their optimum; --npmle-start cold, removed in
    0.12, changed no decision that counted)."""
    rng = np.random.default_rng(0)
    G, B, M = 16, 40, 2
    ell = np.full((B, M, G), -50.0)
    ell[:, :, 12] = 0.0                       # every bin sits at grid cell 12
    ell += rng.normal(0, 0.01, ell.shape)
    bad = np.full((M, G), 1e-300); bad[:, 2] = 1.0; bad /= bad.sum(1, keepdims=True)
    warm = npmle(ell, init=bad)
    cold = npmle(ell)
    assert warm[:, 12].max() < 1e-3 < cold[:, 12].min()


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


# ---------------------------------------------------------------------------------------- the switch removed
from epirefine.calibrate import FragmentFit  # noqa: E402


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


def test_core_options_are_recorded_and_rerun(community, base_run, tmp_path):
    """Every core option away from its default, with a calibration seed, comes back from params.json: the CLI
    rerun repeats the pieces and the decisions (the WP1 run record covers the core options)."""
    from epirefine.cli import main
    from helpers import DECISION_FILES, assert_same_outputs, logged
    first = copy_cache(base_run, tmp_path / "first")
    core = {"unknown_share": "deployed", "temperature_law": "sites", "evidence_temperature": "contig",
            "relatives": "community-only"}
    opts = [x for k, v in core.items() for x in ("--" + k.replace("_", "-"), v)]
    logged(main, community.cli_args(first, "--workers", 1, "--cache-dir", first / "cache", "--calibration-seed", 2, *opts))
    rec = json.loads((first / "params.json").read_text())
    assert {k: rec[k] for k in core} == core and rec["calibration_seed"] == 2
    log = logged(main, ["rerun", "--params", str(first / "params.json"), "--out", str(tmp_path / "again")])
    assert "reusing site cache" in log and "temperature law" in log
    assert_same_outputs(first, tmp_path / "again", DECISION_FILES + ["calibration/fragments.tsv"])
    again = json.loads((tmp_path / "again" / "params.json").read_text())
    assert {k: again[k] for k in core} == core
