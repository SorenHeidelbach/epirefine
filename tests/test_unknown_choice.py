"""The unknown model checked at its deployed prior (calibration feature deployed-unknown, dev log proposal
2026-11-12): a model with relatives chosen on the fragments at the reference prior is kept only if it still beats
the community-only model when each is scored at the p_unknown its own EM finds on the unbinned contigs."""
import json

import numpy as np
import polars as pl
import pytest

from small_community import run_community
from epirefine.calibrate import FEATURES, FragmentFit, parse_features


def _fragments(H=6, per_bin=12, gap=8.0, seed=0):
    """Fragments that fit their own bin by `gap` nats over every other bin; three fragments per source contig."""
    rng = np.random.default_rng(seed)
    n = H * per_bin
    fbin = np.repeat(np.arange(H), per_bin)
    S = rng.normal(-gap, 1.0, (n, H))
    S[np.arange(n), fbin] = rng.normal(0.0, 0.5, n)
    Su = np.full(n, -gap / 2)
    R = {0.05: S - 0.3}  # a relative of h fits nearly as well as h itself
    return FragmentFit(S=S, fbin=fbin, Su_comm=Su, R=R, kept=np.ones(n, bool), distinct=np.ones(n, bool),
                       length=np.full(n, 5000), n_sites=np.full(n, 300.0), difference=np.zeros(n),
                       contig=np.arange(n) // 3)


def test_deployed_gain_is_zero_for_the_same_model_at_the_same_prior():
    ff = _fragments()
    gain, se = ff.deployed_gain((2.0, 0.05, 0.9), 0.4, (2.0, 0.05, 0.9), 0.4)
    assert gain == pytest.approx(0.0, abs=1e-12) and se == pytest.approx(0.0, abs=1e-12)


def test_a_model_deployed_at_p_unknown_one_loses_to_one_deployed_at_a_moderate_prior():
    ff = _fragments()
    # the same likelihoods; the relatives model's EM ran to 1 (the ZymoFecal collapse), community-only's to 0.6
    gain, se = ff.deployed_gain((2.0, 0.05, 0.9), 1.0, (2.0, 0.05, 1.0), 0.6)
    assert gain < -2 * se and gain < -5     # test A: no fragment can be given to its bin at p_unknown = 1
    # at a moderate deployed prior the relatives model is not rejected out of hand
    gain2, se2 = ff.deployed_gain((2.0, 0.05, 0.9), 0.6, (2.0, 0.05, 1.0), 0.6)
    assert gain2 > gain + 5


def test_deployed_gain_uses_the_contig_bootstrap():
    ff = _fragments()
    ff_frag = _fragments()
    ff_frag.contig = None   # every fragment its own contig: resampled one by one
    _, se_contig = ff.deployed_gain((2.0, 0.05, 0.9), 0.9, (3.0, 0.05, 1.0), 0.5)
    _, se_frag = ff_frag.deployed_gain((2.0, 0.05, 0.9), 0.9, (3.0, 0.05, 1.0), 0.5)
    assert se_contig > 0 and se_frag > 0 and se_contig != se_frag


def test_feature_is_parsed_and_not_in_the_named_sets():
    assert "deployed-unknown" in FEATURES
    assert parse_features("deployed-unknown") == frozenset({"deployed-unknown"})
    assert "deployed-unknown" not in parse_features("coherent") | parse_features("consistent")
    assert parse_features("default") == frozenset({"bin-parsimony"}) and parse_features("legacy") == frozenset()
    assert parse_features("default,-bin-parsimony") == frozenset()
    assert "bin-parsimony" not in parse_features("coherent") | parse_features("consistent")


def test_bin_parsimony_is_the_default_and_deployed_unknown_is_not(small_community, tmp_path, forced_relatives):
    run_community(small_community, tmp_path / "out", seed=3)
    p = json.loads((tmp_path / "out" / "params.json").read_text())
    assert p["calibration_features"] == ["bin-parsimony"] and "unknown_model_deployed_check" not in p


def test_clustered_se_counts_bins_not_fragments():
    """A gain carried by 2 of 40 bins: large against the fragment bootstrap, not against the bin bootstrap; a gain
    shared by every bin stays significant."""
    from epirefine.calibrate import clustered_se
    rng = np.random.default_rng(1)
    bins = np.repeat(np.arange(40), 25)
    few = np.where(bins < 2, 2.0, 0.0) + rng.normal(0, 0.05, len(bins))
    se_frag = np.std([few[rng.integers(0, len(few), len(few))].mean() for _ in range(300)])
    se_bin = clustered_se(few, few[:0], bins, bins[:0])
    assert few.mean() > 2 * se_frag and few.mean() < 2 * se_bin
    spread = 0.1 + rng.normal(0, 0.05, len(bins))
    assert spread.mean() > 2 * clustered_se(spread, spread[::2], bins, bins[::2])
    assert clustered_se(few, few[:0], bins, bins[:0]) == clustered_se(few, few[:0], bins, bins[:0])  # fixed seed


def _decisions(out):
    return pl.read_csv(out / "contig_decisions.tsv", separator="\t", infer_schema_length=0).sort("contig")


def test_without_relatives_the_feature_changes_nothing(small_community, tmp_path):
    """Calibration chooses the community-only model on the small community: decisions identical to legacy."""
    run_community(small_community, tmp_path / "legacy", seed=3, calibration="legacy")
    run_community(small_community, tmp_path / "dep", seed=3, calibration="deployed-unknown")
    assert _decisions(tmp_path / "legacy").equals(_decisions(tmp_path / "dep"))
    p = json.loads((tmp_path / "dep" / "params.json").read_text())
    assert p["unknown_lam"] == 1.0 and "unknown_model_deployed_check" not in p


@pytest.fixture
def forced_relatives(monkeypatch):
    """Calibration's choice replaced by a relatives model that collapses p_unknown: lam 0.1 at T = 20 (the relative's
    likelihood ratio over its bin exceeds 1 under mixing before tempering: (1 - lam) H^(1 - 1/T) > 1 at H = 3)."""
    import epirefine.pipeline as P
    orig = P.calibrate

    def calibrate(*a, **k):
        out = orig(*a, **k)
        assert out["lam"] == 1.0  # the small community does not need relatives
        out["community"] = {"temperature": out["temperature"], "eta": out["eta"], "lam": 1.0}
        out.update(eta=0.05, lam=0.1, temperature=20.0)
        out["fit"] = dict(out["fit"], eta=0.05, lam=0.1, temperature=20.0)
        return out
    monkeypatch.setattr(P, "calibrate", calibrate)


def test_a_collapsing_relatives_model_is_replaced_by_community_only(small_community, tmp_path, forced_relatives):
    run_community(small_community, tmp_path / "legacy", seed=3, calibration="legacy")
    p_leg = json.loads((tmp_path / "legacy" / "params.json").read_text())
    assert p_leg["p_unknown"] > 0.99                       # the collapse, as on ZymoFecal
    assert (_decisions(tmp_path / "legacy")["status"] == "included").sum() == 0
    run_community(small_community, tmp_path / "dep", seed=3, calibration="deployed-unknown")
    p = json.loads((tmp_path / "dep" / "params.json").read_text())
    chk = p["unknown_model_deployed_check"]
    assert chk["p_unknown_relatives"] > 0.99 and chk["kept_relatives"] is False and chk["gain"] < -2 * chk["se"]
    assert p["unknown_lam"] == 1.0 and p["temperature"] < 20 and p["p_unknown"] < 0.9
    assert (_decisions(tmp_path / "dep")["status"] == "included").sum() >= 1


def test_feature_refuses_options_it_cannot_check(small_community, tmp_path):
    with pytest.raises(ValueError, match="deployed-unknown"):
        run_community(small_community, tmp_path / "x", calibration="deployed-unknown,prior-matched")
    with pytest.raises(ValueError, match="deployed-unknown"):
        run_community(small_community, tmp_path / "y", calibration="deployed-unknown", unlisted_genome=True)


def test_check_runs_with_temper_then_mix(small_community, tmp_path, forced_relatives):
    """With the coherent order the check uses the tempered-then-mixed unknown score; the decisions follow its verdict."""
    run_community(small_community, tmp_path / "out", seed=3, calibration="temper-then-mix,deployed-unknown")
    p = json.loads((tmp_path / "out" / "params.json").read_text())
    chk = p["unknown_model_deployed_check"]
    assert set(chk) == {"p_unknown_relatives", "p_unknown_community", "gain", "se", "kept_relatives"}
    if chk["kept_relatives"]:
        assert p["unknown_lam"] == 0.1 and p["temperature"] == 20.0
    else:
        assert p["unknown_lam"] == 1.0 and p["temperature"] < 20


def test_tempered_bin_prior_is_a_shift_of_the_bin_scores():
    """log(H) / T instead of log(H): the same posterior as adding (T - 1) log(H) to every bin score."""
    from epirefine import model as M
    rng = np.random.default_rng(0)
    S, Su, T, H = rng.normal(0, 5, (30, 7)), rng.normal(0, 5, 30), 3.0, 7
    plain = M.posterior(S, Su, 0.3, T)
    old = M.set_tempered_bin_prior(True)
    try:
        tempered = M.posterior(S, Su, 0.3, T)
        per_unit = M.posterior(S, Su, 0.3, np.full(30, T))
    finally:
        M.set_tempered_bin_prior(old)
    assert M._TEMPERED_BIN_PRIOR is False
    shifted = M.posterior(S + (T - 1) * np.log(H), Su * 1.0, 0.3, T)
    np.testing.assert_allclose(tempered, shifted, rtol=1e-10)
    np.testing.assert_allclose(per_unit, tempered, rtol=1e-10)
    assert not np.allclose(plain, tempered)
    assert (tempered[:, H] <= plain[:, H] + 1e-12).all()  # 'unknown' loses the log(H) (1 - 1/T) it gained from T


def test_tempered_bin_prior_at_temperature_one_changes_nothing():
    from epirefine import model as M
    rng = np.random.default_rng(1)
    S, Su = rng.normal(0, 5, (20, 5)), rng.normal(0, 5, 20)
    plain = M.posterior(S, Su, 0.2, 1.0)
    old = M.set_tempered_bin_prior(True)
    try:
        np.testing.assert_allclose(M.posterior(S, Su, 0.2, 1.0), plain, rtol=1e-12)
    finally:
        M.set_tempered_bin_prior(old)


def test_run_with_tempered_bin_prior_restores_the_default_form(small_community, tmp_path):
    from epirefine import model as M
    run_community(small_community, tmp_path / "t", seed=3, calibration="default,tempered-bin-prior")
    p = json.loads((tmp_path / "t" / "params.json").read_text())
    assert p["calibration_features"] == ["bin-parsimony", "tempered-bin-prior"]
    assert M._TEMPERED_BIN_PRIOR is False
    assert "tempered-bin-prior" not in parse_features("default") | parse_features("coherent") | parse_features("consistent")
