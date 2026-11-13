"""The unknown-model choice: the relatives test's standard error over bins (bin-parsimony, 0.11.1) and the collapse a
relatives model causes under the switch of 0.11. (The calibration features deployed-unknown and tempered-bin-prior,
and the feature list itself, were removed in 0.12, U12; their tests went with them.)"""
import json

import numpy as np
import polars as pl
import pytest

from small_community import run_community
from epirefine.calibrate import FragmentFit


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


def test_default_run_records_no_calibration_feature_list(small_community, tmp_path):
    run_community(small_community, tmp_path / "out", seed=3)
    p = json.loads((tmp_path / "out" / "params.json").read_text())
    assert "calibration_features" not in p and "calibration" not in p and "unknown_model_deployed_check" not in p


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
        out["switch"] = {"eta": 0.05, "lam": 0.1, "temperature": 20.0}
        if k.get("community_only"):   # option E: calibrate() itself replaces the switch's choice by community only
            return out
        out.update(eta=0.05, lam=0.1, temperature=20.0)
        out["fit"] = dict(out["fit"], eta=0.05, lam=0.1, temperature=20.0)
        return out
    monkeypatch.setattr(P, "calibrate", calibrate)


def _decisions(out):
    return pl.read_csv(out / "contig_decisions.tsv", separator="\t", infer_schema_length=0).sort("contig")


def test_a_collapsing_relatives_model_needs_the_switch(small_community, tmp_path, forced_relatives):
    # the collapse needs the unknown-model switch of 0.11 (since 0.12 a deprecated option): option E forces
    # community only, so the default run cannot collapse this way
    run_community(small_community, tmp_path / "switch", seed=3, relatives="switch")
    p_sw = json.loads((tmp_path / "switch" / "params.json").read_text())
    assert p_sw["p_unknown"] > 0.99                       # the collapse, as on ZymoFecal
    assert (_decisions(tmp_path / "switch")["status"] == "included").sum() == 0
    run_community(small_community, tmp_path / "e", seed=3)
    p = json.loads((tmp_path / "e" / "params.json").read_text())
    assert p["unknown_lam"] == 1.0 and p["p_unknown"] < 0.9 and "unknown_model_forced" in p
