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
