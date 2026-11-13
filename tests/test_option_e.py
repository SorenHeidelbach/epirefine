"""Option E (0.12; question U13 in the dev log, D63): the unknown-model switch removed by default. Contigs from
genomes without a bin are modelled as random community draws on every run (--relatives community-only), the share of
such contigs is estimated at T = 1 (--unknown-share t1), and the relatives test of 0.11 is reported as a number
(params.json relatives_statistic). The switch stays as the deprecated --relatives switch, which reproduces 0.11.1."""
import json

import pytest

from helpers import assert_same_outputs, copy_cache, run_pipeline
from epirefine.calibrate import CoreOptions
from epirefine.pipeline import resolve_unknown_model
import test_golden as golden


def _params(run):
    return json.loads((run.out / "params.json").read_text())


def _quiet(*a, **k):
    pass


def test_the_default_is_option_e():
    assert resolve_unknown_model(None, None, log=_quiet) == ("community-only", "t1")
    core = CoreOptions(relatives="community-only", unknown_share="t1")
    assert not core.decisions_default          # the decisions run on the community-only path, with the late evidence


def test_switch_alone_is_the_path_of_0_11_1():
    assert resolve_unknown_model("switch", None, log=_quiet) == ("switch", "deployed")
    assert CoreOptions(relatives="switch", unknown_share="deployed").decisions_default
    # an explicit share is kept
    assert resolve_unknown_model("switch", "t1", log=_quiet) == ("switch", "t1")
    assert resolve_unknown_model(None, "deployed", log=_quiet) == ("community-only", "deployed")


def test_a_run_without_calibration_falls_back_to_the_switch():
    said = []
    assert resolve_unknown_model(None, None, do_calibrate=False, log=said.append) == ("switch", "deployed")
    assert "switch" in said[0]
    with pytest.raises(ValueError):   # the removed designs are refused by name
        CoreOptions(relatives="per-bin")


def test_default_run_records_option_e_and_the_relatives_statistic(base_run):
    par = _params(base_run)
    assert par["relatives"] == "community-only" and par["unknown_share"] == "t1"
    rs = par["relatives_statistic"]
    assert set(rs) == {"gain", "se", "statistic", "se_over_bins", "relatives_eta", "relatives_lam",
                       "relatives_temperature", "community_temperature", "switch_choice"}
    assert rs["statistic"] == pytest.approx(rs["gain"] / rs["se"])
    assert rs["relatives_lam"] < 1 and rs["switch_choice"] == "community only"
    assert "relatives statistic: " in base_run.log and "report only" in base_run.log


def test_relatives_switch_reproduces_the_old_path(community, base_run, tmp_path):
    """--relatives switch runs the code path of 0.11.1: its decisions and fitted parameters equal the golden
    snapshot, which was recorded before option E; and it equals --relatives switch --unknown-share deployed."""
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), relatives="switch")
    par = _params(run)
    assert par["relatives"] == "switch" and par["unknown_share"] == "deployed"
    assert "deprecated" in run.log and "p_unknown_t1" not in par
    want = [line.split("\t") for line in golden.GOLDEN.read_text().splitlines()]
    got = [line.split("\t") for line in golden._snapshot(run)]
    header = want[0]
    assert got[0] == header and len(got) == len(want)
    for g, w in zip(got[1:], want[1:]):
        assert g[0] == w[0]
        for name, a, b in zip(header[1:], w[1:], g[1:]):
            assert (a == b) if name in golden.LABELS else golden._same_number(a, b), (g[0], name, a, b)
    gp = json.loads(golden.GOLDEN_PARAMS.read_text())
    assert all(golden._close(gp[k], golden._params(run)[k]) for k in golden.PARAMS)
    out2 = copy_cache(base_run, tmp_path / "out2")
    explicit = run_pipeline(**community.args(out2), cache_dir=str(out2 / "cache"), relatives="switch",
                            unknown_share="deployed")
    assert_same_outputs(run.out, explicit.out)


def test_where_the_switch_picks_relatives_only_the_old_path_follows_it(sisters_run, sisters_switch_run):
    """The sister community (synth.SISTERS), where the 2-SE switch of 0.11 really chooses relatives: the default
    forces community only and records it; --relatives switch keeps the relatives model as 0.11.1 did."""
    e, s = _params(sisters_run), _params(sisters_switch_run)
    assert e["relatives_statistic"]["switch_choice"] == "relatives" and e["relatives_statistic"]["statistic"] > 2
    assert e["unknown_lam"] == 1.0 and e["unknown_model_forced"]["switch_lam"] < 1
    assert e["unknown_model_forced"]["community_temperature"] == e["temperature"]
    assert e["unknown_model_forced"]["switch_temperature"] == s["temperature"] != e["temperature"]
    assert "unknown model forced to community only" in sisters_run.log
    assert s["unknown_lam"] < 1 and "unknown_model_forced" not in s


def test_calibration_tables_describe_the_community_fit_used(sisters, sisters_run, sisters_switch_run, tmp_path,
                                                            monkeypatch):
    """Review F2: where the switch would choose relatives, the calibration tables and the log line describe the
    community-only fit the decisions use (they were made from the switch's choice before the forcing). The
    reference is computed independently: the best community-only row of temperature_fit.tsv, reported by the
    calibration's own report function."""
    import polars as pl

    import epirefine.pipeline as P
    T = _params(sisters_run)["temperature"]
    Ts = _params(sisters_switch_run)["temperature"]
    assert T != Ts
    assert f"-> used {T:g}" in sisters_run.log and f"-> used {Ts:g}" not in sisters_run.log
    seen = {}
    real = P.calibrate

    def keep(*a, **k):
        seen["cal"] = real(*a, **k)
        return seen["cal"]
    monkeypatch.setattr(P, "calibrate", keep)
    out = copy_cache(sisters_run, tmp_path / "again")
    run = run_pipeline(**sisters.args(out), cache_dir=str(out / "cache"))
    tf = pl.read_csv(run.out / "calibration" / "temperature_fit.tsv", separator="\t")
    com = tf.filter((pl.col("lam") >= 1) & (pl.col("temperature") >= 1)).sort("log_loss", maintain_order=True).row(
        0, named=True)
    assert com["temperature"] == T
    fit = {"temperature": T, "eta": com["eta"], "lam": 1.0, "curve": None,
           "fitted_temperature": tf.filter(pl.col("lam") >= 1).sort("log_loss", maintain_order=True)["temperature"][0]}
    summary, reliability = seen["cal"]["report"](fit, 0.1)
    for name, want in (("summary", summary), ("reliability", reliability)):
        path = tmp_path / f"{name}.tsv"
        want.write_csv(path, separator="\t", float_precision=4)
        assert (run.out / "calibration" / f"{name}.tsv").read_text() == path.read_text(), name
    # the fragments' expected log Bayes factors are divided by the temperature used (the margins at T = 1 are the
    # same pieces and model in both runs)
    fe = pl.read_csv(run.out / "calibration" / "fragments.tsv", separator="\t")
    fs = pl.read_csv(sisters_switch_run.out / "calibration" / "fragments.tsv", separator="\t")
    a = (fe["expected_log_bf_vs_nearest"] * T).drop_nulls().drop_nans()
    b = (fs["expected_log_bf_vs_nearest"] * Ts).drop_nulls().drop_nans()
    assert len(a) == len(b) > 0 and ((a - b).abs() <= 1e-3 * (1 + b.abs())).all()


def test_rerun_of_a_run_from_before_option_e_uses_the_switch(base_run, tmp_path):
    from epirefine.runinfo import rerun_arguments
    par = _params(base_run)
    for k in ("relatives", "unknown_share"):
        par.pop(k)
    said = []
    kw = rerun_arguments(par, tmp_path / "again", log=said.append)
    assert kw["relatives"] == "switch" and kw["unknown_share"] == "deployed"
    assert any("predates option E" in s for s in said)
    kw = rerun_arguments(_params(base_run), tmp_path / "again", log=_quiet)
    assert kw["relatives"] == "community-only" and kw["unknown_share"] == "t1"
