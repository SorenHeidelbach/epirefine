"""The guard for calibrations without a usable temperature (0.12, dev log proposal 2026-11-13 calib_guard): when the
calibrated temperature is at the cap of the grid (20), the run warns, records `calibration_guard` in params.json and
makes no call (no flag, inclusion or move). It never fires on a normal calibration, never with a temperature given
by the user, and never on the deprecated `--relatives switch` path (which reproduces 0.11.1)."""
import json

import polars as pl

from epirefine.calibrate import DEFAULT_TEMPS, calibration_guard
from helpers import copy_cache, run_pipeline

CALLS = ("contaminant", "included", "moved")


def _params(run):
    return json.loads((run.out / "params.json").read_text())


def _calls(run):
    st = pl.read_csv(run.out / "contig_decisions.tsv", separator="\t")["status"].to_list()
    return sum(s in CALLS for s in st)


def test_guard_fires_only_at_the_cap():
    cap = max(DEFAULT_TEMPS)
    assert cap == 20.0
    g = calibration_guard(cap, DEFAULT_TEMPS, 1, 0)
    assert g == {"fired": True, "reason": "temperature at the grid cap (20)", "temperature": 20.0, "cap": 20.0,
                 "pieces": 1, "test_b_pieces": 0, "fallback": "no calls"}
    below = sorted(DEFAULT_TEMPS)[-2]       # 17.154: the grid point below the cap
    for T in (1.082, 5.857, below):
        g = calibration_guard(T, DEFAULT_TEMPS, 1, 0)   # few pieces alone never fire it
        assert not g["fired"] and g["reason"] is None and g["fallback"] is None and g["temperature"] == T
    # a temperature per contig (a sites law) fires when any contig is at the cap
    assert calibration_guard([1.0, 20.0], DEFAULT_TEMPS, 10, 5)["fired"]


def test_normal_run_records_a_silent_guard(base_run):
    g = _params(base_run)["calibration_guard"]
    assert g["fired"] is False and g["fallback"] is None and g["temperature"] < g["cap"] == 20.0
    assert g["pieces"] > 0 and g["test_b_pieces"] > 0
    assert "calibration guard" not in base_run.log


def _at_cap(monkeypatch):
    """A calibration that ends at the cap, as on the stress runs with one uninformative piece."""
    import epirefine.pipeline as P
    real = P.calibrate

    def capped(*a, **k):
        cal = real(*a, **k)
        cal["fit"] = dict(cal["fit"], temperature=20.0)
        cal.update(temperature=20.0)
        if cal["community"] is not None:
            cal["community"] = dict(cal["community"], temperature=20.0)
        return cal
    monkeypatch.setattr(P, "calibrate", capped)


def test_guard_at_the_cap_warns_and_makes_no_call(community, base_run, tmp_path, monkeypatch):
    assert _calls(base_run) > 0
    _at_cap(monkeypatch)
    out = copy_cache(base_run, tmp_path / "g")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"))
    par = _params(run)
    g = par["calibration_guard"]
    assert g["fired"] is True and g["fallback"] == "no calls" and g["temperature"] == 20.0
    assert par["temperature"] == 20.0
    assert run.log.count("WARNING: calibration guard:") == 1
    assert _calls(run) == 0
    # the Nanomotif-compatible files list no change: the refined binning is the given one
    assert pl.read_csv(run.out / "bin_contamination.tsv", separator="\t").height == 0
    assert pl.read_csv(run.out / "include_contigs.tsv", separator="\t").height == 0
    given = pl.read_csv(community.contig_bin, separator="\t", has_header=False).sort(pl.all())
    for f in ("new_contig_bin.tsv", "decontaminated_contig_bin.tsv"):
        got = pl.read_csv(run.out / f, separator="\t").sort(pl.all())   # these two carry a header
        assert got.rows() == given.rows(), f
    # the posteriors are still written
    dec = pl.read_csv(run.out / "contig_decisions.tsv", separator="\t")
    assert dec["p_map"].drop_nulls().len() > 0


def test_a_user_temperature_or_the_switch_bypasses_the_guard(community, base_run, tmp_path, monkeypatch):
    _at_cap(monkeypatch)
    out = copy_cache(base_run, tmp_path / "t")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), temperature=1.0)
    assert _params(run)["calibration_guard"] is None and "calibration guard" not in run.log
    assert _calls(run) > 0
    out = copy_cache(base_run, tmp_path / "s")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), relatives="switch")
    assert _params(run)["calibration_guard"] is None and "calibration guard" not in run.log


def test_no_calibration_records_no_guard(community, base_run, tmp_path):
    out = copy_cache(base_run, tmp_path / "n")
    run = run_pipeline(**community.args(out), cache_dir=str(out / "cache"), do_calibrate=False)
    assert _params(run)["calibration_guard"] is None
