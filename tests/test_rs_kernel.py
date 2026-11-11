"""The optional compiled pileup reader (package epirefine-rs): used when importable and the run has the site
definition it implements, switched off by EPIREFINE_RS=0, and never needed. The comparison tests are skipped
when the package is not installed; the fallback tests always run."""
import importlib.util
import sys

import polars as pl
import pytest

from epirefine import sites
from helpers import INVARIANT_FILES, assert_same_outputs, run_pipeline

HAVE_RS = importlib.util.find_spec("epirefine_rs") is not None
needs_rs = pytest.mark.skipif(not HAVE_RS, reason="epirefine-rs is not installed")
LEGACY = {"group_rank": None, "pairing": "sequence", "strand_sites": "sum", "motifs": [("GATC", 1, "a")]}


def test_switched_off_by_the_environment(monkeypatch):
    for off in ("0", "false", "no", "off", " OFF "):
        monkeypatch.setenv("EPIREFINE_RS", off)
        assert sites._rs() is None and sites._rs(LEGACY) is None


def test_absent_package_means_the_python_pass(monkeypatch):
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    monkeypatch.setitem(sys.modules, "epirefine_rs", None)  # import epirefine_rs -> ImportError
    assert sites._rs() is None and sites._rs(LEGACY) is None


@needs_rs
def test_used_only_for_the_site_definition_it_implements(monkeypatch):
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    assert sites._rs() is not None and sites._rs(LEGACY) is not None and sites._rs({"motifs": [("GATC", 1, "a")]}) is not None
    assert sites._rs({**LEGACY, "strand_sites": "auto"}) is None
    assert sites._rs({**LEGACY, "strand_sites": "split"}) is None
    # a pairing rule counts as implemented when it pairs the motif list as "sequence" does
    plain = [("GATC", 1, "a"), ("GGATG", 2, "a"), ("CATCC", 1, "a")]
    two_positions = plain + [("GAANNNNNNTGC", 1, "a"), ("GAANNNNNNTGC", 2, "a"), ("GCANNNNNNTTC", 1, "a")]
    assert sites._rs({**LEGACY, "pairing": "position", "motifs": plain}) is not None
    assert sites._rs({**LEGACY, "pairing": "position", "motifs": two_positions}) is None
    assert sites._rs({**LEGACY, "pairing": "separate", "motifs": plain}) is None
    assert sites._rs({**LEGACY, "group_rank": [1, 2]}) is None


def _site_rows(run) -> pl.DataFrame:
    parts = sorted((run.out / "cache" / "sites").glob("*.parquet"))
    return pl.concat([pl.read_parquet(p) for p in parts]).sort("contig", "group", "pos")


@needs_rs
def test_kernel_and_python_pass_give_the_same_sites_and_outputs(community, tmp_path, monkeypatch, many_ranges):
    """Same site rows (values and types) and the same output files, on a pileup read in several ranges."""
    kernel_sites = dict(shared_sites="sample", strand_sites="sum")  # what it implements (default pairing: same pairs here)
    monkeypatch.setenv("EPIREFINE_RS", "0")
    py = run_pipeline(**community.args(tmp_path / "py"), workers=2, **kernel_sites)
    monkeypatch.setenv("EPIREFINE_RS", "1")
    rs = run_pipeline(**community.args(tmp_path / "rs"), workers=2, **kernel_sites)
    assert "epirefine-rs" in rs.log and "epirefine-rs" not in py.log
    a, b = _site_rows(py), _site_rows(rs)
    assert a.schema == b.schema and a.equals(b)
    assert_same_outputs(py.out, rs.out, INVARIANT_FILES)
