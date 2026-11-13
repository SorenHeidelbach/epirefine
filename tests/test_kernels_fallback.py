"""epirefine._rs, the one way to the compiled kernels: the reason the numpy code runs, the log lines, the
record in params.json, and EPIREFINE_RS=0. The pure wheel has no extension; this is simulated here by hiding
epirefine._kernels. EPIREFINE_EXPECT_KERNELS=1 (set by the CI jobs that test a compiled wheel) makes a missing
or broken extension a failure instead of a fallback."""
import os
import sys

import pytest

from epirefine import _rs as R
from epirefine import runinfo


def test_expected_kernels_are_used(monkeypatch):
    if os.environ.get("EPIREFINE_EXPECT_KERNELS") != "1":
        pytest.skip("EPIREFINE_EXPECT_KERNELS is not 1")
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    assert R.load()[1] is None and R.kernels("model_kernels") is not None
    k = runinfo.kernels()
    assert k["available"] and k["used"] and k["reason"] is None and k["cpu_path"] in ("avx2_fma", "generic")
    if os.environ.get("EPIREFINE_RS_NO_SIMD") == "1":
        assert k["cpu_path"] == "generic"


def test_no_extension_gives_numpy_with_one_warning(monkeypatch):
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    monkeypatch.setitem(sys.modules, "epirefine._kernels", None)  # the pure wheel: the import fails
    rs, why = R.load()
    assert rs is None and why.startswith("no compiled extension in this install")
    assert R.kernels() is None and R.kernels("pileup_reader") is None
    assert R.summary() == f"kernels: numpy ({why})"
    assert R.notice() == f"WARNING: compiled kernels not used ({why}): running the numpy code; slower, same decisions"
    k = runinfo.kernels()
    assert k["available"] is False and k["used"] is False and k["reason"] == why
    lines = []
    runinfo.begin(_args(), lines.append)
    assert sum("compiled kernels not used" in x for x in lines) == 1 and f"kernels: numpy ({why})" in lines


def test_switched_off_gives_numpy_without_warning(monkeypatch):
    monkeypatch.setenv("EPIREFINE_RS", "0")
    assert R.kernels() is None and R.reason() == "switched off by EPIREFINE_RS=0" and R.notice() is None
    lines = []
    runinfo.begin(_args(), lines.append)
    assert "kernels: numpy (switched off by EPIREFINE_RS=0)" in lines
    assert not any("compiled kernels not used" in x for x in lines)


def test_broken_extension_is_reported(monkeypatch):
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    import importlib.abc

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "epirefine._kernels":
                raise ImportError("libfoo.so: cannot open shared object file")
            return None
    monkeypatch.delitem(sys.modules, "epirefine._kernels", raising=False)
    monkeypatch.delitem(sys.modules, "epirefine._kernels._core", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Finder(), *sys.meta_path])
    rs, why = R.load()
    assert rs is None and why == "the compiled extension does not load: libfoo.so: cannot open shared object file"


def test_old_interface_is_refused(monkeypatch):
    rs = R.load()[0]
    if rs is None:
        pytest.skip("no compiled kernels in this install")
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    monkeypatch.setattr(R, "REQUIRED_API", rs.API + 1)
    assert R.kernels() is None and R.reason().startswith(f"the compiled extension has interface {rs.API}")


def _args():
    from inspect import signature
    from epirefine.pipeline import run
    a = {k: v.default for k, v in signature(run).parameters.items()}
    a.update(assembly="a.fa", pileup="p.bed", contig_bin="cb.tsv", motifs_path="m.tsv", out="out")
    return a
