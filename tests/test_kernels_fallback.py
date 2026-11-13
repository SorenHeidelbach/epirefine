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


def test_install_without_the_extension_module(tmp_path):
    """The package as the pure wheel installs it (no _core file), in a fresh interpreter: the reason is "no compiled
    extension", not a failed load (hiding the module in sys.modules, as above, does not show the real import error)."""
    import shutil
    import subprocess
    from pathlib import Path
    import epirefine
    shutil.copytree(Path(epirefine.__file__).parent, tmp_path / "epirefine",
                    ignore=lambda d, names: [n for n in names if n.startswith("_core") or n == "__pycache__"])
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    env.pop("EPIREFINE_RS", None)
    code = "import epirefine, sys; from epirefine import _rs; assert epirefine.__file__.startswith(sys.argv[1]); print(_rs.reason())"
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "no compiled extension in this install (pure wheel, or a source tree without a build)"


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
    R.reset()
    assert R.kernels() is None and R.reason().startswith(f"the compiled extension has interface {rs.API}")


def test_interface_number_comes_from_the_build():
    """The interface number is exported by the compiled extension, not by a Python file of the same tree (that
    compared the code with itself: a stale build passed)."""
    rs = R.load()[0]
    if rs is None:
        pytest.skip("no compiled kernels in this install")
    assert isinstance(rs._core.API, int) and rs.API == rs._core.API >= R.REQUIRED_API


def test_stale_build_is_refused(monkeypatch):
    """A build older than the Python code (its _core reports a lower interface number, or none) runs numpy."""
    rs = R.load()[0]
    if rs is None:
        pytest.skip("no compiled kernels in this install")
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    monkeypatch.setattr(rs._core, "API", R.REQUIRED_API - 1)
    R.reset()  # the import is kept once per process: look again
    assert R.kernels() is None and R.reason().startswith(f"the compiled extension has interface {R.REQUIRED_API - 1}")
    monkeypatch.delattr(rs._core, "API")
    R.reset()
    assert R.kernels() is None and R.reason().startswith("the compiled extension has interface 0")


def test_build_without_cpu_path_does_not_load(monkeypatch):
    """A _core without a function the run record needs (cpu_path) falls back to numpy; the log head and params.json
    are still written (summary() and runinfo.kernels() used to raise AttributeError)."""
    rs = R.load()[0]
    if rs is None:
        pytest.skip("no compiled kernels in this install")
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    monkeypatch.delattr(rs._core, "cpu_path")
    R.reset()
    why = R.reason()
    assert R.kernels() is None and why.startswith("the compiled extension does not load: AttributeError")
    assert R.summary() == f"kernels: numpy ({why})"
    k = runinfo.kernels()
    assert k["available"] is False and k["used"] is False and k["reason"] == why


def _args():
    from inspect import signature
    from epirefine.pipeline import run
    a = {k: v.default for k, v in signature(run).parameters.items()}
    a.update(assembly="a.fa", pileup="p.bed", contig_bin="cb.tsv", motifs_path="m.tsv", out="out")
    return a
