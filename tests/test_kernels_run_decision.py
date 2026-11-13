"""Kernels or numpy is decided once per run (epirefine._rs.start_run), handed to the worker processes, and checked
there; and params.json records which code each kernel's calls took (kernels.ran), not only what was available."""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pytest

from epirefine import _rs as R
from helpers import copy_cache, run_pipeline

HAVE_RS = R.load()[0] is not None
needs_rs = pytest.mark.skipif(not HAVE_RS, reason="no compiled kernels in this install")


def _in_worker():
    """In a spawned worker: does this process use the compiled kernels (raises when it must stop the run)?"""
    from epirefine import _rs
    return _rs.kernels() is not None


def _spawn(fn):
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as ex:
        return ex.submit(fn).result()


@needs_rs
def test_decision_is_fixed_for_the_run(monkeypatch):
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    R.start_run()
    try:
        monkeypatch.setenv("EPIREFINE_RS", "0")  # changed during the run: the run keeps its decision
        assert R.kernels() is not None and R.summary().startswith("kernels: compiled")
    finally:
        R.end_run()
    assert R.kernels() is None  # after the run, EPIREFINE_RS counts again


def test_numpy_decision_is_fixed_for_the_run(monkeypatch):
    monkeypatch.setenv("EPIREFINE_RS", "0")
    R.start_run()
    try:
        monkeypatch.delenv("EPIREFINE_RS")
        assert R.kernels() is None and R.reason() == R.SWITCHED_OFF
    finally:
        R.end_run()


@needs_rs
def test_import_is_kept_once_per_process(monkeypatch):
    """A build removed (or replaced) after the first import does not change the code of this process."""
    import sys
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    rs = R.load()[0]
    monkeypatch.setitem(sys.modules, "epirefine._kernels", None)
    assert R.load()[0] is rs and R.kernels() is rs


@needs_rs
def test_workers_follow_the_main_process(monkeypatch):
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    R.start_run()
    try:
        assert os.environ[R.ENV_RUN] == f"sha256:{R.module_sha256(R.load()[0])}"
        assert _spawn(_in_worker) is True
    finally:
        R.end_run()
    monkeypatch.setenv("EPIREFINE_RS", "0")
    R.start_run()
    try:
        monkeypatch.delenv("EPIREFINE_RS")  # a worker that would load the kernels on its own still runs numpy
        assert os.environ[R.ENV_RUN] == "numpy" and _spawn(_in_worker) is False
    finally:
        R.end_run()


def test_worker_with_another_build_stops_the_run(monkeypatch):
    """A worker that cannot load the build the run started with (another file, or none) raises: the run stops
    instead of mixing two builds or the two code paths."""
    monkeypatch.delenv("EPIREFINE_RS", raising=False)
    monkeypatch.setenv(R.ENV_RUN, "sha256:" + "0" * 64)
    with pytest.raises(RuntimeError, match="worker process"):
        _spawn(_in_worker)


def test_counts_of_workers_are_added():
    R.count("cell_sums", True)
    R.add_counts({"cell_sums": [2, 3], "pileup_reader": [1, 0]})
    assert R.ran() == {"cell_sums": {"compiled": 4, "numpy": 2}, "pileup_reader": {"compiled": 0, "numpy": 1}}
    assert R.take_counts() is None  # the main process keeps its counts


def _ran(run) -> dict:
    return json.loads((run.out / "params.json").read_text())["kernels"]["ran"]


def test_params_record_which_code_ran(base_run):
    """kernels.ran: the calls of each kernel by path, the worker processes' included (the pileup pass and the unit
    tables run in workers)."""
    ran = _ran(base_run)
    assert {"pileup_reader", "cell_sums", "bin_loglik", "npmle"} <= set(ran)
    used = json.loads((base_run.out / "params.json").read_text())["kernels"]["used"]
    if used:
        assert ran["pileup_reader"]["compiled"] >= 1 and ran["cell_sums"]["compiled"] >= 1
    else:
        assert all(v["compiled"] == 0 for v in ran.values())
    assert "kernels ran: " in base_run.log


def test_numpy_run_records_numpy(base_run, tmp_path, monkeypatch):
    """With EPIREFINE_RS=0 the record says numpy for every call (before, `used` only said what was available)."""
    monkeypatch.setenv("EPIREFINE_RS", "0")
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**{**base_run_args(base_run), "out": out, "cache_dir": out / "cache"})
    k = json.loads((run.out / "params.json").read_text())["kernels"]
    assert k["used"] is False and k["ran"] and all(v["compiled"] == 0 for v in k["ran"].values())
    assert sum(v["numpy"] for v in k["ran"].values()) > 0


def base_run_args(base_run) -> dict:
    from epirefine.runinfo import rerun_arguments
    p = json.loads((base_run.out / "params.json").read_text())
    return rerun_arguments(p, base_run.out)
