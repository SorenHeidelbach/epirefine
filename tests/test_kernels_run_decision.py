"""Kernels or numpy is decided once per run (epirefine._rs.start_run), handed to the worker processes, and checked
there."""
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pytest

from epirefine import _rs as R

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
