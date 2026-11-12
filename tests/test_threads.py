"""The thread count must not change a result (gate R2 of the 0.12 mission).

The model's passes take their thread count from the cores the process may use (model._cores, capped by
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS), and BLAS has a thread count of its own. A run on 1 and on 12 threads, each
with BLAS on as many threads, must give the same output files, params.json in full precision included, as the run
on 4 threads, with the numpy code and with the compiled kernels each. (Kernels against numpy:
tests/test_rs_model.py, test_run_takes_the_same_decisions, which allows the last bits of params.json to differ.)
Up to 0.11 the numpy scoring sums were taken in one block of motif groups per thread, so every score depended on
the core count in its last bits.
"""
import pytest

import epirefine.model as M
from helpers import INVARIANT_FILES, assert_same_outputs, copy_cache, run_pipeline


def _kernels_available() -> bool:
    try:
        import epirefine_rs
    except ImportError:
        return False
    return epirefine_rs.supports("model_kernels")


BACKENDS = ["numpy", pytest.param("kernels", marks=pytest.mark.skipif(not _kernels_available(),
                                                                       reason="epirefine-rs not installed"))]


def _on_threads(monkeypatch, community, base_run, out, n: int, backend: str):
    """A run from base_run's site cache with every thread count of the process set to n."""
    with monkeypatch.context() as mp:
        mp.setenv("EPIREFINE_RS", "0" if backend == "numpy" else "1")
        mp.setattr(M, "_cores", lambda: n)
        mp.setenv("OMP_NUM_THREADS", str(n))
        mp.setenv("OPENBLAS_NUM_THREADS", str(n))
        api = M._blas_thread_api()
        before = api[1]() if api else None
        if api:
            api[0](n)
        try:
            assert M._n_threads() == min(16, n)
            return run_pipeline(**community.args(copy_cache(base_run, out)))
        finally:
            if api and before:
                api[0](before)


@pytest.fixture(scope="module")
def reference(community, base_run, tmp_path_factory):
    """Per backend, the run on 4 threads."""
    runs = {}
    with pytest.MonkeyPatch.context() as mp:
        for backend in ("numpy", "kernels") if _kernels_available() else ("numpy",):
            runs[backend] = _on_threads(mp, community, base_run, tmp_path_factory.mktemp(f"t4_{backend}") / "out", 4,
                                        backend)
    return runs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("threads", [1, 12])
def test_thread_count_gives_identical_outputs(monkeypatch, community, base_run, reference, tmp_path, threads, backend):
    run = _on_threads(monkeypatch, community, base_run, tmp_path / "out", threads, backend)
    assert run.reused_cache
    assert_same_outputs(reference[backend].out, run.out, INVARIANT_FILES)
