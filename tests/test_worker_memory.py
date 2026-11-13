"""The worker peak memory in params.json is the worker's own, not the main process's.

getrusage(RUSAGE_CHILDREN) gave a spawned worker the main process's high-water mark at the moment the pool
started (the kernel carries ru_maxrss through fork and exec), so on ZymoFecal "the largest worker" read 13.4 GB,
the main log's peak at 137 s.
"""
import sys

import numpy as np
import pytest

from epirefine import detection, runinfo


def _square(x):
    return x * x


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_worker_peak_is_not_the_main_process_peak(monkeypatch):
    runinfo.start_clock()
    big = np.ones(300_000_000 // 8)  # 0.3 GB resident in the main process before the pool starts
    assert runinfo.peak_rss_gb() > 0.3
    monkeypatch.setattr(detection, "POOL_MIN_ROWS", 0)
    assert detection._map(_square, [np.arange(3.0)] * 4, workers=2)[0].tolist() == [0.0, 1.0, 4.0]
    del big
    worker = runinfo.peak_rss_gb(children=True)
    assert 0.0 < worker < 0.25, worker  # a worker that imports numpy holds about 0.1 GB
    assert runinfo.finish({})["peak_rss_worker_gb"] == round(worker, 3)
