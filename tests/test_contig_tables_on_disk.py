"""The contig tables (L_t) written to a file as the unit-table parts arrive and read back after the calibration:
the same values, the same output files, with one worker and with three."""
import numpy as np
import pytest

from helpers import copy_cache, differing_files, run_pipeline


def test_rowfile_round_trip(tmp_path, monkeypatch):
    from epirefine import sites
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1)  # write back and drop pages after every write
    rng = np.random.default_rng(3)
    ref = np.zeros((40, 5, 7), np.float32)
    f = sites.RowFile(tmp_path, ref.shape)
    assert not list(tmp_path.iterdir())  # unlinked temporary file
    for ids in (np.array([3, 9, 10, 11, 30]), np.array([0, 1, 2, 7]), np.array([39]), np.array([12, 13])):
        rows = rng.random((len(ids), 5, 7)).astype(np.float32)  # parts arrive out of row order
        f.write_rows(ids, rows)
        ref[ids] = rows
    f.finish()
    f.prefetch()
    assert f.shape == ref.shape and len(f) == 40 and f.nbytes == ref.nbytes and f.dtype == np.float32
    assert np.array_equal(np.asarray(f), ref)  # rows never written read as zeros
    for sl in (slice(0, 8), slice(5, 6), slice(35, 100), slice(12, 12), slice(8, 14), slice(None)):
        assert np.array_equal(f[sl], ref[sl])
    with pytest.raises(TypeError):
        f[[1, 2]]
    f.close()


@pytest.mark.parametrize("workers", [1, 3])
def test_contig_tables_on_disk_same_outputs(community, base_run, tmp_path, monkeypatch, workers):
    """Every output file of a run whose contig tables go to a file (forced: threshold 0, page drop after every
    part) equals the reference run's (one worker, tables in memory)."""
    from epirefine import sites
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1)
    made, read = [], []
    real_init, real_get = sites.RowFile.__init__, sites.RowFile.__getitem__
    monkeypatch.setattr(sites.RowFile, "__init__", lambda self, *a, **k: (made.append(1), real_init(self, *a, **k))[1])
    monkeypatch.setattr(sites.RowFile, "__getitem__", lambda self, k: (read.append(k), real_get(self, k))[1])
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")), workers=workers)
    assert made == [1] and read  # the disk path ran and the tables were read back from it
    assert not [p for p in run.out.iterdir() if p.name.startswith("tmp")]  # no file left behind
    assert differing_files(base_run.out, run.out) == []


def test_contig_tables_on_disk_with_group_weights(community, base_run, tmp_path, monkeypatch):
    """With down-weighted motif groups (weights < 1; the synthetic community has none, so one is forced) the
    weights multiplied into the rows as they are written give the same outputs as the in-memory tables."""
    import epirefine.pipeline as P
    from epirefine import sites
    real = P.redundancy_weights

    def halved(*a, **k):
        w, tab = real(*a, **k)
        w = w.copy()
        w[0] = 0.5
        return w, tab
    monkeypatch.setattr(P, "redundancy_weights", halved)
    mem = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "mem")))
    assert "1 groups in" in mem.log
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    disk = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "disk")))
    assert differing_files(mem.out, disk.out) == []
    assert differing_files(base_run.out, mem.out) != []  # the weight did change the run
