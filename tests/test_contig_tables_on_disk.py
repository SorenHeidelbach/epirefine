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


def _bad_writeback(sites, monkeypatch, fail_at=1):
    """drop_pages as on a network file system whose write-back fails (quota): from call `fail_at` on, the range to
    write back is lost (it reads back as a hole, zeros) and the error is reported (EDQUOT)."""
    import errno
    import os
    calls = []

    def bad(fd, sync=True, offset=0, length=0):
        calls.append((offset, length))
        if len(calls) >= fail_at:
            n = length or (os.fstat(fd).st_size - offset)
            os.pwrite(fd, bytes(n), offset)
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
    monkeypatch.setattr(sites, "drop_pages", bad)
    return calls


def test_rowfile_writeback_error_is_fatal(tmp_path, monkeypatch):
    """A failed write-back (background thread or finish()) stops with TableFileError; the file is not read back
    (before the fix the next write raised OSError, and the caller read zeros back as 'no sites')."""
    from epirefine import sites
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1)
    _bad_writeback(sites, monkeypatch, fail_at=2)
    read = []
    real_get = sites.RowFile.__getitem__
    monkeypatch.setattr(sites.RowFile, "__getitem__", lambda self, k: (read.append(k), real_get(self, k))[1])
    rows = np.ones((1, 2, 3), np.float32)
    f = sites.RowFile(tmp_path, (6, 2, 3))
    with pytest.raises(sites.TableFileError, match="free space"):
        for k in range(6):
            f.write_rows(np.array([k]), rows)
            f._flusher.join()
    assert k == 2 and not read
    f.close()
    g = sites.RowFile(tmp_path, (6, 2, 3))  # the write-back in finish() fails
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1 << 30)
    g.write_rows(np.array([0]), rows)
    with pytest.raises(sites.TableFileError):
        g.finish()
    g.close()
    assert not list(tmp_path.iterdir())


def test_writeback_error_stops_the_run(community, base_run, tmp_path, monkeypatch):
    """The whole run stops (it used to end normally, the lost rows read as contigs without sites); no file is
    left behind."""
    from epirefine import sites
    from epirefine.pipeline import run
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1)
    calls = _bad_writeback(sites, monkeypatch)
    out = copy_cache(base_run, tmp_path / "out")
    with pytest.raises(sites.TableFileError, match=str(out)):
        run(**community.args(out), workers=1)
    assert calls
    assert not [p for p in out.iterdir() if p.name.startswith("tmp")]


def test_rowfile_failed_write(tmp_path, monkeypatch):
    """A pwrite that fails before any row was written raises OSError (the caller keeps the table in memory); once
    rows were written, TableFileError (the earlier rows may be lost with it: never read back)."""
    import errno
    from epirefine import sites

    def full(*a):
        raise OSError(errno.ENOSPC, "No space left on device")
    f = sites.RowFile(tmp_path, (10, 2, 3))
    real = f._pwrite
    monkeypatch.setattr(f, "_pwrite", full)
    with pytest.raises(OSError) as first:
        f.write_rows(np.array([4]), np.ones((1, 2, 3), np.float32))
    assert not isinstance(first.value, sites.TableFileError)
    monkeypatch.setattr(f, "_pwrite", real)
    f.write_rows(np.array([4]), np.ones((1, 2, 3), np.float32))
    monkeypatch.setattr(f, "_pwrite", full)
    with pytest.raises(sites.TableFileError, match="free space"):
        f.write_rows(np.array([7]), np.ones((1, 2, 3), np.float32))
    f.close()


def _unit_tables_with_parts(sites, monkeypatch, tmp_path, blocks):
    """sites.unit_tables on finished target blocks (a stand-in executor returns them as the parts' results)."""
    from concurrent.futures import Future
    results = iter([{"targets": {"ids": ids, "L": L, "n_sites": np.ones(len(ids), np.int64),
                                 "n_sites_group": (ids, np.zeros(len(ids), int), np.ones(len(ids), np.int32))}}
                    for ids, L in blocks])

    class Ex:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, *a):
            f = Future()
            f.set_result(next(results))
            return f

        def shutdown(self, wait=True, cancel_futures=False):
            pass
    monkeypatch.setattr(sites, "ProcessPoolExecutor", Ex)
    U, M, G = 60, blocks[0][1].shape[1], blocks[0][1].shape[2]
    logs = []
    res = sites.unit_tables([str(i) for i in range(len(blocks))], n_groups=M, det={}, grid_size=G, min_cov=1,
                            chunk_offset=np.zeros(1, np.int64), chunk_size=1, n_chunks=0,
                            target_of=np.zeros(1, np.int64), n_targets=U, frags=None, workers=1,
                            log=logs.append, targets_dir=tmp_path)
    return res, logs


def test_write_reporting_lost_rows_stops(tmp_path, monkeypatch):
    """NFS: a pwrite can report (and so use up) an earlier failed write-back whose pages were lost; a sync after it
    returns no error and the hole reads as zeros. unit_tables must stop, not read the file back (mode D of the
    second review check: 18 true rows came back as zeros with only a warning)."""
    import errno
    import os
    from epirefine import sites
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1 << 40)  # no write-back of our own has run yet
    rng = np.random.default_rng(1)
    blocks = [(ids, (rng.random((len(ids), 3, 8)) + 0.1).astype(np.float32))
              for ids in np.array_split(rng.permutation(60)[:50], 6)]
    real, calls = sites.RowFile._pwrite, []

    def pw(self, buf, off):
        calls.append(off)
        if len(calls) == 3:
            os.pwrite(self._fd, bytes(self._end), 0)  # the earlier pages lost: a hole
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real(self, buf, off)
    monkeypatch.setattr(sites.RowFile, "_pwrite", pw)
    read = []
    real_get = sites.RowFile.__getitem__
    monkeypatch.setattr(sites.RowFile, "__getitem__", lambda self, k: (read.append(k), real_get(self, k))[1])
    with pytest.raises(sites.TableFileError, match="free space"):
        _unit_tables_with_parts(sites, monkeypatch, tmp_path, blocks)
    assert len(calls) == 3 and not read
    assert not list(tmp_path.iterdir())


def test_first_write_fails_table_in_memory(tmp_path, monkeypatch):
    """The first block cannot be written: nothing is in the file, the table is kept in memory (one warning), equal
    to the table built in memory."""
    import errno
    from epirefine import sites
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    rng = np.random.default_rng(2)
    blocks = [(ids, rng.random((len(ids), 3, 8)).astype(np.float32))
              for ids in np.array_split(rng.permutation(60)[:50], 4)]
    truth = np.zeros((60, 3, 8), np.float32)
    for ids, L in blocks:
        truth[ids] = L

    def full(*a):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(sites.RowFile, "_pwrite", full)
    res, logs = _unit_tables_with_parts(sites, monkeypatch, tmp_path, blocks)
    assert isinstance(res["L_t"], np.ndarray) and np.array_equal(res["L_t"], truth)
    assert [m for m in logs if "WARNING" in m] == [logs[0]] and "kept in memory" in logs[0]
    assert not list(tmp_path.iterdir())


def test_error_text_names_space_only_for_space_errors():
    import errno
    from epirefine import sites
    e = sites.table_file_error("contig tables'", "/out", 6e9, OSError(errno.EIO, "Input/output error"))
    assert "free space" not in str(e) and "/out" in str(e)
    for no in (errno.ENOSPC, errno.EDQUOT):
        assert "6.0 GB of free space" in str(sites.table_file_error("x", "/out", 6e9, OSError(no, "full")))


def test_drop_pages_syncs_without_fadvise(tmp_path, monkeypatch):
    """Where posix_fadvise is missing (macOS) drop_pages still syncs: the sync is what reports a failed write-back."""
    import os
    from epirefine import sites
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    synced = []
    name = "fdatasync" if hasattr(os, "fdatasync") else "fsync"
    monkeypatch.setattr(os, name, lambda fd: synced.append(fd))
    with open(tmp_path / "f", "wb") as fh:
        sites.drop_pages(fh.fileno())
        assert synced == [fh.fileno()]


def test_rowfile_checks_blocks(tmp_path):
    from epirefine import sites
    f = sites.RowFile(tmp_path, (5, 2, 3))
    with pytest.raises(ValueError):
        f.write_rows(np.array([0, 1]), np.zeros((3, 2, 3), np.float32))  # more rows than ids
    with pytest.raises(ValueError):
        f.write_rows(np.array([0]), np.zeros((1, 2, 2), np.float32))  # another row shape
    f.write_rows(np.array([2]), np.ones((1, 2, 3), np.float32))
    assert np.array_equal(np.asarray(f)[2], np.ones((2, 3)))
    if int(np.__version__.split(".")[0]) >= 2:
        with pytest.raises(ValueError):
            np.array(f, copy=False)  # a RowFile is always read into a new array
    f.close()
    f.close()  # closing twice is allowed (the run closes it, and closes it again if it fails)


@pytest.mark.parametrize("how", ["no file", "write fails"])
def test_fall_backs_to_memory(community, base_run, tmp_path, monkeypatch, how):
    """The two fall-backs (the file cannot be made; its first write fails) keep the table in memory and give the
    reference outputs."""
    import errno
    from epirefine import sites
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)

    def fail(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    if how == "no file":
        monkeypatch.setattr(sites.RowFile, "__init__", fail)
    else:
        monkeypatch.setattr(sites.RowFile, "_pwrite", fail)  # the first write: nothing in the file yet
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")))
    assert "contig tables kept in memory" in run.log
    assert differing_files(base_run.out, run.out) == []


def test_table_file_closed_when_the_run_fails(community, base_run, tmp_path, monkeypatch):
    import epirefine.pipeline as P
    from epirefine import sites
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    closed = []
    real_close = sites.RowFile.close
    monkeypatch.setattr(sites.RowFile, "close", lambda self: (closed.append(self), real_close(self))[1])

    def broken(*a, **k):
        raise RuntimeError("calibration failed")
    monkeypatch.setattr(P, "calibrate", broken)
    with pytest.raises(RuntimeError, match="calibration failed"):
        P.run(**community.args(copy_cache(base_run, tmp_path / "out")), workers=1)
    assert len(closed) == 1 and closed[0]._fh.closed
    assert not P._OPEN_TABLES


@pytest.mark.parametrize("kernels", [True, False])
def test_score_in_batches_on_a_rowfile(tmp_path, monkeypatch, kernels):
    """FittedModel.score on a RowFile of more than SCORE_BATCH rows (the path of every large sample: scored in
    row batches read from the file) returns what it returns on the array, bit for bit; also with every unit's
    rows kept (lp_rows None: the table is read whole) and with the numpy code (EPIREFINE_RS=0)."""
    import epirefine.model as M
    from epirefine import sites
    from epirefine.grid import Grid
    if kernels:
        if M._rs("score_sums") is None:
            pytest.skip("epirefine-rs without score_sums")
    else:
        monkeypatch.setenv("EPIREFINE_RS", "0")
        monkeypatch.setattr(M, "ROWS_MIN", 1)  # without the kernel only the row form is scored in batches
    rng = np.random.default_rng(11)

    def tables(U, Mg=6, G=64):
        L = -rng.gamma(2.0, 40.0, (U, Mg, G))
        L[rng.random((U, Mg)) < 0.2] = 0.0
        return L.astype(np.float32)
    grid = Grid(64)
    L, cb = tables(150), rng.integers(0, 4, 150)
    m = M.FittedModel(L, cb, 4, ["a"] * 6, grid, 100.0)
    m.set_unknown_params(0.05, 0.9)
    U = M.FittedModel.SCORE_BATCH + 300
    Lt = tables(U)
    Lt[rng.random(U) < 0.1] = 0.0  # rows never written read as zeros
    f = sites.RowFile(tmp_path, Lt.shape)
    order = rng.permutation(np.flatnonzero(Lt.any(axis=(1, 2))))
    for ids in np.array_split(order, 7):  # parts arrive out of row order
        f.write_rows(ids, Lt[ids])
    f.finish()
    f.prefetch()
    batched = []
    real = M.FittedModel._score_in_batches
    monkeypatch.setattr(M.FittedModel, "_score_in_batches", lambda self, *a: (batched.append(1), real(self, *a))[1])
    rows = np.sort(rng.choice(U, 900, replace=False))
    for lp_rows in (rows, None):
        ref, got = m.score(Lt, lp_rows=lp_rows), m.score(f, lp_rows=lp_rows)
        for fld in ("S", "Su", "Au", "Lp", "Su_comm", "R", "lp_rows"):
            a, b = getattr(ref, fld), getattr(got, fld)
            assert (a is None and b is None) or np.array_equal(a, b), fld
    assert batched == [1, 1]  # the list of kept rows took the batched path (on the array and on the file)
    f.close()


@pytest.mark.parametrize("numpy_only", [False, True])
def test_disk_path_whole_table_options(community, base_run, tmp_path, monkeypatch, numpy_only):
    """The disk path with options that read the whole contig table at scoring (the unbinned prior: lp_rows None),
    with gate_unweighted on a down-weighted group, with and without the compiled kernels: the in-memory outputs."""
    import epirefine.pipeline as P
    from epirefine import sites
    if numpy_only:
        monkeypatch.setenv("EPIREFINE_RS", "0")
    real = P.redundancy_weights

    def halved(*a, **k):
        w, tab = real(*a, **k)
        w = w.copy()
        w[0] = 0.5
        return w, tab
    monkeypatch.setattr(P, "redundancy_weights", halved)
    opts = dict(unknown_prior="unbinned", gate_unweighted=True)
    mem = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "mem")), **opts)
    assert "1 groups in" in mem.log and "unbinned-genome prior" in mem.log
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    monkeypatch.setattr(sites, "DISK_FLUSH_BYTES", 1)
    made = []
    real_init = sites.RowFile.__init__
    monkeypatch.setattr(sites.RowFile, "__init__", lambda self, *a, **k: (made.append(1), real_init(self, *a, **k))[1])
    disk = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "disk")), **opts)
    assert made == [1]
    assert differing_files(mem.out, disk.out) == []


def test_failed_spill_never_read_warns(community, base_run, tmp_path, monkeypatch):
    """A chunk-table write-back that fails while nothing reads the file (no mixed contig in the synthetic community):
    the outputs stand, and one warning says the folder had a write error."""
    import errno
    import epirefine.pipeline as P
    monkeypatch.setattr(P, "SPILL_MIN_BYTES", 1)

    def fail(fd, *args):
        raise OSError(errno.EIO, "Input/output error")
    monkeypatch.setattr(P, "drop_pages", fail)
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")))
    assert run.log.count("chunk tables' temporary file") == 1 and "not needed for the results" in run.log
    assert differing_files(base_run.out, run.out) == []
