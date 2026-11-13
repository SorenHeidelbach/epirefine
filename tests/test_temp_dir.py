"""--temp-dir (where the temporary files of the two large tables go) and --tables-in-memory (the memory path of
0.11: contig tables in memory, the chunk tables' file left in the page cache)."""
import json
import os
import tempfile

import pytest

from helpers import copy_cache, differing_files, logged, run_pipeline


def _force_files(monkeypatch):
    """Both tables go to files even on the synthetic community (thresholds 0)."""
    import epirefine.pipeline as P
    from epirefine import sites
    monkeypatch.setattr(sites, "TARGETS_DISK_MIN_BYTES", 0)
    monkeypatch.setattr(P, "SPILL_MIN_BYTES", 0)


def _record_temp_files(monkeypatch):
    """The folder of every temporary file the run makes, and where the open file really lies (/proc)."""
    made = []
    real = tempfile.TemporaryFile

    def recording(*a, **k):
        fh = real(*a, **k)
        where = None
        try:
            where = os.readlink(f"/proc/self/fd/{fh.fileno()}")
        except OSError:
            pass
        made.append((str(k.get("dir")), where, fh))
        return fh
    monkeypatch.setattr(tempfile, "TemporaryFile", recording)
    return made


def test_temp_dir_holds_the_table_files(community, base_run, tmp_path, monkeypatch):
    """With --temp-dir every temporary file (the writability check, the contig tables, the chunk tables) is made
    in that folder, created if missing, none in the output folder; all are closed (removed) when the run ends, and
    the outputs are those of the reference run."""
    from epirefine import sites
    _force_files(monkeypatch)
    made = _record_temp_files(monkeypatch)
    rows = []
    real_init = sites.RowFile.__init__
    monkeypatch.setattr(sites.RowFile, "__init__", lambda self, *a, **k: (rows.append(self), real_init(self, *a, **k))[1])
    tmp = tmp_path / "scratch" / "tables"  # does not exist yet
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")), temp_dir=str(tmp))
    assert len(rows) == 1 and len(made) >= 3  # the check, the contig tables and the chunk tables
    assert {d for d, _, _ in made} == {str(tmp)}
    if all(w is not None for _, w, _ in made):  # Linux: the open file lies in the folder
        assert all(w.startswith(str(tmp)) for _, w, _ in made)
    assert rows[0]._fh.closed and made[0][2].closed
    assert list(tmp.iterdir()) == [] and not [p for p in run.out.iterdir() if p.name.startswith("tmp")]
    assert json.loads((run.out / "params.json").read_text())["temp_dir"] == str(tmp.resolve())
    assert differing_files(base_run.out, run.out) == []


def test_temp_dir_files_removed_when_the_run_fails(community, base_run, tmp_path, monkeypatch):
    import epirefine.pipeline as P
    from epirefine import sites
    _force_files(monkeypatch)
    rows = []
    real_init = sites.RowFile.__init__
    monkeypatch.setattr(sites.RowFile, "__init__", lambda self, *a, **k: (rows.append(self), real_init(self, *a, **k))[1])

    def broken(*a, **k):
        raise RuntimeError("calibration failed")
    monkeypatch.setattr(P, "calibrate", broken)
    tmp = tmp_path / "tables"
    with pytest.raises(RuntimeError, match="calibration failed"):
        P.run(**community.args(copy_cache(base_run, tmp_path / "out")), workers=1, temp_dir=str(tmp))
    assert len(rows) == 1 and rows[0]._fh.closed and rows[0].directory == tmp
    assert list(tmp.iterdir()) == []


@pytest.mark.parametrize("how", ["a file in the way", "read-only folder"])
def test_unwritable_temp_dir_stops_before_any_work(community, tmp_path, how, capsys):
    """A folder that cannot be made, or in which no file can be made, stops the run at once with one line."""
    from epirefine.cli import main
    if how == "a file in the way":
        (tmp_path / "f").write_text("")
        tmp = tmp_path / "f" / "tables"
    else:
        if os.geteuid() == 0:
            pytest.skip("root can write into a read-only folder")
        tmp = tmp_path / "ro"
        tmp.mkdir()
        tmp.chmod(0o500)
    try:
        with pytest.raises(SystemExit) as e:
            logged(main, community.cli_args(tmp_path / "out", "--workers", 1, "--temp-dir", tmp))
    finally:
        if how != "a file in the way":
            tmp.chmod(0o700)
    assert e.value.code == 2
    err = capsys.readouterr().err.strip()
    assert err.startswith(f"epirefine: error: --temp-dir {tmp}: cannot write a file there") and "\n" not in err
    assert not (tmp_path / "out" / "cache").exists()  # stopped before the pileup pass


def test_tables_in_memory_takes_the_memory_path(community, base_run, tmp_path, monkeypatch):
    """--tables-in-memory: no contig-table file, the weights multiplied into the finished table, the chunk tables'
    file neither written back nor dropped (the path of 0.11), even where the thresholds would send both tables
    to files; the outputs are the reference run's."""
    import epirefine.pipeline as P
    from epirefine import sites
    _force_files(monkeypatch)
    calls, drops, spills, writebacks = [], [], [], []
    real_ut, real_spill, real_wb = P.unit_tables, P._spill, P._SpillWriteback.__init__
    monkeypatch.setattr(P, "unit_tables", lambda *a, **k: (calls.append(k), real_ut(*a, **k))[1])
    monkeypatch.setattr(P, "_spill", lambda *a, **k: (spills.append(k), real_spill(*a, **k))[1])
    monkeypatch.setattr(P, "drop_pages", lambda *a, **k: drops.append(a))
    monkeypatch.setattr(P._SpillWriteback, "__init__", lambda self, *a: (writebacks.append(a), real_wb(self, *a))[1])
    monkeypatch.setattr(sites.RowFile, "__init__", lambda *a, **k: pytest.fail("a contig-table file was made"))
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")), tables_in_memory=True)
    assert calls[0]["targets_dir"] is None and calls[0]["target_weight"] is None
    assert spills == [{"keep_pages": True}] and not drops and not writebacks
    assert json.loads((run.out / "params.json").read_text())["tables_in_memory"] is True
    assert differing_files(base_run.out, run.out) == []


def test_tables_in_memory_with_group_weights(community, base_run, tmp_path, monkeypatch):
    """With a down-weighted group the in-memory path (weights on the finished table) and the disk path (weights per
    row as it is written) give the same outputs."""
    import epirefine.pipeline as P
    real = P.redundancy_weights

    def halved(*a, **k):
        w, tab = real(*a, **k)
        w = w.copy()
        w[0] = 0.5
        return w, tab
    monkeypatch.setattr(P, "redundancy_weights", halved)
    _force_files(monkeypatch)
    disk = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "disk")))
    mem = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "mem")), tables_in_memory=True)
    assert differing_files(disk.out, mem.out) == []
    assert differing_files(base_run.out, mem.out) != []  # the weight did change the run


def test_defaults_unchanged(community, base_run, tmp_path, monkeypatch):
    """Without the options the tables' files go to the output folder and the disk path is on (above the
    thresholds); params.json records both options."""
    import epirefine.pipeline as P
    from epirefine.cli import build_parser
    a = build_parser().parse_args(["run", "--assembly", "a", "--pileup", "p", "--contig-bin", "c", "--motifs", "m",
                                   "--out", "o"])
    assert a.temp_dir is None and a.tables_in_memory is False
    p = json.loads((base_run.out / "params.json").read_text())
    assert p["temp_dir"] is None and p["tables_in_memory"] is False
    _force_files(monkeypatch)
    made = _record_temp_files(monkeypatch)
    calls = []
    real_ut = P.unit_tables
    monkeypatch.setattr(P, "unit_tables", lambda *a, **k: (calls.append(k), real_ut(*a, **k))[1])
    out = copy_cache(base_run, tmp_path / "out")
    run_pipeline(**community.args(out))
    assert calls[0]["targets_dir"] == out
    assert len(made) == 2 and {d for d, _, _ in made} == {str(out)}  # no check file: the folder was just written


def test_rerun_replays_the_options(community, base_run, tmp_path):
    from epirefine.cli import main
    first = copy_cache(base_run, tmp_path / "first")
    logged(main, community.cli_args(first, "--workers", 1, "--cache-dir", first / "cache", "--temp-dir",
                                    tmp_path / "t1", "--tables-in-memory"))
    logged(main, ["rerun", "--params", str(first / "params.json"), "--out", str(tmp_path / "again")])
    p = json.loads((tmp_path / "again" / "params.json").read_text())
    assert p["tables_in_memory"] is True and p["temp_dir"] == str((tmp_path / "t1").resolve())
    logged(main, ["rerun", "--params", str(first / "params.json"), "--out", str(tmp_path / "third"),
                  "--temp-dir", str(tmp_path / "t2")])
    p = json.loads((tmp_path / "third" / "params.json").read_text())
    assert p["temp_dir"] == str((tmp_path / "t2").resolve()) and (tmp_path / "t2").is_dir()


def test_rerun_of_a_run_without_temp_dir_uses_the_new_output_folder(base_run, tmp_path):
    from epirefine.runinfo import rerun_arguments
    p = json.loads((base_run.out / "params.json").read_text())
    kw = rerun_arguments(p, tmp_path / "o", log=lambda m: None)
    assert kw["temp_dir"] is None and kw["tables_in_memory"] is False
    del p["temp_dir"], p["tables_in_memory"]  # a params.json written before the options
    kw = rerun_arguments(p, tmp_path / "o", log=lambda m: None)
    assert kw["temp_dir"] is None and "tables_in_memory" not in kw


def test_warning_for_a_temp_dir_in_memory(community, base_run, tmp_path, monkeypatch):
    """Table files on tmpfs take memory: the log says so, once, when the first file is made there."""
    import epirefine.pipeline as P
    _force_files(monkeypatch)
    monkeypatch.setattr(P, "memory_file_system", lambda path: "tmpfs")
    run = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")))
    assert run.log.count("are on tmpfs, a file system in memory") == 1
    quiet = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "mem")), tables_in_memory=True)
    assert "a file system in memory" not in quiet.log


def test_memory_file_system():
    from epirefine.pipeline import memory_file_system
    mounts = {}
    try:
        with open("/proc/self/mounts") as f:
            for line in f:
                x = line.split()
                mounts[x[1]] = x[2]
    except OSError:
        pytest.skip("no /proc/self/mounts")
    if mounts.get("/dev/shm") == "tmpfs":
        assert memory_file_system("/dev/shm") == "tmpfs"
        assert memory_file_system("/dev/shm/not/yet/made") == "tmpfs"
    if mounts.get("/") not in ("tmpfs", "ramfs"):
        assert memory_file_system("/") is None


def test_rerun_keeps_the_table_options_beside_removed_options(base_run, tmp_path):
    """A record with the table options and the removed 0.12 options at their old defaults (a 0.11.1-style record
    as the merge of perf/lt-disk with the U12 removals sees it) replays the table options; a removed option at
    another value stops the rerun whatever the table options say."""
    from epirefine.io import InputError
    from epirefine.runinfo import REMOVED_OPTIONS, rerun_arguments
    p = json.loads((base_run.out / "params.json").read_text())
    p.update({k: d for k, (_, d) in REMOVED_OPTIONS.items()})
    p.update(temp_dir=str(tmp_path / "t1"), tables_in_memory=True)
    kw = rerun_arguments(p, tmp_path / "o", log=lambda m: None)
    assert kw["temp_dir"] == str(tmp_path / "t1") and kw["tables_in_memory"] is True
    assert not set(REMOVED_OPTIONS) & set(kw)  # run() no longer takes them
    kw = rerun_arguments(p, tmp_path / "o", log=lambda m: None, temp_dir=str(tmp_path / "t2"))
    assert kw["temp_dir"] == str(tmp_path / "t2")
    p.update(unlisted_genome=True)
    with pytest.raises(InputError, match="--unlisted-genome"):
        rerun_arguments(p, tmp_path / "o", log=lambda m: None)
