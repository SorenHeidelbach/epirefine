"""The site cache: rebuilt when the motif table or the pileup changes, and never reused with part files
that another, interrupted run left under an old manifest."""
import contextlib
import io
import json
import os
import shutil

import pytest

import synth
from helpers import DECISION_FILES, Run, assert_same_outputs, copy_cache, logged, run_pipeline


class _Stop(Exception):
    """Raised by the patched pipeline to end a run early."""


def _until_cache(monkeypatch, **kw) -> Run:
    """Run the pipeline only up to the site cache (pass 1 or its reuse, and the manifest), then stop: the
    first step after the cache raises. Enough to see whether the cache was reused or rebuilt, in a
    fraction of a full run."""
    from epirefine import pipeline

    def stop(*a, **k):
        raise _Stop
    buf = io.StringIO()
    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "noisy_groups", stop)
        with pytest.raises(_Stop), contextlib.redirect_stdout(buf):
            pipeline.run(workers=1, **kw)
    return Run(kw["out"], buf.getvalue())


def _private_copy(community, directory) -> synth.Community:
    """The community's input files copied to a directory the test may change."""
    import dataclasses
    directory.mkdir()
    paths = {}
    for name in ("assembly", "pileup", "contig_bin", "motifs"):
        paths[name] = directory / getattr(community, name).name
        shutil.copy(getattr(community, name), paths[name])
    return dataclasses.replace(community, directory=directory, **paths)


def _manifest(out) -> dict:
    return json.loads((out / "cache" / "manifest.json").read_text())


def test_changed_motif_table_rebuilds_the_cache(community, base_run, tmp_path, monkeypatch):
    more = tmp_path / "bin-motifs.tsv"
    more.write_text(community.motifs.read_text() + "binA\tm\tGGCC\t2\t900\t100\tpalindrome\tNA\tNA\tNA\tNA\n")
    out = copy_cache(base_run, tmp_path / "out")
    before = _manifest(out)
    run = _until_cache(monkeypatch, **{**community.args(out), "motifs_path": str(more)})
    assert run.ran_pileup_pass and not run.reused_cache
    after = _manifest(out)
    assert after["inputs"]["motifs"] == before["inputs"]["motifs"] + ["GGCC_m-2"]
    assert after["inputs"]["pileup"] == before["inputs"]["pileup"]


def test_unchanged_inputs_reuse_the_cache_wherever_it_lies(community, base_run, tmp_path, monkeypatch):
    """Same inputs, the cache copied to another directory and passed with --cache-dir: reused."""
    cache = tmp_path / "elsewhere"
    shutil.copytree(base_run.out / "cache", cache)
    run = _until_cache(monkeypatch, **community.args(tmp_path / "out", cache_dir=str(cache)))
    assert run.reused_cache and not run.ran_pileup_pass
    assert not (tmp_path / "out" / "cache").exists()


def test_changed_pileup_rebuilds_the_cache(community, tmp_path, monkeypatch):
    """Rows of one contig removed from the pileup (size changes): the next run re-reads the pileup and that
    contig has no sites."""
    com = _private_copy(community, tmp_path / "in")
    out = tmp_path / "out"
    first = _until_cache(monkeypatch, **com.args(out))
    assert first.ran_pileup_pass
    again = _until_cache(monkeypatch, **com.args(out))
    assert again.reused_cache
    gone = community.orphans[0]
    rows = [r for r in com.pileup.read_text().splitlines(keepends=True) if not r.startswith(gone + "\t")]
    com.pileup.write_text("".join(rows))
    run = run_pipeline(**com.args(out, do_calibrate=False))
    assert run.ran_pileup_pass and not run.reused_cache
    row = run.decisions[gone]
    assert row["n_sites"] == 0 and row["status"] == "low_coverage"


@pytest.mark.xfail(strict=True, reason="the cache key holds the pileup's path, size and mtime in whole seconds, no content "
                   "fingerprint: a pileup replaced by one of the same size and mtime is not noticed and the old sites "
                   "are reused (review 08, section 3.2)")
def test_changed_pileup_of_same_size_and_mtime_rebuilds_the_cache(community, tmp_path, monkeypatch):
    com = _private_copy(community, tmp_path / "in")
    out = tmp_path / "out"
    _until_cache(monkeypatch, **com.args(out))
    st = com.pileup.stat()
    target = next(iter(community.missing))
    data = com.pileup.read_bytes()
    changed = b"".join(r.replace(b"\t+\t", b"\t-\t") if r.startswith(target.encode() + b"\t") else r
                       for r in data.splitlines(keepends=True))
    assert changed != data and len(changed) == len(data)
    com.pileup.write_bytes(changed)
    os.utime(com.pileup, ns=(st.st_atime_ns, st.st_mtime_ns))
    run = _until_cache(monkeypatch, **com.args(out))
    assert run.ran_pileup_pass and not run.reused_cache


def _interrupted_rebuild(community, out, monkeypatch, tmp_path, after_parts: bool):
    """A second run on the same cache with another motif table (the first motif row dropped: one group fewer,
    the others renumbered), killed during the pileup pass (no part written yet) or just after it (all parts
    written, manifest not yet)."""
    from epirefine import pipeline
    header, _, *rows = community.motifs.read_text().splitlines(keepends=True)
    fewer = tmp_path / "fewer-motifs.tsv"
    fewer.write_text(header + "".join(rows))
    real = pipeline.extract_sites

    def killed(*a, **kw):
        if after_parts:
            real(*a, **kw)
        raise _Stop
    with monkeypatch.context() as mp:
        mp.setattr(pipeline, "extract_sites", killed)
        with pytest.raises(_Stop):
            logged(pipeline.run, **{**community.args(out, workers=1), "motifs_path": str(fewer)})


def _rerun_is_safe(community, base_run, out):
    """Repeating the first run's command on that cache must rebuild the cache or stop with an error; if it
    completes, its decisions must be those of the first run."""
    try:
        rerun = run_pipeline(**community.args(out))
    except Exception:
        return  # loud: the user sees that the cache is unusable
    assert_same_outputs(base_run.out, rerun.out, DECISION_FILES)
    assert not rerun.reused_cache, "the cache of an interrupted rebuild was reused"


def test_interrupted_rebuild_is_not_reused(community, base_run, tmp_path, monkeypatch):
    """A rebuild for other motifs killed between the pileup pass and the manifest: the manifest is deleted before
    a rebuild starts, so the first command rebuilds instead of reusing the other run's part files (review 01
    finding 2, review 08 F7; a strict xfail until 0.11)."""
    out = copy_cache(base_run, tmp_path / "out")
    _interrupted_rebuild(community, out, monkeypatch, tmp_path, after_parts=True)
    _rerun_is_safe(community, base_run, out)


def test_rebuild_killed_during_the_pileup_pass_is_not_reused(community, base_run, tmp_path, monkeypatch):
    """Killed before any part was rewritten: the old parts are gone, so the stale manifest cannot be used
    silently (today the rerun stops with a missing-file error; rebuilding would be fine too)."""
    out = copy_cache(base_run, tmp_path / "out")
    _interrupted_rebuild(community, out, monkeypatch, tmp_path, after_parts=False)
    assert not list((out / "cache" / "sites").glob("*.parquet"))
    _rerun_is_safe(community, base_run, out)
