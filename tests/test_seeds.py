"""Random streams: the motif-grouping sample and the calibration pieces have their own generators.

--seed draws the grouping sample, which is part of the site-cache key; --calibration-seed (default: --seed) draws
the calibration pieces and nothing else. Up to 0.11.1 one generator drew both, so a new seed rebuilt the site
cache of an assembly larger than the grouping sample. At the default the grouping and the pieces must stay those
of 0.11.1: tests/data/seed_streams_0111.json holds them, written by the 0.11.1 code (release/0.12 at 9e08bf9)
through `_until_pieces` below on the synthetic community with a grouping sample of a third of the assembly (so that
the sample is drawn and uses the generator)."""
import contextlib
import hashlib
import io
import json
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from helpers import run_pipeline

REFERENCE = Path(__file__).parent / "data" / "seed_streams_0111.json"


class _Stop(Exception):
    pass


def _until_pieces(out, community, **kw) -> dict:
    """Run the pipeline up to the calibration pieces (the site cache is built or reused on the way), then stop.
    Returns the grouping in the site-cache manifest, a hash of the pieces and whether the cache was reused."""
    from epirefine import pipeline
    seen = {}
    real = pipeline.select_fragments

    def capture(*a, **k):
        seen["frags"] = real(*a, **k)
        raise _Stop
    buf = io.StringIO()
    with mock.patch.object(pipeline, "select_fragments", capture), contextlib.redirect_stdout(buf):
        try:
            pipeline.run(**community.args(out), workers=1, **kw)
        except _Stop:
            pass
    man = json.loads((Path(kw.get("cache_dir") or Path(out) / "cache") / "manifest.json").read_text())["inputs"]
    frags = seen["frags"].select("unit", "contig", "bin_idx", "tile", "length", "start", "end")
    return {"group_of": man["group_of"], "motif_rank": man["motif_rank"], "n_fragments": frags.height,
            "fragments_sha256": hashlib.sha256(frags.write_csv().encode()).hexdigest(),
            "reused_cache": "reusing site cache" in buf.getvalue()}


def _sample_bp(community) -> float:
    """A third of the assembly: the grouping sample is drawn at random."""
    return sum(len(ln) for ln in community.assembly.read_text().splitlines() if not ln.startswith(">")) / 3


def _write_reference(community, directory):  # pragma: no cover - run once with the 0.11.1 code
    ref = {}
    for seed in (1, 2):
        r = _until_pieces(Path(directory) / f"seed{seed}", community, seed=seed, group_sample_bp=_sample_bp(community))
        ref[str(seed)] = {k: v for k, v in r.items() if k != "reused_cache"}
    REFERENCE.write_text(json.dumps(ref, indent=1) + "\n")


@pytest.mark.parametrize("seed", [1, 2, 3, 17])
@pytest.mark.parametrize("n", [1, 37, 5000])
def test_calibration_generator_continues_the_single_stream_of_0111(seed, n):
    """With the calibration seed equal to the seed, the calibration generator is in the state the one generator of
    0.11.1 had after the grouping sample: the same pieces follow."""
    from epirefine.pipeline import _calibration_rng, _sample_names
    lengths = {f"c{i}": 1000 + i for i in range(n)}
    old = np.random.default_rng(seed)
    sample = _sample_names(lengths, list(lengths), 0.4 * sum(lengths.values()), old)
    new_group = np.random.default_rng(seed)
    assert _sample_names(lengths, list(lengths), 0.4 * sum(lengths.values()), new_group) == sample
    cal = _calibration_rng(seed, n, grouping_draws=True)
    assert cal.bit_generator.state == old.bit_generator.state
    # a grouping sample that takes everything draws nothing
    assert (_calibration_rng(seed, n, grouping_draws=False).bit_generator.state
            == np.random.default_rng(seed).bit_generator.state)


@pytest.mark.parametrize("seed", [1, 2])
def test_grouping_and_pieces_at_the_default_are_those_of_0111(community, tmp_path, seed):
    ref = json.loads(REFERENCE.read_text())[str(seed)]
    got = _until_pieces(tmp_path / "out", community, seed=seed, group_sample_bp=_sample_bp(community))
    assert got["group_of"] == ref["group_of"]
    assert got["motif_rank"] == ref["motif_rank"]
    assert (got["n_fragments"], got["fragments_sha256"]) == (ref["n_fragments"], ref["fragments_sha256"])


def test_calibration_seed_changes_the_pieces_and_keeps_grouping_and_cache(community, tmp_path):
    """Another calibration seed on the cache of the default run: the cache is reused, the grouping is that of
    0.11.1 at --seed 1, the pieces are others; an explicit calibration seed equal to the seed is the default."""
    ref = json.loads(REFERENCE.read_text())["1"]
    bp = _sample_bp(community)
    first = _until_pieces(tmp_path / "a", community, seed=1, group_sample_bp=bp)
    cache = tmp_path / "cache"
    shutil.copytree(tmp_path / "a" / "cache", cache)
    other = _until_pieces(tmp_path / "b", community, seed=1, calibration_seed=7, group_sample_bp=bp,
                          cache_dir=str(cache))
    assert other["reused_cache"]
    assert other["group_of"] == ref["group_of"] and other["motif_rank"] == ref["motif_rank"]
    assert other["fragments_sha256"] != first["fragments_sha256"]
    same = _until_pieces(tmp_path / "c", community, seed=1, calibration_seed=1, group_sample_bp=bp,
                         cache_dir=str(cache))
    assert same["reused_cache"] and same["fragments_sha256"] == first["fragments_sha256"]


def test_seed_still_moves_the_pieces_when_no_calibration_seed_is_given(community, tmp_path):
    """--seed alone keeps its meaning of 0.11.1 for the pieces (both streams follow it)."""
    bp = _sample_bp(community)
    a = _until_pieces(tmp_path / "a", community, seed=1, group_sample_bp=bp)
    b = _until_pieces(tmp_path / "b", community, seed=1, calibration_seed=2, group_sample_bp=bp)
    c = _until_pieces(tmp_path / "c", community, seed=2, calibration_seed=2, group_sample_bp=bp)
    assert a["fragments_sha256"] != b["fragments_sha256"]
    # the pieces follow the calibration seed alone, whatever seed drew the grouping sample
    assert b["fragments_sha256"] == c["fragments_sha256"]


def test_params_record_the_calibration_seed(base_run):
    p = json.loads((base_run.out / "params.json").read_text())
    assert p["seed"] == 1 and p["calibration_seed"] == 1


def test_cli_passes_the_calibration_seed(community, tmp_path):
    from epirefine import cli
    seen = {}
    with mock.patch("epirefine.pipeline.run", lambda *a, **k: seen.update(k)):
        cli.main(community.cli_args(tmp_path / "o", "--calibration-seed", 5))
    assert seen["seed"] == 1 and seen["calibration_seed"] == 5
    with mock.patch("epirefine.pipeline.run", lambda *a, **k: seen.update(k)):
        cli.main(community.cli_args(tmp_path / "o"))
    assert seen["calibration_seed"] is None


def test_calibration_seed_reproduces_through_run_pipeline(community, base_run, tmp_path):
    """An explicit calibration seed equal to the seed gives the decisions of the default run."""
    from helpers import DECISION_FILES, assert_same_outputs, copy_cache
    out = copy_cache(base_run, tmp_path / "out")
    run = run_pipeline(**community.args(out), calibration_seed=1)
    assert run.reused_cache
    assert_same_outputs(base_run.out, run.out, DECISION_FILES)
