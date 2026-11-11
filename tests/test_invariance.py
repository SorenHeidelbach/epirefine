"""Things that must not change the outputs: the number of workers, a reused site cache, the compression of
the pileup, and the row order of the contig_bin and motif tables. Each run is compared file by file with
the session's reference run (one worker, cold cache, plain pileup)."""
import json
import random

from helpers import DECISION_FILES, INVARIANT_FILES, assert_same_outputs, copy_cache, run_pipeline
from test_bgzf import _bgzf


def test_workers_1_and_3_give_identical_outputs(base_run, cli_run):
    """Three workers on 12 byte ranges of the pileup (several site-cache parts) against one worker on one."""
    n_parts = lambda run: len(json.loads((run.out / "cache" / "manifest.json").read_text())["parts"])
    assert n_parts(base_run) == 1 and n_parts(cli_run) > 1
    assert_same_outputs(base_run.out, cli_run.out, INVARIANT_FILES)


def test_rerun_on_the_site_cache_gives_identical_outputs(community, base_run, tmp_path):
    assert base_run.ran_pileup_pass and not base_run.reused_cache
    rerun = run_pipeline(**community.args(copy_cache(base_run, tmp_path / "out")))
    assert rerun.reused_cache and not rerun.ran_pileup_pass
    assert_same_outputs(base_run.out, rerun.out, INVARIANT_FILES)


def test_bgzipped_pileup_gives_identical_outputs(community, base_run, tmp_path, many_ranges):
    from epirefine.sites import is_bgzf
    gz = tmp_path / "pileup.bed.gz"
    gz.write_bytes(_bgzf(community.pileup.read_bytes(), 65280))
    assert is_bgzf(gz)
    run = run_pipeline(**{**community.args(tmp_path / "out"), "pileup": str(gz)}, workers=2)
    assert "pileup pass: 8 byte ranges on 2 workers" in run.log  # read block-wise, not as one gzip stream
    assert_same_outputs(base_run.out, run.out, INVARIANT_FILES)


def test_row_order_of_contig_bin_does_not_matter(community, base_run, tmp_path):
    rows = community.contig_bin.read_text().splitlines(keepends=True)
    random.Random(3).shuffle(rows)
    shuffled = tmp_path / "contig_bin.tsv"
    shuffled.write_text("".join(rows))
    assert shuffled.read_text() != community.contig_bin.read_text()
    run = run_pipeline(**{**community.args(copy_cache(base_run, tmp_path / "out")), "contig_bin": str(shuffled)})
    assert run.reused_cache  # the bins are applied after the cache
    assert_same_outputs(base_run.out, run.out, INVARIANT_FILES)


def test_row_order_of_the_motif_table_does_not_matter(community, base_run, tmp_path):
    """Motif groups are numbered by their first motif, so host_profiles.tsv and the other per-group tables may
    list the groups in another order; the decisions and their probabilities must not move."""
    header, *rows = community.motifs.read_text().splitlines(keepends=True)
    random.Random(3).shuffle(rows)
    shuffled = tmp_path / "bin-motifs.tsv"
    shuffled.write_text(header + "".join(rows))
    assert shuffled.read_text() != community.motifs.read_text()
    run = run_pipeline(**{**community.args(tmp_path / "out"), "motifs_path": str(shuffled)})
    assert_same_outputs(base_run.out, run.out, DECISION_FILES)
