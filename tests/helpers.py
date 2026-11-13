"""Helpers for the end-to-end tests: run the pipeline with its log captured, read and compare outputs."""
from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# every file docs/outputs.md lists for a default run (motif_fit.tsv needs --motif-fit)
OUTPUT_FILES = [
    "new_contig_bin.tsv", "contig_decisions.tsv", "decontaminated_contig_bin.tsv", "bin_contamination.tsv",
    "include_contigs.tsv", "bin_evidence.tsv", "host_profiles.tsv", "motif_groups.tsv", "motif_noise.tsv",
    "group_weights.tsv", "detection_model.tsv", "host_chunk_outliers.tsv", "params.json",
    "calibration/summary.tsv", "calibration/reliability.tsv", "calibration/temperature_fit.tsv",
    "calibration/temperature_by_length.tsv", "calibration/tau_fit.tsv", "calibration/tau_groups.tsv",
    "calibration/fragments.tsv", "calibration/assignability.tsv", "calibration/lookalikes.tsv",
    "calibration/false_flag_rate.tsv", "calibration/reliability_deployed.tsv",
    "cache/manifest.json",
]
# files that must not change with the worker count, the cache state or the pileup's compression
INVARIANT_FILES = [f for f in OUTPUT_FILES if not f.startswith("cache/")]
# the decisions themselves: these must also survive a reordering of the input tables
DECISION_FILES = ["contig_decisions.tsv", "new_contig_bin.tsv", "decontaminated_contig_bin.tsv",
                  "bin_contamination.tsv", "include_contigs.tsv"]


def read_tsv(path) -> pl.DataFrame:
    """A TSV output with empty text fields as "" (polars reads them as null)."""
    df = pl.read_csv(path, separator="\t")
    return df.with_columns(pl.col(pl.Utf8).fill_null(""))


@dataclass
class Run:
    out: Path
    log: str

    def table(self, name: str) -> pl.DataFrame:
        return read_tsv(self.out / name)

    @property
    def decisions(self) -> dict[str, dict]:
        """contig -> its row of contig_decisions.tsv."""
        return {r["contig"]: r for r in self.table("contig_decisions.tsv").iter_rows(named=True)}

    @property
    def reused_cache(self) -> bool:
        return "reusing site cache" in self.log

    @property
    def ran_pileup_pass(self) -> bool:
        return "pileup pass" in self.log

    def log_counts(self) -> dict[str, int]:
        """The numbers of the final 'refinement:' log line."""
        m = re.search(r"(\d+) of (\d+) testable binned contigs flagged, (\d+) of (\d+) unbinned included, "
                      r"(\d+) contaminants moved", self.log)
        keys = ["flagged", "testable", "included", "unbinned", "moved"]
        return dict(zip(keys, map(int, m.groups())))


def logged(fn, *args, **kw) -> str:
    """Call fn and return what it printed (the pipeline logs with print)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kw)
    return buf.getvalue()


def run_pipeline(out, workers: int = 1, **kw) -> Run:
    """epirefine.pipeline.run with its log captured. One worker by default: the test machine is shared and
    each worker process costs about a second to start."""
    from epirefine.pipeline import run
    return Run(Path(out), logged(run, out=str(out), workers=workers, **kw))


def copy_cache(src: Run, out) -> Path:
    """Start a new output directory with a copy of another run's site cache."""
    out = Path(out)
    shutil.copytree(src.out / "cache", out / "cache")
    return out


# entries of params.json that describe the run and not its result (provenance, 0.11): they differ between two
# runs by construction (start time, command line, paths, worker count, time and memory used)
RUN_SPECIFIC_PARAMS = ("command_line", "start_time", "inputs", "out", "cache_dir", "workers", "python", "elapsed_s",
                       "peak_rss_gb", "peak_rss_worker_gb",
                       "threads", "kernels",  # what ran where (0.12): differ with --workers and EPIREFINE_RS by design
                       "write_motif_fit")  # an extra output file, asked for by the command-line test run only


def _comparable_lines(path: Path) -> list[str]:
    """The lines of an output file to compare; params.json without its run-specific entries."""
    if path.name == "params.json":
        params = json.loads(path.read_text())
        return json.dumps({k: v for k, v in params.items() if k not in RUN_SPECIFIC_PARAMS}, indent=1).splitlines()
    return path.read_text().splitlines()


def differing_files(a: Path, b: Path, files=INVARIANT_FILES) -> list[str]:
    """Names of the files whose content differs between two output directories, with the first differing line."""
    out = []
    for f in files:
        x, y = _comparable_lines(Path(a) / f), _comparable_lines(Path(b) / f)
        if x != y:
            i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            out.append(f"{f} line {i + 1}:\n    {x[i] if i < len(x) else '<end>'}\n    {y[i] if i < len(y) else '<end>'}")
    return out


def assert_same_outputs(a: Path, b: Path, files=INVARIANT_FILES):
    diff = differing_files(a, b, files)
    assert not diff, "outputs differ:\n" + "\n".join(diff)


def assert_same_outputs_to_rounding(a: Path, b: Path, rel: float = 1e-9):
    """Two runs that differ in the last bits of their arithmetic (the compiled model kernels against numpy):
    every output table equal as written (four decimals), the numbers of params.json (written in full) to `rel`."""
    import pytest
    assert_same_outputs(a, b, [f for f in INVARIANT_FILES if f != "params.json"])
    pa, pb = (json.loads((Path(d) / "params.json").read_text()) for d in (a, b))
    for key in pa:
        if key in RUN_SPECIFIC_PARAMS:
            continue
        if isinstance(pa[key], float):
            assert pb[key] == pytest.approx(pa[key], rel=rel, abs=1e-12), key
        elif isinstance(pa[key], dict):  # e.g. relatives_statistic: its floats to `rel`, the rest as written
            assert pa[key].keys() == pb[key].keys(), key
            for k, v in pa[key].items():
                if isinstance(v, float):
                    assert pb[key][k] == pytest.approx(v, rel=rel, abs=1e-12), (key, k)
                else:
                    assert pb[key][k] == v, (key, k)
        else:
            assert pa[key] == pb[key], key
