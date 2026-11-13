"""The small example dataset in tests/data/example_community (docs/getting-started.md, "Try it on the example data").

It is the synthetic community of tests/synth.py at seed 1 (the golden-snapshot community), with the assembly and
the pileup gzipped: synthetic only, no real reads. These tests check that the shipped files are exactly what the
generator writes (so the example cannot drift from the community the other tests use) and that the command shown in
the docs runs end to end and gives the decisions of the reference run.
"""
import gzip
from pathlib import Path

from epirefine.cli import main
from helpers import DECISION_FILES, Run, logged

EXAMPLE = Path(__file__).parent / "data" / "example_community"
FILES = {"assembly.fasta.gz": "assembly", "pileup.bed.gz": "pileup", "contig_bin.tsv": "contig_bin",
         "bin-motifs.tsv": "motifs"}


def _content(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def test_example_files_are_the_synthetic_community(community):
    """Regenerate with tests/synth.py (make_community(dir, seed=1)), then `gzip -9 -n` the assembly and pileup."""
    assert sorted(p.name for p in EXAMPLE.iterdir()) == sorted(FILES)
    for name, attr in FILES.items():
        assert _content(EXAMPLE / name) == Path(getattr(community, attr)).read_bytes(), name
    assert sum((EXAMPLE / n).stat().st_size for n in FILES) < 2_000_000   # small enough to keep in git


def test_example_runs_end_to_end(base_run, tmp_path):
    """The command of docs/getting-started.md, on one worker: the same decisions as the reference run of the
    uncompressed community, and the outcome the docs describe."""
    out = tmp_path / "example_out"
    run = Run(out, logged(main, ["run", "--assembly", str(EXAMPLE / "assembly.fasta.gz"),
                                 "--pileup", str(EXAMPLE / "pileup.bed.gz"),
                                 "--contig-bin", str(EXAMPLE / "contig_bin.tsv"),
                                 "--motifs", str(EXAMPLE / "bin-motifs.tsv"), "--out", str(out), "--workers", "1"]))
    for name in DECISION_FILES:
        assert (out / name).read_bytes() == (base_run.out / name).read_bytes(), name
    # what getting-started.md says to expect
    assert run.log_counts() == {"flagged": 6, "testable": 42, "included": 6, "unbinned": 13, "moved": 4}
    status = run.table("contig_decisions.tsv")["status"].value_counts()
    assert dict(status.iter_rows()) == {"kept": 35, "included": 6, "moved": 4, "contaminant": 2, "unbinned": 5,
                                        "redundant": 1, "low_coverage": 1, "too_short": 1, "sole_contig_of_bin": 1}
