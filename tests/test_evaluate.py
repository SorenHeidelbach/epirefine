import sys
from pathlib import Path

import polars as pl
import pytest

# the benchmark evaluator lives in the repository's benchmarks/, which the source distribution leaves out
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
evaluate = pytest.importorskip("evaluate_refinement").evaluate


def test_unjudged_contigs_are_left_out(tmp_path):
    # a: flagged contaminant; b: missed contaminant without methylation evidence; c: low coverage
    # contaminant; d: correct member
    pl.DataFrame({"contig": ["a", "b", "c", "d"], "length": [5000] * 4, "true_bin": ["y", "y", "y", "x"],
                  "given_bin": ["x"] * 4, "category": ["contaminant", "contaminant", "contaminant", "member"]}
                 ).write_csv(tmp_path / "truth.tsv", separator="\t")
    pl.DataFrame({"contig": ["a", "b", "c", "d"], "status": ["contaminant", "no_methylation_evidence", "low_coverage", "kept"]}
                 ).write_csv(tmp_path / "dec.tsv", separator="\t")
    pl.DataFrame({"contig": ["a"], "bin": ["x"]}).write_csv(tmp_path / "contam.tsv", separator="\t")
    pl.DataFrame({"contig": ["b", "c", "d"], "bin": ["x", "x", "x"]}).write_csv(tmp_path / "new.tsv", separator="\t")
    r = evaluate(tmp_path / "truth.tsv", tmp_path / "contam.tsv", tmp_path / "new.tsv", "t", tmp_path / "dec.tsv")
    assert r["excluded_low_coverage"] == 1 and r["excluded_no_methylation_evidence"] == 1
    assert r["contaminants"] == 1 and r["contam_recall"] == 1.0
