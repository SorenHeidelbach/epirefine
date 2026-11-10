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


def test_suite_paths_resolve_against_data_root(tmp_path, monkeypatch):
    """Suite files hold no machine path: entries are relative to <data root>/<root>, absolute ones are kept."""
    import json

    import pytest
    from benchmark_suite import load_suite

    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"root": "benchmarks/x", "scenarios": [
        {"name": "a", "group": "g", "assembly": "a/assembly.fasta", "cache": "/abs/cache", "optional": True}]}))
    sc = load_suite(suite, tmp_path / "data")[0]
    assert sc["assembly"] == str(tmp_path / "data/benchmarks/x/a/assembly.fasta")
    assert sc["cache"] == "/abs/cache" and sc["optional"] is True
    monkeypatch.setenv("EPIREFINE_DATA", str(tmp_path / "env"))
    assert load_suite(suite)[0]["assembly"] == str(tmp_path / "env/benchmarks/x/a/assembly.fasta")
    monkeypatch.delenv("EPIREFINE_DATA")
    with pytest.raises(SystemExit):
        load_suite(suite)


def test_shipped_suite_has_no_absolute_paths():
    import json

    suite = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/suite_monocultures.json").read_text())
    assert suite["root"] and not suite["root"].startswith("/")
    assert not any(isinstance(v, str) and v.startswith("/") for sc in suite["scenarios"] for v in sc.values())
