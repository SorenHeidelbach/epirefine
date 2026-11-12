"""benchmarks/benchmark_suite.py records what each run used (review 2026-11-10 08 F10): the version the runs report,
the motif table's md5 and the calibration's choices, as columns appended to older record files."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
bs = pytest.importorskip("benchmark_suite")

OLD_HEADER = ["date", "version", "commit", "group", "scenarios", "contam_recall", "contam_precision", "inclusion_recall",
              "placed_wrongly", "should_stay_unbinned", "contamination_after", "completeness_after", "wall_s",
              "peak_rss_gb", "note", "jobs"]


def test_group_provenance_combines_scenarios():
    rows = [{"scenario": "b", "motifs_md5": "22", "run_version": "0.11.0", "unknown_lam": 1.0, "temperature": 5.0,
             "p_unknown": 0.7},
            {"scenario": "a", "motifs_md5": "11", "run_version": "0.11.0", "unknown_lam": 0.9, "temperature": 1.715,
             "p_unknown": 0.999999}]
    g = bs.group_provenance(rows)
    import hashlib
    assert g["motifs_md5"] == hashlib.md5(b"a\t11\nb\t22").hexdigest()  # order of the rows does not matter
    assert g["run_version"] == "0.11.0" and g["unknown_lam"] == "0.9,1"
    assert g["temperature_range"] == "1.715-5" and g["p_unknown_range"] == "0.7-1"
    one = bs.group_provenance(rows[:1])
    assert one["motifs_md5"] == "22" and one["unknown_lam"] == "1"
    empty = bs.group_provenance([{"scenario": "x"}])
    assert all(v is None for v in empty.values())


def test_old_record_files_get_the_new_columns_at_the_end(tmp_path):
    old = pl.DataFrame({c: ["2026-11-02 12:53", "0.10.1", "082812f", "mono-train", 6, 0.9481, 1.0, 0.893, 2.0, 159.1667,
                            0.0042, 0.9829, 32.2, 1.38, "a note", 1][i:i + 1] for i, c in enumerate(OLD_HEADER)})
    path = tmp_path / "results.tsv"
    old.write_csv(path, separator="\t")
    new = old.with_columns(pl.lit("0.11.1").alias("version"), *[pl.lit("v").alias(c) for c in bs.PROVENANCE])
    out = bs.append_rows(path, new)
    assert out.columns == OLD_HEADER + bs.PROVENANCE
    assert out.height == 2 and out["motifs_md5"].to_list() == [None, "v"]
    out.write_csv(path, separator="\t")
    again = bs.append_rows(path, new)  # a file that already has them keeps its order
    assert again.columns == OLD_HEADER + bs.PROVENANCE and again.height == 3


@pytest.mark.skipif(not (REPO / ".venv/bin/epirefine").exists() or not Path("/usr/bin/time").exists(),
                    reason="needs the repository's .venv and /usr/bin/time")
def test_suite_records_provenance_end_to_end(small_community, tmp_path):
    suite = tmp_path / "suite.json"
    sc = {"name": "small", "group": "tiny", "assembly": str(small_community["assembly"]),
          "pileup": str(small_community["pileup"]), "contig_bin": str(small_community["contig_bin"]),
          "motifs": str(small_community["motifs"]),
          "args": ["--frag-lengths", "2000,5000", "--chunk-size", "10000", "--frags-per-bin", "6"]}
    suite.write_text(json.dumps({"scenarios": [sc]}))
    devlog = tmp_path / "devlog"
    (devlog / "benchmarks").mkdir(parents=True)
    results = tmp_path / "results.tsv"
    pl.DataFrame({c: [] for c in OLD_HEADER}, schema={c: pl.Utf8 for c in OLD_HEADER}).write_csv(results, separator="\t")
    p = subprocess.run([sys.executable, str(REPO / "benchmarks/benchmark_suite.py"), "--suite", str(suite),
                        "--results", str(results), "--devlog", str(devlog), "--out-root", str(tmp_path / "runs"),
                        "--workers", "1", "--note", "test"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    res = pl.read_csv(results, separator="\t", infer_schema_length=None)
    assert res.columns == OLD_HEADER + bs.PROVENANCE
    row = res.row(-1, named=True)
    params = json.loads(next((tmp_path / "runs").glob("*/small/params.json")).read_text())
    assert row["motifs_md5"] == bs.file_md5(small_community["motifs"])
    assert row["run_version"] == params["version"]
    assert str(row["unknown_lam"]) == f"{params['unknown_lam']:g}"
    t = params["temperature"]
    assert row["temperature_range"] == f"{t:.4g}-{t:.4g}"
    per = pl.read_csv(next((devlog / "benchmarks/runs").glob("*.tsv")), separator="\t")
    assert {"motifs_md5", "run_version", "unknown_eta", "unknown_lam", "temperature", "fitted_temperature"} <= set(per.columns)
    hist = pl.read_csv(devlog / "benchmarks/history.tsv", separator="\t")
    assert hist.columns[-len(bs.PROVENANCE):] == bs.PROVENANCE
    shutil.rmtree(tmp_path / "runs")
