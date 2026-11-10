"""The command line: errors on the progress stream."""
import subprocess
import sys


def test_errors_go_to_the_progress_stream(tmp_path):
    """Progress is on standard output; an input error and an unexpected traceback must be there too, with a
    one-line message on standard error and a non-zero exit code."""
    args = ["--pileup", str(tmp_path / "p.bed"), "--contig-bin", str(tmp_path / "cb.tsv"), "--motifs",
            str(tmp_path / "m.tsv"), "--out", str(tmp_path / "out")]
    p = subprocess.run([sys.executable, "-m", "epirefine.cli", "run", "--assembly", str(tmp_path / "missing.fa"), *args],
                       capture_output=True, text=True)
    assert p.returncode == 2
    assert "ERROR: assembly not found" in p.stdout and "starting" in p.stdout.splitlines()[0]
    assert "epirefine: error: assembly not found" in p.stderr and "Traceback" not in p.stderr
    # an unexpected failure: the traceback is on the progress stream
    code = ("import sys, epirefine.pipeline as pl, epirefine.cli as cli\n"
            "def boom(*a, **k): raise RuntimeError('boom')\n"
            "pl.run = boom\n"
            "cli.main(sys.argv[1:])\n")
    p = subprocess.run([sys.executable, "-c", code, "run", "--assembly", "a.fa", *args], capture_output=True, text=True)
    assert p.returncode == 1
    assert "Traceback" in p.stdout and "RuntimeError: boom" in p.stdout
    assert "epirefine: error: RuntimeError: boom" in p.stderr
