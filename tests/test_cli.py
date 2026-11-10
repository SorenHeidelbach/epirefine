"""The command line: help that stands on its own, the version, errors on the progress stream."""
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from epirefine import cli

ROOT = Path(__file__).resolve().parents[1]


def _run_parser():
    return cli.build_parser()._subparsers._group_actions[0].choices["run"]


def test_every_option_has_help_and_shows_its_default():
    actions = [a for a in _run_parser()._actions if a.option_strings and a.dest != "help"]
    assert len(actions) >= 36
    for a in actions:
        assert a.help and len(a.help) > 10, f"{a.option_strings[0]} has no help text"
    # the help entry of each option (entries start at the left margin of the option column)
    entries = {e.split()[0]: " ".join(e.split()) for e in re.split(r"\n  (?=--)", _run_parser().format_help())[1:]}
    # value options with a default show it; options estimated or calibrated when not given say so
    for opt, shown in (("--contamination-threshold", "default: 0.5"), ("--include-threshold", "default: 0.9"),
                       ("--min-cov", "default: 1"), ("--grid-size", "default: 64"), ("--seed", "default: 1"),
                       ("--frag-lengths", "default: 2000,5000,10000,20000,50000,100000"),
                       ("--chunk-size", "default: 50000"), ("--contam-eps", "default: 0.05"),
                       ("--tau", "default: calibrated"), ("--p-unknown", "default: estimated")):
        assert shown in entries[opt], (opt, entries[opt])
    for a in actions:  # no value option is left without a stated default
        if a.nargs != 0 and not a.required:
            assert "default" in cli._Help(prog="epirefine")._get_help_string(a), a.option_strings[0]


def test_help_has_no_internal_references():
    text = _run_parser().format_help()
    assert not re.search(r"\(D\d+\)|dev log|test B|LOO", text)
    assert "50 kb windows" not in text  # the window is --chunk-size


def test_cli_passes_every_run_argument():
    from epirefine.pipeline import run
    dests = {a.dest for a in _run_parser()._actions}
    renamed = {"motifs_path": "motifs", "motif_noise_filter": "no_motif_noise_filter", "group_weights": "no_group_weights",
               "regional_model": "no_regional_model", "do_calibrate": "no_calibrate", "group_sample_bp": "group_sample_mb",
               "write_motif_fit": "motif_fit"}
    for name in inspect.signature(run).parameters:
        assert renamed.get(name, name) in dests, name


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
