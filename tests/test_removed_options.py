"""Options that lost, removed in 0.12 (U12 in the dev log, D63): the command line no longer offers them, and a rerun
of a params.json that used one says so in one line instead of silently running without it."""
import json

import pytest

from epirefine import cli
from epirefine.io import InputError
from epirefine.runinfo import REMOVED_OPTIONS, removed_options, rerun_arguments

REMOVED_FLAGS = ["--calibration", "--calibration-exclusion", "--tau-select", "--npmle-start", "--slot-objective",
                 "--slot-test-b", "--slot-kappa", "--robust-fit", "--max-flagged-share", "--tau-support",
                 "--unlisted-genome", "--inclusion-support", "--regional-inclusion"]


def _run_parser():
    return cli.build_parser()._subparsers._group_actions[0].choices["run"]


def test_help_lists_no_removed_option():
    flags = {s for a in _run_parser()._actions for s in a.option_strings}
    assert not flags & set(REMOVED_FLAGS)
    text = _run_parser().format_help()
    for f in REMOVED_FLAGS:
        assert f"{f} " not in text and f"{f}\n" not in text, f
    assert "{community-only,per-bin" not in text and "slot" not in text
    assert {flag for flag, _ in REMOVED_OPTIONS.values()} == set(REMOVED_FLAGS)


@pytest.mark.parametrize("argv", [["--calibration", "coherent"], ["--robust-fit", "damped"], ["--regional-inclusion"],
                                  ["--relatives", "per-bin"], ["--relatives", "slot"]])
def test_the_command_line_refuses_a_removed_option(argv, capsys):
    base = ["run", "--assembly", "a", "--pileup", "p", "--contig-bin", "c", "--motifs", "m", "--out", "o"]
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(base + argv)
    assert e.value.code == 2


def test_removed_options_at_their_old_default_are_accepted(base_run):
    """A params.json of 0.11.1 records every option; at the value whose behaviour remains it reruns as before."""
    par = json.loads((base_run.out / "params.json").read_text())
    par.update({k: d for k, (_, d) in REMOVED_OPTIONS.items()})
    assert removed_options(par) == []
    rerun_arguments(par, "again", log=lambda *a: None)


def test_rerun_of_a_removed_option_fails_in_one_line(base_run, tmp_path, capsys):
    par = json.loads((base_run.out / "params.json").read_text())
    par.update(calibration="coherent", robust_fit="damped", regional_inclusion=True, relatives="slot")
    assert sorted(removed_options(par)) == ["--calibration coherent", "--regional-inclusion", "--relatives slot",
                                            "--robust-fit damped"]
    with pytest.raises(InputError, match="removed in 0.12"):
        rerun_arguments(par, tmp_path / "again", log=lambda *a: None)
    path = tmp_path / "params.json"
    path.write_text(json.dumps(par))
    with pytest.raises(SystemExit) as e:
        cli.main(["rerun", "--params", str(path), "--out", str(tmp_path / "again")])
    assert e.value.code == 2
    err = capsys.readouterr().err.strip()
    assert err.count("\n") == 0 and "--calibration coherent" in err and "removed in 0.12" in err
    assert not (tmp_path / "again").exists()
