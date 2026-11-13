"""params.json as a record of the run (gate R8): the code (commit), the libraries, the compiled kernels, the threads
and the seeds; and a rerun from params.json that repeats the decisions."""
import json
import subprocess
from pathlib import Path

import pytest

from helpers import DECISION_FILES, assert_same_outputs, copy_cache, logged, run_pipeline

# fields that must differ between two runs of the same command (or a rerun into another folder)
RUN_SPECIFIC = {"start_time", "elapsed_s", "peak_rss_gb", "peak_rss_worker_gb", "out", "cache_dir", "command_line"}
REPO = Path(__file__).resolve().parents[1]


def _params(run) -> dict:
    return json.loads((run.out / "params.json").read_text())


def test_params_record_code_libraries_kernels_threads_seeds(base_run):
    import numpy, polars, scipy
    p = _params(base_run)
    import epirefine
    # the commit is the checkout's only when the package runs from it (an editable install), not from an installed wheel
    if (REPO / ".git").exists() and Path(epirefine.__file__).resolve().parents[1] == REPO / "src":
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert p["commit"]["source"] == "git" and p["commit"]["commit"] == head
        assert isinstance(p["commit"]["dirty"], bool)
    lib = p["libraries"]
    assert (lib["numpy"], lib["scipy"], lib["polars"]) == (numpy.__version__, scipy.__version__, polars.__version__)
    k = p["kernels"]
    from epirefine import _rs as R
    if R.load()[0] is not None:
        assert k["available"] and k["cpu_path"] in ("avx2_fma", "generic") and f"cpu:{k['cpu_path']}" in k["features"]
        assert len(k["module_sha256"]) == 64 and k["api"] >= 1
    else:
        assert not k["available"] and not k["used"] and k["reason"].startswith("no compiled extension")
    from epirefine.model import _rs
    assert k["used"] == (_rs() is not None) and (k["reason"] is None) == k["used"]
    t = p["threads"]
    assert t["workers"] == 1 and t["cores"] >= 1 and t["model_threads"] >= 1 and "openblas_num_threads" in t
    assert p["seeds"] == {"seed": 1, "calibration_seed": 1, "bootstrap": 0}


def test_commit_without_git_from_environment_or_file(tmp_path, monkeypatch):
    from epirefine.runinfo import tool_commit
    monkeypatch.delenv("EPIREFINE_COMMIT", raising=False)
    assert tool_commit(tmp_path) == {"commit": None, "source": None}
    (tmp_path / "COMMIT").write_text("abc1234\n")
    assert tool_commit(tmp_path)["commit"] == "abc1234"
    monkeypatch.setenv("EPIREFINE_COMMIT", "def5678")
    assert tool_commit(tmp_path) == {"commit": "def5678", "source": "EPIREFINE_COMMIT"}


def test_kernels_off_is_recorded(monkeypatch):
    from epirefine.runinfo import kernels
    monkeypatch.setenv("EPIREFINE_RS", "0")
    k = kernels()
    assert k["used"] is False and k["EPIREFINE_RS"] == "0" and k["reason"] == "switched off by EPIREFINE_RS=0"


def test_rerun_from_params_repeats_the_decisions(community, base_run, tmp_path):
    """The default run repeated from its params.json (into another folder, from a copy of its cache): the same
    decisions, and params.json the same except the fields that belong to one run."""
    from epirefine.runinfo import rerun_arguments
    out = copy_cache(base_run, tmp_path / "out")
    kw = rerun_arguments(_params(base_run), out, cache_dir=str(out / "cache"))
    run = run_pipeline(**{k: v for k, v in kw.items() if k != "workers"}, workers=kw["workers"])
    assert run.reused_cache
    assert_same_outputs(base_run.out, run.out, DECISION_FILES + ["calibration/fragments.tsv"])
    a, b = _params(base_run), _params(run)
    # kernels.ran counts the calls of this run: a rerun from the cache makes no pileup pass
    ka, kb = a["kernels"].pop("ran"), b["kernels"].pop("ran")
    assert {k: v for k, v in ka.items() if k != "pileup_reader"} == {k: v for k, v in kb.items() if k != "pileup_reader"}
    assert "pileup_reader" in ka and "pileup_reader" not in kb
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} <= RUN_SPECIFIC


def test_cli_rerun_of_a_run_with_other_options(community, base_run, tmp_path):
    """Options away from the default (a calibration seed, thresholds, piece lengths) come back from params.json."""
    from epirefine.cli import main
    first = copy_cache(base_run, tmp_path / "first")
    logged(main, community.cli_args(first, "--workers", 1, "--cache-dir", first / "cache", "--calibration-seed", 3,
                                    "--include-threshold", 0.8, "--frag-lengths", "2000,5000,20000",
                                    "--contamination-threshold", 0.6))
    log = logged(main, ["rerun", "--params", str(first / "params.json"), "--out", str(tmp_path / "again")])
    assert "reusing site cache" in log and "since the recorded run" not in log and "the recorded run used" not in log
    assert_same_outputs(first, tmp_path / "again", DECISION_FILES + ["calibration/fragments.tsv"])
    p = json.loads((tmp_path / "again" / "params.json").read_text())
    assert p["calibration_seed"] == 3 and p["include_threshold"] == 0.8 and p["frag_lengths"] == [2000, 5000, 20000]


def test_rerun_warns_about_a_changed_input(community, base_run, tmp_path):
    from epirefine.runinfo import rerun_arguments
    p = _params(base_run)
    p["inputs"]["pileup"]["size"] += 1
    msgs = []
    rerun_arguments(p, tmp_path / "o", log=msgs.append)
    assert any("has changed since the recorded run" in m for m in msgs)
