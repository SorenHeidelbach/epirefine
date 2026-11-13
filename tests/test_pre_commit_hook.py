"""dev/hooks/pre-commit: when the .venv can import the compiled extension, the kernel tests must run against it
(EPIREFINE_EXPECT_KERNELS=1, and skips "as if no kernels" fail the hook); without the extension it warns loudly.
Run on a scratch git repository with a stand-in interpreter. Skipped where dev/ is not shipped (an sdist)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "dev" / "hooks" / "pre-commit"
pytestmark = pytest.mark.skipif(not HOOK.is_file() or shutil.which("git") is None, reason="no hook or git here")

PYTHON = """#!/bin/bash
# stand-in .venv/bin/python: "import xdist" fails, the extension imports when CORE=1, pytest prints PYTEST_OUT
if [ "$1" = "-c" ]; then
  case "$2" in *_core*) [ "$CORE" = 1 ]; exit $? ;; *) exit 1 ;; esac
fi
echo "expect=${EPIREFINE_EXPECT_KERNELS:-}"
printf "%b\\n" "$PYTEST_OUT"
exit 0
"""


def _hook(tmp_path, core: bool, pytest_out: str, rust: bool = True):
    repo = tmp_path / "repo"
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / "tests").mkdir()
    if rust:
        (repo / "rust").mkdir()
    py = repo / ".venv/bin/python"
    py.write_text(PYTHON)
    py.chmod(0o755)
    subprocess.run(["git", "init", "-q", "-b", "work", str(repo)], check=True)
    log = tmp_path / "log.txt"
    env = {**os.environ, "CORE": "1" if core else "0", "PYTEST_OUT": pytest_out, "EPIREFINE_PRECOMMIT_LOG": str(log)}
    env.pop("EPIREFINE_EXPECT_KERNELS", None)
    r = subprocess.run(["bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True)
    return r, log.read_text() if log.is_file() else ""


def test_extension_present_means_kernel_tests_are_expected(tmp_path):
    r, log = _hook(tmp_path, core=True, pytest_out="10 passed")
    assert r.returncode == 0 and "expect=1" in log


def test_extension_present_but_kernel_tests_skipped_fails(tmp_path):
    out = "SKIPPED [3] tests/test_rs_model.py:11: no compiled kernels in this install\\n7 passed, 3 skipped"
    r, _ = _hook(tmp_path, core=True, pytest_out=out)
    assert r.returncode == 1 and "kernel tests were skipped" in r.stderr


def test_no_extension_warns_loudly(tmp_path):
    out = "SKIPPED [3] tests/test_rs_model.py:11: no compiled kernels in this install\\n7 passed, 3 skipped"
    r, log = _hook(tmp_path, core=False, pytest_out=out)
    assert r.returncode == 0 and "WARNING: this .venv has no compiled extension" in r.stderr and "expect=\n" in log
