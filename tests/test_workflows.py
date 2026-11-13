"""The release and test workflows (.github/workflows) keep the rules that protect a release: uploads only from a
pushed v* tag, every upload waits for every test, build inputs pinned. Read as text (no YAML parser is a
dependency); skipped where the workflows are not shipped (an sdist or an installed copy of the tests)."""
import re
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
pytestmark = pytest.mark.skipif(not (WF / "release.yml").is_file(), reason="no .github/workflows here")


def jobs(name: str) -> dict[str, str]:
    """{job id: its text} of a workflow file (jobs are the keys indented by two spaces under `jobs:`)."""
    text = (WF / name).read_text()
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def job_if(text: str) -> str:
    m = re.search(r"^    if: (.+)$", text, re.M)
    return m.group(1) if m else ""


def test_uploads_only_from_a_pushed_version_tag():
    """A manual run started on a v* tag must not publish (B6 review P1)."""
    j = jobs("release.yml")
    for job in ("version", "testpypi", "pypi", "github-release"):
        cond = job_if(j[job])
        assert "github.event_name == 'push'" in cond and "startsWith(github.ref, 'refs/tags/v')" in cond, job
    for job, text in j.items():
        if "id-token: write" in text or "contents: write" in text or "gh-action-pypi-publish" in text:
            assert job in ("testpypi", "pypi", "github-release"), job


def test_every_upload_waits_for_every_test():
    """TestPyPI (and through it PyPI and the GitHub release) needs every build and test job, the sdist built from
    source among them (B6 review P2)."""
    j = jobs("release.yml")
    needs = re.search(r"^    needs: \[(.+)\]$", j["testpypi"], re.M).group(1).replace(" ", "").split(",")
    tests = {k for k in j if k.startswith("test-") or k in ("licences",)}
    assert "test-sdist" in tests and tests <= set(needs) and "version" in needs
    assert "--no-binary epirefine" in j["test-sdist"] and "_rs.kernels('model_kernels') is not None" in j["test-sdist"]
