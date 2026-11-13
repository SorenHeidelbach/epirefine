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


def test_pure_wheel_runs_with_the_lowest_pins_on_python_310():
    """The numpy code is what users of the pure wheel run: both workflows test it with the lowest dependency versions
    on Python 3.10 (B6 review P3), reading the bounds with tomllib."""
    t = jobs("tests.yml")["numpy-only"]
    assert re.search(r'wheel: "\*-py3-none-any.whl"\n\s+rs: ""\n\s+python: "3.10"\n\s+lowest: true', t)
    r = jobs("release.yml")["test-pure"]
    assert '{ python: "3.10", lowest: true }' in r
    for text in (t, r):
        assert "--resolution lowest-direct" in text and "tomllib.load" in text


def test_build_tools_are_pinned_and_the_lock_file_is_used():
    """maturin and the Rust toolchain are fixed versions, the same in both workflows; every build uses Cargo.lock as
    it is (--locked); release.yml checks the bundled licences against Cargo.lock before any upload (B6 review P4)."""
    pins = []
    for name in ("release.yml", "tests.yml"):
        text = (WF / name).read_text()
        m, r = re.search(r'MATURIN_VERSION: "([0-9.]+)"', text), re.search(r'RUST_TOOLCHAIN: "([0-9.]+)"', text)
        assert m and r, name
        pins.append((m.group(1), r.group(1)))
        assert "dtolnay/rust-toolchain@stable" not in text and "maturin>=" not in text
        for line in re.findall(r"maturin build[^\n]*|args: --release[^\n]*", text):
            assert "--locked" in line, line
        assert len(re.findall(r"^\s+toolchain: \$\{\{ env.RUST_TOOLCHAIN \}\}$", text, re.M)) == text.count("dtolnay/rust-toolchain@")
    assert pins[0] == pins[1]
    j = jobs("release.yml")
    assert "maturin-version: ${{ env.MATURIN_VERSION }}" in j["wheels"] and "rust-toolchain: ${{ env.RUST_TOOLCHAIN }}" in j["wheels"]
    assert "bundle_licenses.py --check" in j["licences"] and "cargo fetch --locked" in j["licences"]


def test_read_only_token_and_actions_pinned_by_commit():
    """Both workflows give the token read rights unless a job asks for more, and every action is pinned to a full
    commit SHA (a moved tag cannot change what builds the published files)."""
    for name in ("release.yml", "tests.yml"):
        text = (WF / name).read_text()
        assert re.search(r"^permissions:\n  contents: read$", text, re.M), name
        uses = re.findall(r"uses: (\S+)", text)
        assert uses and all(re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u) for u in uses), name


def test_release_workflow_builds_main_and_checks_the_installed_command():
    """release.yml builds and tests every platform on pushes to main (before a tag); every job that installs a wheel
    or the sdist runs the `epirefine` command; the lowest pins are read with tomllib, not a regex; tests.yml runs the
    Rust unit tests."""
    text = (WF / "release.yml").read_text()
    branches = re.search(r"^    branches: \[(.+)\]$", text, re.M).group(1)
    assert '"main"' in branches
    j = jobs("release.yml")
    for job in ("test-wheels", "test-musl", "test-pure", "test-sdist"):
        assert "epirefine --version" in j[job], job
    assert "re.findall" not in j["test-wheels"] and "tomli" in j["test-wheels"]
    t = jobs("tests.yml")
    for job in ("test", "lowest-dependencies", "numpy-only"):
        assert "epirefine --version" in t[job], job
    assert "cargo test --manifest-path rust/Cargo.toml --release --locked" in t["build"]
