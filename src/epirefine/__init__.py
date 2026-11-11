"""EpiRefine: bin refinement from nanopore methylation."""
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _source_tree_version():
    """Version in the pyproject.toml of a source checkout (src/epirefine two levels below it), None for an
    installed package. An editable install keeps the metadata of the day it was made: after a version bump the
    code is new and the metadata old, and outputs would be stamped with the wrong version."""
    try:
        text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r'^name\s*=\s*"epirefine"', text, re.M):
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def _version() -> str:
    v = _source_tree_version()
    if v is not None:
        return v
    try:
        return version("epirefine")
    except PackageNotFoundError:  # a copy of the package without metadata or pyproject.toml
        return "unknown"


__version__ = _version()
