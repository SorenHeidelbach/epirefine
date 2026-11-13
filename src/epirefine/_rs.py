"""The one way EpiRefine reaches its compiled kernels (`epirefine._kernels`, built from `rust/` into the platform
wheels; D73). The numpy code is the reference and the last resort: it runs when the kernels are switched off
(EPIREFINE_RS=0), when this install has no compiled extension (the pure py3-none-any wheel, or a source tree without
a build), when the extension does not load, or when it lacks the interface or the feature the caller needs. Both
paths take the same decisions; the kernels are faster. Which path ran, and why, goes into the log head and
params.json (`kernels`), and a run on numpy says so in one warning line.
"""
from __future__ import annotations

import importlib
import os

#: the interface version of epirefine._kernels this code was written for (its API constant)
REQUIRED_API = 1

_OFF = ("0", "false", "no", "off")


def switched_off() -> bool:
    """EPIREFINE_RS=0 (or false, no, off): the numpy code everywhere."""
    return os.environ.get("EPIREFINE_RS", "1").strip().lower() in _OFF


def load():
    """(module, None) when the compiled kernels load and offer the interface this code needs, else (None, reason).
    Does not look at EPIREFINE_RS. The interface number is the one the compiled extension exports (`_core.API`), so
    a build older than this Python code is refused; a build that lacks a function this code calls at load (an
    AttributeError or TypeError) counts as one that does not load."""
    try:
        _kernels = importlib.import_module("epirefine._kernels")  # by name: honours sys.modules, as tests rely on
    except ModuleNotFoundError as e:
        if e.name in ("epirefine._kernels._core", "epirefine._kernels"):
            return None, "no compiled extension in this install (pure wheel, or a source tree without a build)"
        return None, f"the compiled extension does not load: {e}"
    except (ImportError, AttributeError, TypeError) as e:  # a build for another platform or Python, a missing
        return None, f"the compiled extension does not load: {e}"  # shared library, an incomplete build
    try:
        api = int(getattr(_kernels._core, "API", 0))
        if api < REQUIRED_API:
            return None, f"the compiled extension has interface {api}, this code needs {REQUIRED_API} (a stale build?)"
        _kernels.cpu_path(), _kernels.features()  # every run records them: a build without them is refused here
    except (AttributeError, TypeError, ValueError) as e:
        return None, f"the compiled extension does not load: {type(e).__name__}: {e}"
    return _kernels, None


def kernels(feature: str | None = None):
    """The compiled kernels, or None when the numpy code is to run: switched off, not available, or (feature
    given) without that feature."""
    if switched_off():
        return None
    rs, _ = load()
    if rs is None or (feature is not None and not rs.supports(feature)):
        return None
    return rs


def reason() -> str | None:
    """Why the numpy code runs instead of the kernels (None when the kernels are used)."""
    if switched_off():
        return "switched off by EPIREFINE_RS=0"
    return load()[1]


def summary() -> str:
    """One line for the log head: which path runs."""
    rs = kernels()
    if rs is None:
        return f"kernels: numpy ({reason()})"
    return f"kernels: compiled ({rs.cpu_path()}; built from {getattr(rs, '__commit__', None) or 'an sdist'})"


def notice() -> str | None:
    """The warning of a run on numpy that was not asked for (None when the kernels run or EPIREFINE_RS=0)."""
    if switched_off() or kernels() is not None:
        return None
    return f"WARNING: compiled kernels not used ({reason()}): running the numpy code; slower, same decisions"
