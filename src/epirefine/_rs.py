"""The one way EpiRefine reaches its compiled kernels (`epirefine._kernels`, built from `rust/` into the platform
wheels; D73). The numpy code is the reference and the last resort: it runs when the kernels are switched off
(EPIREFINE_RS=0), when this install has no compiled extension (the pure py3-none-any wheel, or a source tree without
a build), when the extension does not load, or when it lacks the interface or the feature the caller needs. Both
paths take the same decisions; the kernels are faster. Which path ran, and why, goes into the log head and
params.json (`kernels`), and a run on numpy says so in one warning line.

One decision per run. The extension is imported once per process (load() keeps the result), and a run fixes
kernels-or-numpy at its start (start_run, called by runinfo.begin): every later call in the main process follows
that decision, also when EPIREFINE_RS or the installed build changes during the run. The worker processes of the
run receive the decision through the environment (EPIREFINE_KERNELS_RUN: "numpy", or the SHA-256 of the compiled
module the main process loaded); a worker that cannot load that same build stops the run with an error, so one run
never mixes the two code paths or two builds. Each call site counts which path it took (count); the counts of the
workers come back with their results, and params.json records them at the end of the run (kernels.ran).
"""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

#: the interface version of epirefine._kernels this code was written for (its API constant)
REQUIRED_API = 1

#: the environment variable that hands the main process's decision to the worker processes of a run
ENV_RUN = "EPIREFINE_KERNELS_RUN"

_OFF = ("0", "false", "no", "off")
SWITCHED_OFF = "switched off by EPIREFINE_RS=0"

_LOADED = None      # (module or None, reason) of the first load() in this process
_RUN = None         # the decision of the run in progress in this (main) process: start_run
_HANDED = None      # (value of ENV_RUN, (module or None, reason)) in a worker process, checked once
_COUNTS: dict[str, list[int]] = {}  # kernel -> [numpy calls, compiled calls] in this process
_MISSING: set[str] = set()          # features a call asked for and the loaded build lacks


def switched_off() -> bool:
    """EPIREFINE_RS=0 (or false, no, off): the numpy code everywhere."""
    return os.environ.get("EPIREFINE_RS", "1").strip().lower() in _OFF


def _load():
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


def load():
    """_load() once per process: the result of the first import is kept (a failed import too), so that a build
    installed, removed or rebuilt while a run is going does not change which code runs in this process."""
    global _LOADED
    if _LOADED is None:
        _LOADED = _load()
    return _LOADED


def reset() -> None:
    """Forget the import, the run's decision and the counts of this process (tests that change the installed
    build in sys.modules or in the module itself, then look again)."""
    global _LOADED, _RUN, _HANDED
    _LOADED = _RUN = _HANDED = None
    _COUNTS.clear()
    _MISSING.clear()
    os.environ.pop(ENV_RUN, None)


def module_sha256(rs) -> str | None:
    """SHA-256 of the compiled module file that is loaded (rs._core.__file__): identifies the build exactly."""
    try:
        return hashlib.sha256(Path(rs._core.__file__).read_bytes()).hexdigest()
    except (AttributeError, TypeError, OSError):
        return None


def start_run() -> dict:
    """Fix kernels-or-numpy for the run that starts in this process, clear the counts, and hand the decision to
    the run's worker processes (ENV_RUN in the environment they inherit). Returns the decision."""
    global _RUN
    rs, why = (None, SWITCHED_OFF) if switched_off() else load()
    sha = module_sha256(rs) if rs is not None else None
    if rs is not None and sha is None:
        rs, why = None, "the compiled extension does not load: its module file cannot be read"
    _RUN = {"rs": rs, "reason": why, "sha256": sha}
    os.environ[ENV_RUN] = f"sha256:{sha}" if rs is not None else "numpy"
    _COUNTS.clear()
    _MISSING.clear()
    return _RUN


def end_run() -> None:
    """The run in this process has ended (also when it failed): later calls decide afresh."""
    global _RUN
    _RUN = None
    os.environ.pop(ENV_RUN, None)


def _in_worker() -> bool:
    import multiprocessing
    return multiprocessing.parent_process() is not None


def _handed(value: str):
    """In a worker process of a run: the main process's decision, checked against what this process loads."""
    global _HANDED
    if _HANDED is not None and _HANDED[0] == value:
        return _HANDED[1]
    if value == "numpy":
        got = (None, "the main process of this run runs the numpy code")
    else:
        rs, why = load()
        if rs is None:
            raise RuntimeError(f"a worker process cannot load the compiled kernels the run started with ({why}); "
                               "stopping, so that one run does not mix the compiled and the numpy code")
        sha = module_sha256(rs)
        if f"sha256:{sha}" != value:
            raise RuntimeError(f"a worker process loaded another build of the compiled kernels (sha256 {sha}) than "
                               f"the run started with ({value}): the extension changed during the run; stopping")
        got = (rs, None)
    _HANDED = (value, got)
    return got


def decision():
    """(module or None, reason) for a call in this process now: the run's decision in its main process, the
    handed-down one in its workers, else EPIREFINE_RS and load()."""
    if _RUN is not None:
        return _RUN["rs"], _RUN["reason"]
    value = os.environ.get(ENV_RUN)
    if value and _in_worker():
        return _handed(value)
    if switched_off():
        return None, SWITCHED_OFF
    return load()


def kernels(feature: str | None = None):
    """The compiled kernels, or None when the numpy code is to run: switched off, not available, or (feature
    given) without that feature (noted for the run record)."""
    rs = decision()[0]
    if rs is None:
        return None
    if feature is not None and not rs.supports(feature):
        _MISSING.add(feature)
        return None
    return rs


def reason() -> str | None:
    """Why the numpy code runs instead of the kernels (None when the kernels are used)."""
    return decision()[1]


def summary() -> str:
    """One line for the log head: which path runs."""
    rs, why = decision()
    if rs is None:
        return f"kernels: numpy ({why})"
    return f"kernels: compiled ({rs.cpu_path()}; built from {getattr(rs, '__commit__', None) or 'an sdist'})"


def notice() -> str | None:
    """The warning of a run on numpy that was not asked for (None when the kernels run or EPIREFINE_RS=0)."""
    rs, why = decision()
    if rs is not None or why == SWITCHED_OFF:
        return None
    return f"WARNING: compiled kernels not used ({why}): running the numpy code; slower, same decisions"


def count(kernel: str, compiled: bool) -> None:
    """A call site took the compiled kernel (compiled=True) or the numpy code for `kernel`."""
    _COUNTS.setdefault(kernel, [0, 0])[1 if compiled else 0] += 1


def take_counts() -> dict | None:
    """In a worker process: the counts since the last take (cleared), to send back with the task's result. In the
    main process None (its counts stay where they are)."""
    if not _in_worker():
        return None
    out = {k: list(v) for k, v in _COUNTS.items()}
    _COUNTS.clear()
    return out


def add_counts(counts: dict | None) -> None:
    """Add the counts a worker sent back (take_counts) to this process's."""
    for k, (n, c) in (counts or {}).items():
        v = _COUNTS.setdefault(k, [0, 0])
        v[0] += n
        v[1] += c


def ran() -> dict:
    """{kernel: {"compiled": calls, "numpy": calls}} of this process (with the workers' counts added)."""
    return {k: {"compiled": c, "numpy": n} for k, (n, c) in sorted(_COUNTS.items())}


def missing_features() -> list[str]:
    return sorted(_MISSING)
