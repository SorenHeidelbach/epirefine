"""Run bookkeeping: the log, the checks made on the inputs and options before any heavy work, and params.json,
the record of what a run was made with.

Nothing here touches the model; a run with valid inputs gives the same decisions with or without it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import InputError, _open


try:  # not on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None

_T0 = [time.time()]


def start_clock():
    """Elapsed times in the log count from here (the start of a run), not from the import."""
    _T0[0] = time.time()


def elapsed() -> float:
    return time.time() - _T0[0]


def peak_rss_gb(children: bool = False) -> float:
    """Peak resident memory of this process (or of its largest finished worker process), in GB."""
    if resource is None:
        return float("nan")
    r = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    return r / (1e9 if sys.platform == "darwin" else 1e6)  # bytes on macOS, kB on Linux


def log(msg):
    """One progress line on standard output: elapsed time, peak memory of the main process so far, message.
    Errors are reported on the same stream by the command line (cli.main), so one captured stream holds both."""
    print(f"[{elapsed():7.1f}s {peak_rss_gb():5.1f} GB] {msg}", file=sys.stdout, flush=True)


def package_version() -> str:
    from . import __version__
    return __version__


# ---------------------------------------------------------------------------------------------------------
# checks before any heavy work

INPUT_NAMES = {"assembly": "assembly", "pileup": "pileup", "contig_bin": "contig-bin table", "motifs_path": "motif table"}


def begin(args: dict, log=log) -> None:
    """Start of a run, before anything is read (args: the arguments of pipeline.run by name): start the clock,
    write the first log lines (version, command line, inputs, workers, seed) and check the option ranges. A run
    killed while the assembly is loaded then still leaves a log that says what it was."""
    start_clock()
    log(f"epirefine {package_version()} starting; command: {' '.join(sys.argv)}")
    log("inputs: " + ", ".join(f"{k.removesuffix('_path')}={args[k]}" for k in INPUT_NAMES) + f"; out={args['out']}; "
        f"workers={n_workers(args['workers'])}; seed={args['seed']}"
        + (f"; calibration seed={args['calibration_seed']}" if args.get("calibration_seed") is not None else ""))
    from . import _rs  # which code runs: the compiled kernels, or numpy and why (one warning when not asked for)
    log(_rs.summary())
    if (warn := _rs.notice()) is not None:
        log(warn)
    check_settings(args)
    if args["contam_eps"] == 0:
        log("--contam-eps 0: no robust bin profiles and no regional model (long contigs are judged as a whole)")


def n_workers(workers) -> int:
    from .model import _cores
    return workers or min(16, _cores())


def check_inputs(args: dict) -> None:
    """The four input files of pipeline.run (args: its arguments by name) are there and readable."""
    for k, what in INPUT_NAMES.items():
        check_file(args[k], what)


def check_file(path, what: str) -> None:
    """The input exists, is a readable non-empty file, and its compression matches its name (readers decide
    by the .gz suffix)."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"{what} not found: {p}")
    if not p.is_file():
        raise InputError(f"{what} is not a file: {p}")
    try:
        with open(p, "rb") as fh:
            head = fh.read(2)
    except OSError as e:
        raise InputError(f"{what} cannot be read: {p} ({e.strerror or e})") from None
    if not head:
        raise InputError(f"{what} is empty: {p}")
    gz = head == b"\x1f\x8b"
    if gz and not str(p).endswith(".gz"):
        raise InputError(f"{what} is gzip-compressed but not named .gz: {p} (rename it; files are read by suffix)")
    if not gz and str(p).endswith(".gz"):
        raise InputError(f"{what} is named .gz but is not gzip-compressed: {p}")


def check_pileup_head(path) -> str:
    """First data row of the pileup must look like modkit bedMethyl; returns its contig name."""
    with _open(path) as fh:
        line = next((ln for ln in fh if ln.strip() and not ln.startswith("#")), None)
    if line is None:
        raise InputError(f"pileup has no rows: {path}")
    f = line.rstrip("\r\n").replace(" ", "\t").split("\t")
    if len(f) < 12:
        raise InputError(f"pileup does not look like a modkit bedMethyl table: {path}: the first row has {len(f)} "
                         f"columns, at least 12 are needed (contig, start, end, modification code, score, strand, "
                         f"..., N_valid, fraction, N_mod): {line.strip()[:120]!r}")
    try:
        int(f[1]); int(f[9]); int(f[11])
    except ValueError:
        raise InputError(f"pileup does not look like a modkit bedMethyl table: {path}: columns 2, 10 and 12 of the "
                         f"first row must be whole numbers (was it written with a header line, modkit --header? "
                         f"remove it): {line.strip()[:120]!r}") from None
    return f[0]


def match_contig_bin(cb: dict[str, str], lengths: dict[str, int], path, log=log) -> dict[str, str]:
    """Rows of the contig-bin table whose contig is in the assembly; stops when none is, says how many were
    dropped otherwise (they would vanish from the refined bins without a trace)."""
    kept = {c: b for c, b in cb.items() if c in lengths}
    missing = [c for c in cb if c not in lengths]
    if not kept and sum(b in lengths for b in cb.values()) > len(cb) / 2:
        raise InputError(f"contig-bin table {path}: the second column holds contig names of the assembly; the columns "
                         f"are swapped (it needs contig<TAB>bin)")
    if not kept:
        raise InputError(
            f"no bin left: none of the {len(cb)} contigs in {path} is in the assembly (contig-bin names e.g. "
            f"{', '.join(map(repr, missing[:3]))}; assembly names e.g. {', '.join(map(repr, list(lengths)[:3]))}). "
            f"The table needs contig<TAB>bin with the contig names of the assembly (the first word of each "
            f"FASTA header) in column 1.")
    if missing:
        share = len(kept) / len(cb)
        emptied = sorted(set(cb.values()) - set(kept.values()))
        if emptied:
            log(f"WARNING: contig-bin table: {len(emptied)} bins have none of their contigs in the assembly and are "
                f"left out: {', '.join(emptied[:5])}{', ...' if len(emptied) > 5 else ''}")
        log(f"WARNING: contig-bin table: {len(missing)} of {len(cb)} contigs are not in the assembly and are dropped "
            f"(e.g. {', '.join(missing[:3])}); {len(kept)} contigs in {len(set(kept.values()))} of "
            f"{len(set(cb.values()))} bins remain"
            + ("; fewer than half match: check that the table belongs to this assembly" if share < 0.5 else ""))
    return kept


def check_settings(a: dict) -> None:
    """Ranges of the numeric options; a value outside gives NaN or an obscure error much later."""
    def need(ok, msg):
        if not ok:
            raise InputError(msg)
    for k in ("contamination_threshold", "include_threshold", "include_threshold_unmethylated", "max_redundancy",
              "min_methylome_difference", "min_jaccard"):
        need(0 <= a[k] <= 1, f"--{k.replace('_', '-')} must be between 0 and 1, got {a[k]}")
    need(a["p_unknown"] is None or 0 <= a["p_unknown"] <= 1, f"--p-unknown must be between 0 and 1, got {a['p_unknown']}")
    # contam_eps is a prior share: at 1 (or above) its log-odds are infinite; 0 switches the robust profiles and
    # the regional model off
    need(0 <= a["contam_eps"] < 1, f"--contam-eps must be at least 0 (off) and below 1, got {a['contam_eps']}")
    for k in ("chunk_size", "grid_size", "frags_per_bin", "frag_budget"):
        need(a[k] >= 1, f"--{k.replace('_', '-')} must be at least 1, got {a[k]}")
    need(a["min_cov"] >= 0, f"--min-cov must not be negative, got {a['min_cov']}")
    need(a["workers"] is None or a["workers"] >= 0, f"--workers must not be negative, got {a['workers']}")
    need(len(a["frag_lengths"]) > 0 and all(x > 0 for x in a["frag_lengths"]),
         f"--frag-lengths needs positive lengths, got {a['frag_lengths']}")
    need(a["temperature"] is None or a["temperature"] > 0, f"--temperature must be positive, got {a['temperature']}")


def check_calibration_contigs(lengths_ok: list[int], n_binned: int, min_contig_coverage: float, frag_lengths) -> None:
    """Calibration cuts pseudo-contigs from binned contigs with enough coverage (their lengths: lengths_ok):
    there must be one at least as long as the shortest piece."""
    shortest, n_covered = min(frag_lengths), len(lengths_ok)
    if not any(L >= shortest for L in lengths_ok):
        raise InputError(
            f"no calibration fragments: calibration needs at least one binned contig with median coverage >= "
            f"{min_contig_coverage:g} reads per motif site (--min-contig-coverage) and length >= {shortest} bp (the "
            f"shortest of --frag-lengths). Of {n_binned} binned contigs, {n_covered} have that coverage and "
            f"{sum(L >= shortest for L in lengths_ok)} of those that length. Lower --min-contig-coverage or "
            f"--frag-lengths, or run with --no-calibrate (uncalibrated probabilities).")


def no_sites_error(pileup, pileup_first: str | None, lengths: dict[str, int]) -> InputError:
    return InputError(
        f"no pileup row lies at a motif site of an assembly contig: {pileup}. Either the pileup's contig names do "
        f"not match the assembly's (pileup: {pileup_first!r}{'' if pileup_first in lengths else ', not in the assembly'}"
        f"; assembly e.g. {', '.join(map(repr, list(lengths)[:3]))}), or its modification codes are not among "
        f"a, m, h, 21839, or the motifs do not occur. The pileup must come from reads mapped to this assembly.")


# ---------------------------------------------------------------------------------------------------------
# params.json

def json_safe(o):
    """Plain JSON values: numpy types to Python, NaN to null, infinities to the strings "inf" / "-inf"
    (json.dumps would write the bare words NaN and Infinity, which are not JSON)."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, np.ndarray)):
        return [json_safe(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return o if math.isfinite(o) else None if math.isnan(o) else "inf" if o > 0 else "-inf"
    if o is None or isinstance(o, str):
        return o
    return str(o)


def write_params(path, params: dict) -> None:
    Path(path).write_text(json.dumps(json_safe(params), indent=2, allow_nan=False) + "\n")


def _input(path) -> dict:
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}


def provenance(args: dict) -> dict:
    """What a run was made with, for params.json. args: the arguments of pipeline.run by name. Holds the
    version, command line, start time, inputs (path, size, modification time), seed, worker count and every
    option as given."""
    inputs = {"assembly": args["assembly"], "pileup": args["pileup"], "contig_bin": args["contig_bin"],
              "motifs": args["motifs_path"]}
    given = {"p_unknown": "p_unknown_given", "tau": "tau_given", "temperature": "temperature_given"}
    skip = {"assembly", "pileup", "contig_bin", "motifs_path", "out", "workers", "cache_dir"}
    prov = {
        "version": package_version(),
        "command_line": list(sys.argv),
        "start_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished": False,
        "inputs": {k: _input(v) for k, v in inputs.items()},
        "out": str(Path(args["out"]).resolve()),
        "cache_dir": str(Path(args["cache_dir"]).resolve()) if args["cache_dir"] else None,
        "workers": n_workers(args["workers"]),
        "python": sys.version.split()[0],
    }
    # the options as given (p_unknown, tau and temperature under *_given: their plain keys hold the values
    # used, estimated or calibrated when not given)
    prov.update(environment(args))
    prov.update({given.get(k, k): v for k, v in args.items() if k not in skip})
    if prov.get("calibration_seed") is None and "seed" in prov:
        prov["calibration_seed"] = prov["seed"]  # the seed the pieces were drawn with, also when not given
    return prov


def tool_commit(root=None) -> dict:
    """The commit of the epirefine code that runs: from its git checkout when it runs from one (with `dirty` true
    when tracked files differ from the commit), else from the environment variable EPIREFINE_COMMIT, else from a
    file COMMIT next to `src` (or inside the package) that a snapshot made with `git archive` can carry. `commit`
    is null when none of these says (an installed release: its version names it)."""
    pkg = Path(__file__).resolve().parent
    root = Path(root) if root is not None else pkg.parents[1]  # <checkout>/src/epirefine
    if (root / ".git").exists():
        try:
            def git(*a):
                return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=20,
                                      check=True).stdout.strip()
            commit = git("rev-parse", "HEAD")
            dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
            return {"commit": commit, "source": "git", "dirty": dirty}
        except (OSError, subprocess.SubprocessError):
            pass
    env = os.environ.get("EPIREFINE_COMMIT", "").strip()
    if env:
        return {"commit": env, "source": "EPIREFINE_COMMIT"}
    for f in (root / "COMMIT", pkg / "COMMIT"):
        try:
            text = f.read_text().split()
        except OSError:
            continue
        if text:
            return {"commit": text[0], "source": f"COMMIT file ({f.name} next to {'src' if f.parent == root else 'the package'})"}
    return {"commit": None, "source": None}


def kernels() -> dict:
    """Whether the compiled kernels (epirefine._kernels, built into the platform wheels) are available and used, and
    if not, why; and which build: the commit it was built from, the CPU path of the model kernels (avx2_fma or
    generic), the API level, the features and the SHA-256 of the compiled module (identifies the build exactly).
    `used` follows EPIREFINE_RS and the kernels' interface, as the model code does (epirefine._rs)."""
    from . import _rs
    rs, _ = _rs.load()
    used = _rs.kernels() is not None
    rec = {"available": rs is not None, "used": used, "reason": None if used else _rs.reason(),
           "EPIREFINE_RS": os.environ.get("EPIREFINE_RS"), "EPIREFINE_RS_NO_SIMD": os.environ.get("EPIREFINE_RS_NO_SIMD")}
    if rs is None:
        return rec
    rec.update(commit=getattr(rs, "__commit__", None), api=getattr(rs, "API", None))
    try:
        rec["cpu_path"] = rs.cpu_path()
        rec["features"] = sorted(rs.features())
    except Exception:
        rec["cpu_path"], rec["features"] = None, None
    so = sorted(Path(rs.__file__).parent.glob("_core*"))
    if so:
        rec["module_sha256"] = hashlib.sha256(so[0].read_bytes()).hexdigest()
    return rec


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS")


def environment(args: dict) -> dict:
    """The parts of params.json that say which code ran where: commit, libraries, kernels, threads, seeds."""
    import numpy
    import polars
    import scipy
    from .model import _cores, _n_threads
    return {
        "commit": tool_commit(),
        "libraries": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
                      "polars": polars.__version__},
        "kernels": kernels(),
        "threads": {"workers": n_workers(args["workers"]), "cores": _cores(), "model_threads": _n_threads(),
                    "polars_threads": polars.thread_pool_size(),
                    **{v.lower(): os.environ.get(v) for v in THREAD_VARS}},
        "seeds": {"seed": args.get("seed"),
                  "calibration_seed": args["seed"] if args.get("calibration_seed") is None else args["calibration_seed"],
                  "bootstrap": 0},
    }


# options of pipeline.run recorded under another name in params.json (their plain names hold fitted values)
_GIVEN = {"p_unknown": "p_unknown_given", "tau": "tau_given", "temperature": "temperature_given"}
_INPUT_KEYS = {"assembly": "assembly", "pileup": "pileup", "contig_bin": "contig_bin", "motifs_path": "motifs"}


def rerun_arguments(params: dict, out, workers=None, cache_dir=None, log=log) -> dict:
    """Keyword arguments of pipeline.run that repeat the run params.json describes (inputs from their recorded
    paths, every option as it was given), with a new output folder. Says in the log what differs from the
    recorded run: the inputs' size or time, the version, the commit. The worker count and the cache do not change
    decisions; they are taken from the arguments (default: as recorded)."""
    import inspect
    from .pipeline import run
    names = [n for n in inspect.signature(run).parameters]
    kw = {}
    for name in names:
        if name in _INPUT_KEYS:
            rec = params["inputs"][_INPUT_KEYS[name]]
            kw[name] = rec["path"]
            try:
                st = Path(rec["path"]).stat()
                if (st.st_size, int(st.st_mtime)) != (rec["size"], rec["mtime"]):
                    log(f"WARNING: {rec['path']} has changed since the recorded run (size or modification time)")
            except OSError:
                pass  # check_inputs reports it
        elif name == "out":
            kw[name] = str(out)
        elif name == "workers":
            kw[name] = workers if workers is not None else params.get("workers")
        elif name == "cache_dir":
            # the recorded run's cache (its default was <out>/cache): a rerun reads the pileup again only if that is gone
            kw[name] = cache_dir if cache_dir is not None else (params.get("cache_dir")
                                                                or str(Path(params["out"]) / "cache"))
        elif _GIVEN.get(name, name) in params:
            v = params[_GIVEN.get(name, name)]
            kw[name] = tuple(v) if isinstance(v, list) else v
    if params.get("version") != package_version():
        log(f"WARNING: the recorded run used epirefine {params.get('version')}, this is {package_version()}")
    rec, now = (params.get("commit") or {}).get("commit"), tool_commit().get("commit")
    if rec and now and rec != now:
        log(f"WARNING: the recorded run used commit {rec}, this code is {now}")
    return kw


def finish(params: dict) -> dict:
    """Closing entries of params.json."""
    return {**params, "finished": True, "elapsed_s": round(elapsed(), 1), "peak_rss_gb": round(peak_rss_gb(), 3),
            "peak_rss_worker_gb": round(peak_rss_gb(children=True), 3)}
