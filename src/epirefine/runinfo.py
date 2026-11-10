"""Run bookkeeping: the log, the checks made on the inputs and options before any heavy work, and params.json,
the record of what a run was made with.

Nothing here touches the model; a run with valid inputs gives the same decisions with or without it.
"""
from __future__ import annotations

import json
import math
import os
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
        f"workers={n_workers(args['workers'])}; seed={args['seed']}")
    check_settings(args)
    if args["contam_eps"] == 0:
        log("--contam-eps 0: no robust bin profiles and no regional model (long contigs are judged as a whole)")


def n_workers(workers) -> int:
    return workers or min(16, os.cpu_count() or 1)


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
    if not kept:
        raise InputError(
            f"no bin left: none of the {len(cb)} contigs in {path} is in the assembly (contig-bin names e.g. "
            f"{', '.join(map(repr, missing[:3]))}; assembly names e.g. {', '.join(map(repr, list(lengths)[:3]))}). "
            f"The table needs contig<TAB>bin with the contig names of the assembly (the first word of each "
            f"FASTA header) in column 1.")
    if missing:
        share = len(kept) / len(cb)
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
    prov.update({given.get(k, k): v for k, v in args.items() if k not in skip})
    return prov


def finish(params: dict) -> dict:
    """Closing entries of params.json."""
    return {**params, "finished": True, "elapsed_s": round(elapsed(), 1), "peak_rss_gb": round(peak_rss_gb(), 3),
            "peak_rss_worker_gb": round(peak_rss_gb(children=True), 3)}
