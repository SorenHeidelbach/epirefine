"""Run bookkeeping: params.json, the record of what a run was made with.

Nothing here touches the model; a run gives the same decisions with or without it.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def package_version() -> str:
    from . import __version__
    return __version__


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
        "workers": args["workers"] or min(16, os.cpu_count() or 1),
        "python": sys.version.split()[0],
    }
    # the options as given (p_unknown, tau and temperature under *_given: their plain keys hold the values
    # used, estimated or calibrated when not given)
    prov.update({given.get(k, k): v for k, v in args.items() if k not in skip})
    return prov
