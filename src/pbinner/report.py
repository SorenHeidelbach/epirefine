"""Self-contained interactive HTML report for one pbinner run directory."""
from __future__ import annotations

import html
import json
import math
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import polars as pl

MOD_NAME = {"a": "6mA", "m": "5mC", "21839": "4mC", "h": "5hmC"}


def _read(path: Path):
    return pl.read_csv(path, separator="\t", infer_schema_length=100_000) if path.exists() else None


def _clean(v):
    """JSON-safe: NaN -> null, +-inf -> "inf"/"-inf", recursively."""
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _rows(df: pl.DataFrame | None):
    return [] if df is None else list(df.iter_rows(named=True))


def _motif_parts(key: str):
    """'GATC_a-1' -> {'seq': 'GATC', 'pos': 1, 'mod': '6mA'}"""
    seq, rest = key.rsplit("_", 1)
    mod, pos = rest.rsplit("-", 1)
    return {"seq": seq, "pos": int(pos), "mod": MOD_NAME.get(mod, mod)}


def collect(run_dir, truth=None) -> dict:
    run = Path(run_dir)
    cal = run / "calibration"
    groups = _read(run / "motif_groups.tsv")
    group_info = []
    for r in groups.iter_rows(named=True):
        members = [_motif_parts(k) for k in r["motifs"].split(",")]
        group_info.append({"group": r["group"], "members": members, "n_sites": r.get("n_sites"),
                           "short": members[0]["seq"] + (f" +{len(members) - 1}" if len(members) > 1 else ""),
                           "mod": members[0]["mod"]})
    data = {
        "run": str(run.resolve()),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": json.loads((run / "params.json").read_text()),
        "groups": group_info,
        "assignments": _rows(_read(run / "assignments.tsv")),
        "posteriors": _rows(_read(run / "posteriors.tsv")),
        "motif_fit": _rows(_read(run / "motif_fit.tsv")),
        "host_profiles": _rows(_read(run / "host_profiles.tsv")),
        "chunks": _rows(_read(run / "host_chunk_outliers.tsv")),
        "detection": _rows(_read(run / "detection_model.tsv")),
        "plasmid_fit": _rows(_read(run / "plasmid_param_fit.tsv")),
        "cal_summary": _rows(_read(cal / "summary.tsv")),
        "reliability": _rows(_read(cal / "reliability.tsv")),
        "assignability": _rows(_read(cal / "assignability.tsv")),
        "tau_fit": _rows(_read(cal / "tau_fit.tsv")),
        "lookalikes": _rows(_read(cal / "lookalikes.tsv")),
        "truth": {},
    }
    if truth:
        t = pl.read_csv(truth, separator="\t", infer_schema_length=100_000)
        data["truth"] = dict(zip(t["contig"].to_list(), t["expected"].to_list()))
    return data


def render(data: dict, title: str) -> str:
    """The page as a fragment: title, styles, markup and scripts, no document skeleton."""
    template = files("pbinner").joinpath("report_template.html").read_text()
    blob = json.dumps(_clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return template.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def write_report(run_dir, out=None, truth=None, title=None, fragment=False) -> Path:
    run = Path(run_dir)
    out = Path(out) if out else run / "report.html"
    page = render(collect(run, truth), title or f"pbinner {run.name}")
    if not fragment:
        page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
    out.write_text(page)
    return out
