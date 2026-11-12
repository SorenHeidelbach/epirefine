"""Golden snapshot of the decisions on the synthetic community (seed 1, default options).

tests/data/golden_decisions.tsv holds, per contig, its given bin, status, new bin and best alternative, and
the probabilities of the decision table rounded to 6 decimals; tests/data/golden_params.json the fitted
parameters. Any change to a label, or to a number by more than the tolerance below, fails with the list of
changed contigs.

A change that is meant to alter results regenerates the snapshot on purpose:

    EPIREFINE_UPDATE_GOLDEN=1 .venv/bin/python -m pytest -q tests/test_golden.py

and commits the new files, so the changed decisions show up in the diff of that commit. Never regenerate to
make an unexpected failure go away: the failure means behaviour changed.

Labels are compared exactly. Numbers are compared with an absolute tolerance of 1e-3 (the outputs are written
with 4 decimals, and the last digit can differ between machines: another CPU or BLAS library rounds sums and
exponentials differently, and so do the optional compiled kernels). The thread count no longer enters (0.12:
tests/test_threads.py checks byte-identical outputs on 1, 4 and 12 threads).
"""
import json
import math
import os
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "golden_decisions.tsv"
GOLDEN_PARAMS = DATA / "golden_params.json"
LABELS = ["given_bin", "status", "new_bin", "best_alternative", "map"]
NUMBERS = ["p_contaminant", "p_best_alternative", "p_map", "p_unknown", "methylome_difference", "redundancy_with_bin"]
PARAMS = ["tau", "temperature", "fitted_temperature", "p_unknown", "contamination_rate", "unknown_eta", "unknown_lam",
          "n_fragments"]
ATOL = 1e-3
UPDATE = os.environ.get("EPIREFINE_UPDATE_GOLDEN") == "1"


def _fmt(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{round(float(x), 6):.6f}"


def _snapshot(run) -> list[str]:
    lines = ["\t".join(["contig"] + LABELS + NUMBERS)]
    for contig, row in sorted(run.decisions.items()):
        lines.append("\t".join([contig] + [row[c] or "-" for c in LABELS] + [_fmt(row[c]) for c in NUMBERS]))
    return lines


def _params(run) -> dict:
    p = json.loads((run.out / "params.json").read_text())

    def number(v):  # params.json is strict JSON since 0.11: an infinite tau is the string "inf"
        return [number(x) for x in v] if isinstance(v, list) else float(v) if v in ("inf", "-inf") else v
    return {k: number(p[k]) for k in PARAMS}


def _same_number(a: str, b: str) -> bool:
    if "NA" in (a, b):
        return a == b
    return abs(float(a) - float(b)) <= ATOL


def _close(a, b) -> bool:
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if math.isinf(a) or math.isinf(b):  # tau = inf (no variation around the bin's level)
        return a == b
    return abs(a - b) <= ATOL * max(1.0, abs(a))


def test_decisions_match_the_golden_snapshot(base_run):
    got = _snapshot(base_run)
    if UPDATE:
        DATA.mkdir(exist_ok=True)
        GOLDEN.write_text("\n".join(got) + "\n")
        pytest.skip(f"golden snapshot rewritten: {GOLDEN.name}")
    want = GOLDEN.read_text().splitlines()
    header = want[0].split("\t")
    assert got[0].split("\t") == header, "snapshot columns changed; regenerate on purpose (see module docstring)"
    exp = {line.split("\t")[0]: line.split("\t") for line in want[1:]}
    new = {line.split("\t")[0]: line.split("\t") for line in got[1:]}
    diff = [f"{c}: only in the golden snapshot" for c in exp.keys() - new.keys()]
    diff += [f"{c}: not in the golden snapshot" for c in new.keys() - exp.keys()]
    for c in sorted(exp.keys() & new.keys()):
        for name, a, b in zip(header[1:], exp[c][1:], new[c][1:]):
            same = a == b if name in LABELS else _same_number(a, b)
            if not same:
                diff.append(f"{c}: {name} {a} -> {b}")
    assert not diff, (f"{len(diff)} differences from tests/data/{GOLDEN.name} (golden -> now):\n  " + "\n  ".join(diff)
                      + "\nIf the change is intended, regenerate with EPIREFINE_UPDATE_GOLDEN=1 and commit the file.")


def test_fitted_parameters_match_the_golden_snapshot(base_run):
    got = _params(base_run)
    if UPDATE:
        GOLDEN_PARAMS.write_text(json.dumps(got, indent=1) + "\n")
        pytest.skip(f"golden parameters rewritten: {GOLDEN_PARAMS.name}")
    want = json.loads(GOLDEN_PARAMS.read_text())
    diff = [f"{k}: {want.get(k)} -> {got.get(k)}" for k in PARAMS if not _close(want.get(k), got.get(k))]
    assert not diff, (f"fitted parameters differ from tests/data/{GOLDEN_PARAMS.name} (golden -> now):\n  "
                      + "\n  ".join(diff) + "\nIf the change is intended, regenerate with EPIREFINE_UPDATE_GOLDEN=1.")


def test_snapshot_comparison_reports_changes():
    """The comparison itself: labels exactly, numbers within the tolerance, missing values only with each other."""
    assert _same_number("0.998600", "0.998700") and not _same_number("0.998600", "0.996000")
    assert _same_number("NA", "NA") and not _same_number("NA", "0.000000")
    assert _close([30.0, float("inf")], [30.0, float("inf")]) and not _close([30.0, float("inf")], [30.0, 1e4])
    assert _fmt(float("nan")) == "NA" and _fmt(0.12345678) == "0.123457"
