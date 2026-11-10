"""Run epirefine on the benchmark suite and record accuracy, speed and memory per version.

    python benchmarks/benchmark_suite.py                   # monoculture training communities (~5 min)
    python benchmarks/benchmark_suite.py --all             # also the held-out test and validation sets
    python benchmarks/benchmark_suite.py --only mono_train_1
    python benchmarks/benchmark_suite.py --suite <other suite.json>   # e.g. the workflow repository's benchmarks

Nothing here names a location on a particular machine. The benchmark data live under a data root given with
--data-root or the EPIREFINE_DATA environment variable; a suite file lists its scenarios' inputs relative to
<data root>/<the suite's "root" entry> (absolute paths in a suite are used as they are). Run outputs go to
--out-root (default <data root>/runs/epirefine/dev/bench_runs) and the per-version records to the development
log given with --devlog or EPIREFINE_DEVLOG.

Scenarios are listed in benchmarks/suite_monocultures.json (benchmarks/monoculture_sets.py); the held-out test and
validation communities are optional and are for final evaluation only, never for tuning. Each run reuses its
dataset's site cache, so wall time measures everything after the pileup pass unless the cache had to be rebuilt
(column `cache_reused`). One summary row per benchmark group and version is appended to benchmarks/results.tsv in
this repository and to <devlog>/benchmarks/history.tsv, per-scenario rows go to
<devlog>/benchmarks/runs/<version>_<commit>.tsv, and the change against the last recorded version of each group is
printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
from evaluate_refinement import evaluate  # noqa: E402

METRICS = ["contam_recall", "contam_precision", "inclusion_recall", "placed_wrongly", "should_stay_unbinned",
           "contamination_after", "completeness_after", "wall_s", "peak_rss_gb"]


PATH_KEYS = ("assembly", "pileup", "contig_bin", "motifs", "truth", "cache")


def data_root(arg: str | None = None) -> Path:
    """The benchmark data root: `arg` (--data-root) or $EPIREFINE_DATA. No machine-specific default."""
    root = arg or os.environ.get("EPIREFINE_DATA")
    if not root:
        sys.exit("benchmark data root not set: pass --data-root <dir> or set the EPIREFINE_DATA environment variable "
                 "(the directory that holds benchmarks/mono_bench)")
    return Path(root)


def load_suite(path: str | Path, root: str | Path | None = None) -> list[dict]:
    """Scenarios of a suite file with their input paths made absolute.

    Paths in a suite are relative to <data root>/<suite "root">; absolute ones are kept. The data root (`root`,
    else $EPIREFINE_DATA) is only required when the suite holds a relative path.
    """
    suite = json.loads(Path(path).read_text())
    base = None
    out = []
    for sc in suite["scenarios"]:
        sc = dict(sc)
        for k in PATH_KEYS:
            if sc.get(k) and not os.path.isabs(sc[k]):
                if base is None:
                    base = data_root(str(root) if root else None) / suite.get("root", "")
                sc[k] = str(base / sc[k])
        out.append(sc)
    return out


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def version() -> str:
    m = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else "unknown"


def run_scenario(sc: dict, out_root: Path, workers: int = 16, run_args: list[str] | None = None) -> dict:
    out = out_root / sc["name"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/time", "-v", str(REPO / ".venv/bin/epirefine"), "run", "--assembly", sc["assembly"], "--pileup", sc["pileup"],
           "--contig-bin", sc["contig_bin"], "--motifs", sc["motifs"], "--out", str(out), "--workers", str(workers)]
    if sc.get("cache"):
        cmd += ["--cache-dir", sc["cache"]]
    cmd += run_args or []
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    (out / "run.log").write_text(p.stdout + p.stderr)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    row = {"scenario": sc["name"], "group": sc["group"], "status": "ok" if p.returncode == 0 else f"failed ({p.returncode})",
           "wall_s": round(wall, 1), "peak_rss_gb": round(int(rss.group(1)) / 1e6, 2) if rss else None,
           "cache_reused": "reusing site cache" in p.stdout}
    m = re.search(r"refinement: p_unknown=([\d.]+), contamination rate=([\d.]+)", p.stdout)
    if m:
        row.update(p_unknown=float(m.group(1)), contamination_rate=float(m.group(2)))
    if p.returncode == 0 and sc.get("truth"):
        r = evaluate(sc["truth"], out / "bin_contamination.tsv", out / "new_contig_bin.tsv", "epirefine",
                     out / "contig_decisions.tsv")
        row.update({k: r[k] for k in r if k != "tool"})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(REPO / "benchmarks/suite_monocultures.json"))
    ap.add_argument("--results", default=str(REPO / "benchmarks/results.tsv"),
                    help="per-version summary kept in the repository (one row per benchmark group and run)")
    ap.add_argument("--data-root", default=None,
                    help="directory holding the benchmark data (default: the EPIREFINE_DATA environment variable); "
                         "suite paths are relative to <data root>/<suite root>")
    ap.add_argument("--devlog", default=os.environ.get("EPIREFINE_DEVLOG"),
                    help="development log directory; records go to <devlog>/benchmarks "
                         "(default: the EPIREFINE_DEVLOG environment variable)")
    ap.add_argument("--out-root", default=None, help="run outputs (default: <data root>/runs/epirefine/dev/bench_runs)")
    ap.add_argument("--all", action="store_true", help="include optional scenarios")
    ap.add_argument("--only", nargs="+", help="run only these scenario names")
    ap.add_argument("--note", default="", help="what changed, stored with the history row")
    ap.add_argument("--run-args", default="",
                    help="extra options passed to every `epirefine run` (quoted), to benchmark a non-default setting, "
                         "e.g. --run-args='--calibration legacy'")
    ap.add_argument("--jobs", type=int, default=1,
                    help="scenarios run at once (faster; timings then aren't comparable with jobs=1 records)")
    a = ap.parse_args()
    if not a.devlog:
        sys.exit("development log not set: pass --devlog <dir> or set the EPIREFINE_DEVLOG environment variable")
    scen = load_suite(a.suite, a.data_root)
    scen = [s for s in scen if (s["name"] in a.only if a.only else a.all or not s.get("optional"))]
    ver, sha = version(), git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    tag = f"{ver}_{sha}{'-dirty' if dirty else ''}"
    out_root = (Path(a.out_root) if a.out_root else data_root(a.data_root) / "runs/epirefine/dev/bench_runs") / tag
    rows = []
    workers = max(1, 16 // a.jobs)

    def one(sc):
        print(f"[{datetime.now():%H:%M:%S}] {sc['name']} ...", flush=True)
        r = run_scenario(sc, out_root, workers, shlex.split(a.run_args))
        print(f"    {sc['name']}: {r['status']}, {r['wall_s']} s, {r['peak_rss_gb']} GB" +
              (f", contaminants found {r['contam_recall']:.3f}, put back {r['inclusion_recall']:.3f}, "
               f"wrongly placed {r['placed_wrongly']}" if "contam_recall" in r else ""), flush=True)
        return r

    if a.jobs > 1:
        # scenarios sharing a site cache must not build it at the same time: one per cache first
        first, rest, seen = [], [], set()
        for sc in scen:
            (rest if sc.get("cache") in seen else first).append(sc)
            seen.add(sc.get("cache"))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(one, first)) + list(ex.map(one, rest))
    else:
        rows = [one(sc) for sc in scen]
    rows = [dict(r, jobs=a.jobs) for r in rows]
    df = pl.DataFrame(rows, infer_schema_length=None)
    bdir = Path(a.devlog) / "benchmarks"
    (bdir / "runs").mkdir(parents=True, exist_ok=True)
    df.write_csv(bdir / "runs" / f"{tag}.tsv", separator="\t", float_precision=4)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    cols = [c for c in METRICS if c in df.columns]
    summ = (df.group_by("group", maintain_order=True)
            .agg(pl.len().alias("scenarios"), *[pl.col(c).mean() for c in cols], pl.col("jobs").first())
            .with_columns(pl.lit(ver).alias("version"), pl.lit(sha + ("-dirty" if dirty else "")).alias("commit"),
                          pl.lit(now).alias("date"), pl.lit(a.note).alias("note")))
    hist_path = bdir / "history.tsv"
    hist = pl.read_csv(hist_path, separator="\t") if hist_path.exists() else None
    summ = summ.select("date", "version", "commit", "group", "scenarios", *cols, "jobs", "note")
    if hist is not None:
        print("\nChange against the last recorded version of each benchmark:")
        for r in summ.iter_rows(named=True):
            prev = hist.filter(pl.col("group") == r["group"])
            if not prev.height:
                print(f"  {r['group']}: first record")
                continue
            p = prev.row(-1, named=True)
            parts = []
            same_jobs = (p.get("jobs") or 1) == r["jobs"]
            for c in cols:
                if p.get(c) is None or r[c] is None or (c in ("wall_s", "peak_rss_gb") and not same_jobs):
                    continue
                d = r[c] - p[c]
                parts.append(f"{c} {r[c]:.4g} ({d:+.3g})")
            print(f"  {r['group']} vs {p['version']} ({p['commit']}): " + ", ".join(parts))
        hist = pl.concat([hist, summ], how="diagonal_relaxed")
    else:
        hist = summ
    hist.write_csv(hist_path, separator="\t", float_precision=4)
    res_path = Path(a.results)
    res = pl.read_csv(res_path, separator="\t") if res_path.exists() else None
    res = summ if res is None else pl.concat([res, summ], how="diagonal_relaxed")
    res.write_csv(res_path, separator="\t", float_precision=4)
    print(f"\nrows: {bdir / 'runs' / (tag + '.tsv')}\nhistory: {hist_path}\nresults (repository): {res_path}")


if __name__ == "__main__":
    main()
