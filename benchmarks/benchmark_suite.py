"""Run epirefine on the benchmark suite and record accuracy, speed and memory per version.

    python benchmarks/benchmark_suite.py                   # default scenarios (~15 min)
    python benchmarks/benchmark_suite.py --all             # also optional ones (human_gut, ~1 h)
    python benchmarks/benchmark_suite.py --only sim_s1     # chosen scenarios

Scenarios are listed in benchmarks/suite.json. Each run reuses its dataset's site cache, so wall
time measures everything after the pileup pass unless the cache had to be rebuilt (column
`cache_reused`). Per-scenario rows go to <devlog>/benchmarks/runs/<version>_<commit>.tsv; one
summary row per benchmark group is appended to <devlog>/benchmarks/history.tsv, and the change
against the last recorded version of each group is printed.
"""
from __future__ import annotations

import argparse
import json
import re
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


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def version() -> str:
    m = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1) if m else "unknown"


def run_scenario(sc: dict, out_root: Path, workers: int = 16) -> dict:
    out = out_root / sc["name"]
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/time", "-v", str(REPO / ".venv/bin/epirefine"), "run", "--assembly", sc["assembly"], "--pileup", sc["pileup"],
           "--contig-bin", sc["contig_bin"], "--motifs", sc["motifs"], "--out", str(out), "--workers", str(workers)]
    if sc.get("cache"):
        cmd += ["--cache-dir", sc["cache"]]
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
    ap.add_argument("--suite", default=str(REPO / "benchmarks/suite.json"))
    ap.add_argument("--devlog", default=str(Path.home() / "plasmid_binner_devlog"))
    ap.add_argument("--out-root", default="/home/ubuntu/data/bench_runs")
    ap.add_argument("--all", action="store_true", help="include optional scenarios")
    ap.add_argument("--only", nargs="+", help="run only these scenario names")
    ap.add_argument("--note", default="", help="what changed, stored with the history row")
    ap.add_argument("--jobs", type=int, default=1,
                    help="scenarios run at once (faster; timings then aren't comparable with jobs=1 records)")
    a = ap.parse_args()
    scen = json.loads(Path(a.suite).read_text())["scenarios"]
    scen = [s for s in scen if (s["name"] in a.only if a.only else a.all or not s.get("optional"))]
    ver, sha = version(), git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    tag = f"{ver}_{sha}{'-dirty' if dirty else ''}"
    out_root = Path(a.out_root) / tag
    rows = []
    workers = max(1, 16 // a.jobs)

    def one(sc):
        print(f"[{datetime.now():%H:%M:%S}] {sc['name']} ...", flush=True)
        r = run_scenario(sc, out_root, workers)
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
    print(f"\nrows: {bdir / 'runs' / (tag + '.tsv')}\nhistory: {hist_path}")


if __name__ == "__main__":
    main()
