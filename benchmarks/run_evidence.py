"""Evidence a methods referee asks for, on the development (training) communities only, in one command.

    python benchmarks/run_evidence.py --out <dir> [--taxonomy genome_ranks.tsv] [--jobs 4]

Stages (all by default; each skips what <dir> already holds, so the command can be rerun or resumed):

  build      copies each training scenario's site cache into <dir>/caches (the shared benchmark data are only
             read) and plants the harder contamination variants of benchmarks/hard_scenarios.py
  run        EpiRefine with default settings on every scenario, and every setting of ABLATIONS (existing
             command-line options only: one rule switched off or moved) on the --ablation-groups scenarios
  baselines  benchmarks/trivial_baselines.py scores for every scenario
  report     <dir>/tables/*.tsv and <dir>/report.md: EpiRefine against each baseline (precision-recall area,
             recall at EpiRefine's precision), by coverage and length class, with undecided contigs counted as
             misses and isolate-clustered bootstrap intervals; the ablation table with scenario-bootstrap
             intervals of every difference, and the settings that changed no call anywhere

Only the scenarios of --suite that are not marked optional are used (the held-out test and validation
communities are optional and are never touched). Nothing is written outside <dir>: no row goes to
benchmarks/results.tsv or the development log.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
import evidence_metrics as em  # noqa: E402
import hard_scenarios as hs  # noqa: E402
import trivial_baselines as tb  # noqa: E402
from benchmark_suite import load_suite  # noqa: E402
from evaluate_refinement import evaluate  # noqa: E402

# one rule off or moved per entry; rule numbers are those of the 2026-11-10 decision-rule review
ABLATIONS = {
    "coverage gate 0x (rule 8)": ["--min-contig-coverage", "0"],
    "coverage gate 10x (rule 8)": ["--min-contig-coverage", "10"],
    "methylome gate 0 (rule 3)": ["--min-methylome-difference", "0"],
    "methylome gate 0.2 (rule 3)": ["--min-methylome-difference", "0.2"],
    "methylome gate 0.6 (rule 3)": ["--min-methylome-difference", "0.6"],
    "no noise filter (rule 34)": ["--no-motif-noise-filter"],
    "no group weights (rule 35)": ["--no-group-weights"],
    "no regional model (rules 11-14)": ["--no-regional-model"],
    "unmethylated-bin threshold 0.9 (rule 5)": ["--include-threshold-unmethylated", "0.9"],
    "temperature 1 (rule 23)": ["--temperature", "1"],
    "temperature 3 (rule 23)": ["--temperature", "3"],
    "sharpening allowed (rule 23)": ["--allow-sharpening"],
    "no calibration (rules 20-30)": ["--no-calibrate"],
    "no redundancy check (rule 16)": ["--max-redundancy", "1"],
    "inclusion threshold 0.8 (rule 4)": ["--include-threshold", "0.8"],
    "inclusion threshold 0.95 (rule 4)": ["--include-threshold", "0.95"],
    "contamination threshold 0.8 (rule 1)": ["--contamination-threshold", "0.8"],
    "contamination threshold 0.95 (rule 1)": ["--contamination-threshold", "0.95"],
    "no robust profiles (rule 31)": ["--contam-eps", "0"],
}
NEEDS_SWITCH = [
    "unknown-model parsimony, 2 bootstrap SE (rule 29)", "swap cap eta <= 0.05 and the unknown-model grids (rule 28)",
    "Beta(1, 19) prior on the contamination rate (rule 2)", "definition of an unmethylated bin, 0.5 / 0.5 (rule 6)",
    "regional switch probability 0.05 (rule 12)", "noise-filter constants (rule 34; only on/off)",
    "group-weight constants (rule 35; only on/off)", "caller labels r < 0.2 / r > 0.5 (rule 36)",
    "look-alike distance 0.15 and decidable BF 3 in calibration (rules 26, 27)",
    "prior p_unknown 0.1 during calibration (rule 24)"]
UNDECIDED = ("low_coverage", "too_short", "no_methylation_evidence")
BASELINES = ("coverage", "tetra", "gc", "cov_tetra", "naive_meth", "naive_meth2")
STRATA = (("all", 0.0), (">= 5x", 5.0), (">= 10x", 10.0))


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in name)


# ---------------------------------------------------------------- build
def build(a, out: Path) -> list[dict]:
    base = [s for s in load_suite(a.suite, a.data_root) if not s.get("optional") and (not a.only or s["name"] in a.only)]
    taxonomy = hs.read_taxonomy(a.taxonomy) if a.taxonomy else None
    scen = []
    for s in base:
        cache = out / "caches" / s["name"]
        if not cache.exists():
            shutil.copytree(s["cache"], cache)
        common = {k: s[k] for k in ("assembly", "pileup", "motifs")}
        scen.append({"name": s["name"], "base": s["name"], "group": "standard", **common,
                     "contig_bin": s["contig_bin"], "truth": s["truth"], "cache": str(cache)})
        for i, v in enumerate(sorted(hs.VARIANTS)):
            d = out / "scenarios" / f"{s['name']}__{v}"
            if not (d / "planting.tsv").exists() and not (d / "skipped").exists():
                if hs.build(s["truth"], cache, v, d, seed=a.seed + i, taxonomy=taxonomy) is None:
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "skipped").write_text("no contig could be planted\n")
            if (d / "planting.tsv").exists():
                scen.append({"name": d.name, "base": s["name"], "group": v, **common, "contig_bin": str(d / "contig_bin.tsv"),
                             "truth": str(d / "truth_actions.tsv"), "cache": str(d / "cache" if (d / "cache").exists() else cache)})
                print(f"  {d.name}: {pl.read_csv(d / 'planting.tsv', separator=chr(9)).height} contaminants", flush=True)
    (out / "scenarios.json").write_text(json.dumps(scen, indent=1))
    return scen


# ---------------------------------------------------------------- run
def run_one(sc: dict, extra: list[str], d: Path, workers: int) -> None:
    if (d / "contig_decisions.tsv").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    cmd = [str(REPO / ".venv/bin/epirefine"), "run", "--assembly", sc["assembly"], "--pileup", sc["pileup"], "--contig-bin",
           sc["contig_bin"], "--motifs", sc["motifs"], "--out", str(d / "tmp"), "--workers", str(workers), "--cache-dir", sc["cache"], *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (d / "run.log").write_text(p.stdout + p.stderr)
    if p.returncode or "reusing site cache" not in p.stdout:
        print(f"  FAILED or cache rebuilt: {d} (see run.log)", flush=True)
        if p.returncode:
            return
    for f in ("bin_contamination.tsv", "new_contig_bin.tsv", "contig_decisions.tsv", "params.json"):
        if (d / "tmp" / f).exists():   # the decision table last: its presence marks the run as done
            shutil.move(str(d / "tmp" / f), str(d / f))
    shutil.rmtree(d / "tmp", ignore_errors=True)


def run_all(a, out: Path, scen: list[dict]) -> None:
    jobs = [(sc, [], out / "runs" / "default" / sc["name"]) for sc in scen]
    for name, extra in ABLATIONS.items():
        jobs += [(sc, extra, out / "runs" / slug(name) / sc["name"]) for sc in scen if sc["group"] in a.ablation_groups]
    todo = [j for j in jobs if not (j[2] / "contig_decisions.tsv").exists()]
    print(f"run: {len(todo)} of {len(jobs)} EpiRefine runs to do", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, _ in enumerate(ex.map(lambda j: run_one(j[0], j[1], j[2], a.workers), todo)):
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(todo)} runs", flush=True)


# ---------------------------------------------------------------- baselines
def baselines(a, out: Path, scen: list[dict]) -> None:
    (out / "baselines").mkdir(exist_ok=True)
    for sc in scen:
        pre = out / "baselines" / sc["name"]
        if Path(f"{pre}.inclusion.tsv").exists():
            continue
        t = pl.read_csv(sc["truth"], separator="\t").with_columns(pl.col("given_bin").fill_null(""))
        given = {c: b for c, b in zip(t["contig"], t["given_bin"]) if b}
        f = tb.load_features(sc["cache"])
        pl.DataFrame({"contig": f["names"], "coverage": f["cov_median"]}).write_csv(f"{pre}.features.tsv", separator="\t")
        tb.contamination_scores(f, given, a.min_sites).write_csv(f"{pre}.contamination.tsv", separator="\t", float_precision=4)
        tb.inclusion_scores(f, given, a.min_sites).write_csv(f"{pre}.inclusion.tsv", separator="\t", float_precision=4)
        print(f"  baselines: {sc['name']}", flush=True)


# ---------------------------------------------------------------- report
def calls(sc: dict, d: Path, out: Path) -> pl.DataFrame | None:
    """Per-contig events of one EpiRefine run, with the tool's scores, coverage and the contig's isolate."""
    if not (d / "contig_decisions.tsv").exists():
        return None
    truth = pl.read_csv(sc["truth"], separator="\t")
    dec = pl.read_csv(d / "contig_decisions.tsv", separator="\t", infer_schema_length=0)
    flagged = set(pl.read_csv(d / "bin_contamination.tsv", separator="\t", infer_schema_length=0)["contig"].to_list())
    nb = pl.read_csv(d / "new_contig_bin.tsv", separator="\t", infer_schema_length=0)
    t = em.tally(truth, flagged, dict(zip(nb[nb.columns[0]], nb[nb.columns[1]])))
    num = lambda c: pl.col(c).cast(pl.Float64, strict=False).fill_nan(None).fill_null(0.0)  # noqa: E731
    dec = dec.select("contig", pl.col("status").is_in(UNDECIDED).not_().alias("decided"), num("p_contaminant").alias("p_foreign"),
                     pl.col("best_alternative").alias("alt_bin"), num("p_best_alternative").alias("p_alt"))
    cov = pl.read_csv(out / "baselines" / f"{sc['name']}.features.tsv", separator="\t")
    return (t.join(dec, on="contig", how="left").join(cov, on="contig", how="left")
            .with_columns(pl.col("decided").fill_null(False), pl.col("coverage").fill_null(0.0),
                          pl.col("contig").str.split("__").list.first().alias("isolate"),
                          pl.lit(sc["name"]).alias("scenario"), pl.lit(sc["base"]).alias("base"), pl.lit(sc["group"]).alias("group")))


def _sum(df: pl.DataFrame) -> dict:
    return {k: int(df[k].sum()) for k in em.COUNTS}


def _ci(df: pl.DataFrame, by: str, key: str, n_boot: int) -> str:
    """Bootstrap interval of one rate over clusters `by`, as text."""
    g = df.group_by(by, maintain_order=True).agg([pl.col(k).sum() for k in em.COUNTS])
    lo, hi = em.cluster_bootstrap(g.select(em.COUNTS).to_numpy().astype(float),
                                  lambda r: em.rates(dict(zip(em.COUNTS, r)))[key], n_boot)
    return f"{100 * lo:.1f}-{100 * hi:.1f}"


def _weights(isolate: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """[n_boot, items] multiplicities when isolates are resampled with replacement."""
    names, inv = np.unique(isolate, return_inverse=True)
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=n_boot)[:, inv]


def _curve_row(score, correct, positive, isolate, target_precision, n_boot) -> dict:
    """Average precision and recall at the required precisions of one score, with isolate-bootstrap intervals."""
    row = {"average_precision": em.average_precision(score, correct, positive)}
    levels = {"recall_at_epirefine_precision": target_precision, "recall_at_p95": 0.95, "recall_at_p90": 0.90}
    for k, p in levels.items():
        row[k], thr = em.recall_at_precision(score, correct, p, positive)
        if k == "recall_at_epirefine_precision":
            row["threshold"] = thr
    W = _weights(isolate, n_boot)
    ap = [em.average_precision(score, correct, positive, w) for w in W]
    rr = [em.recall_at_precision(score, correct, target_precision, positive, w)[0] for w in W]
    row["average_precision_ci"] = "{:.3f}-{:.3f}".format(*np.percentile(ap, [2.5, 97.5]))
    row["recall_at_epirefine_precision_ci"] = "{:.1f}-{:.1f}".format(*(100 * np.percentile(rr, [2.5, 97.5])))
    return row


def contamination_tables(D: pl.DataFrame, out: Path, n_boot: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """EpiRefine vs baselines on binned contigs, per scenario group and coverage class; and per length class."""
    rows, lens, cells = [], [], []
    for (group,), g in D.filter(pl.col("given_bin") != "").group_by("group", maintain_order=True):
        sc = pl.concat([pl.read_csv(out / "baselines" / f"{s}.contamination.tsv", separator="\t").with_columns(pl.lit(s).alias("scenario"))
                        for s in g["scenario"].unique(maintain_order=True)])
        g = g.join(sc, on=["scenario", "contig"], how="left")
        flagged = (g["tp"] + g["fp"]).to_numpy() > 0
        # EpiRefine's ranking: its flags first (by posterior), then decided contigs by posterior; undecided last
        epi = np.where(flagged, 1 + g["p_foreign"].to_numpy(), np.where(g["decided"].to_numpy(), np.minimum(g["p_foreign"].to_numpy(), 0.999), 0.0))
        scores = {"EpiRefine": epi, **{b: g[b].fill_null(0.0).to_numpy() for b in BASELINES}}
        pos = (g["tp"] + g["fn"]).to_numpy() > 0
        iso, cov, length = g["isolate"].to_numpy(), g["coverage"].to_numpy(), g["length"].to_numpy()
        thr_all = {}
        for stratum, lo in STRATA:
            m = cov >= lo
            c = _sum(g.filter(pl.Series(m)))
            r = em.rates(c)
            dd = g.filter(pl.Series(m) & pl.col("decided"))
            target = r["contam_precision"] if c["tp"] + c["fp"] else 1.0
            for tool, s in scores.items():
                row = {"group": group, "coverage": stratum, "tool": tool, "scenarios": g["scenario"].n_unique(),
                       "binned_contigs": int(m.sum()), "contaminants": int(pos[m].sum())}
                if tool == "EpiRefine":
                    row.update(flags=c["tp"] + c["fp"], precision=r["contam_precision"], precision_ci=_ci(g.filter(pl.Series(m)), "isolate", "contam_precision", 2000),
                               recall_all=r["contam_recall"], recall_all_ci=_ci(g.filter(pl.Series(m)), "isolate", "contam_recall", 2000),
                               recall_decided=em.rates(_sum(dd))["contam_recall"], undecided_contaminants=int((pos & m & ~g["decided"].to_numpy()).sum()))
                row.update(_curve_row(s[m], pos[m], None, iso[m], target, n_boot))
                rows.append(row)
                if stratum == "all":
                    thr_all[tool] = row["threshold"]
        # per length class: share of planted contaminants found, EpiRefine's calls vs each baseline at the
        # threshold that gives it EpiRefine's pooled precision (chosen with the truth: generous to the baseline)
        cls = em.length_class(length)
        for _, _, name in em.LENGTH_CLASSES:
            m = (cls == name) & pos
            if not m.any():
                continue
            row = {"group": group, "length": name, "contaminants": int(m.sum()), "undecided": int((m & ~g["decided"].to_numpy()).sum()),
                   "EpiRefine": float(flagged[m].mean())}
            row.update({b: float((scores[b][m] >= thr_all[b]).mean()) for b in BASELINES})
            lens.append(row)
        # which contaminants does methylation add? composition/coverage outlier (cov_tetra at 90% precision) x EpiRefine flag
        _, t90 = em.recall_at_precision(scores["cov_tetra"], pos, 0.90)
        outlier = scores["cov_tetra"] >= t90
        cells.append({"group": group, "contaminants": int(pos.sum()),
                      "both": int((pos & outlier & flagged).sum()), "epirefine_only": int((pos & ~outlier & flagged).sum()),
                      "cov_tetra_only": int((pos & outlier & ~flagged).sum()), "neither": int((pos & ~outlier & ~flagged).sum()),
                      "cov_tetra_false_flags": int((~pos & outlier).sum()), "epirefine_false_flags": int((~pos & flagged).sum())})
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(lens), pl.DataFrame(cells)


def inclusion_table(D: pl.DataFrame, out: Path, n_boot: int) -> pl.DataFrame:
    """Placement of the contigs that were unbinned in the input: EpiRefine vs the two inclusion baselines."""
    rows = []
    for (group,), g in D.filter(pl.col("given_bin") == "").group_by("group", maintain_order=True):
        sc = pl.concat([pl.read_csv(out / "baselines" / f"{s}.inclusion.tsv", separator="\t", schema_overrides={"score": pl.Float64, "rule_bin": pl.Utf8, "bin": pl.Utf8})
                        .with_columns(pl.lit(s).alias("scenario")) for s in g["scenario"].unique(maintain_order=True)])
        missing = (g["category"] == "missing").to_numpy()
        true, iso = g["true_bin"].to_numpy(), g["isolate"].to_numpy()
        placed = g["final_bin"].to_numpy() != ""
        right = placed & (g["final_bin"].to_numpy() == true) & missing
        target = right.sum() / max(placed.sum(), 1)
        alt = g["alt_bin"].fill_null("").to_numpy()
        epi_bin = np.where(placed, g["final_bin"].to_numpy(), alt)
        epi = np.where(placed, 1 + g["p_alt"].to_numpy(), np.where(g["decided"].to_numpy() & (alt != "unknown") & (alt != ""), np.minimum(g["p_alt"].to_numpy(), 0.999), -np.inf))
        tools = {"EpiRefine": (epi, epi_bin, None)}
        for b in ("naive_meth", "cov_tetra"):
            j = g.select("scenario", "contig").join(sc.filter(pl.col("baseline") == b), on=["scenario", "contig"], how="left")
            tools[b] = (j["score"].fill_null(-np.inf).to_numpy(), j["bin"].fill_null("").to_numpy(), j["rule_bin"].fill_null("").to_numpy())
        for tool, (s, b, rule) in tools.items():
            ok = missing & (b == true) & (b != "")
            row = {"group": group, "tool": tool, "unbinned_contigs": g.height, "missing": int(missing.sum()), "no_bin_contigs": int((~missing).sum())}
            if tool == "EpiRefine":
                row.update(placed=int(placed.sum()), placement_precision=float(target), put_back=float(right.sum() / max(missing.sum(), 1)),
                           no_bin_placed=int((placed & ~missing).sum()), undecided_missing=int((missing & ~g["decided"].to_numpy()).sum()))
            elif tool == "naive_meth":   # the plain rule: the unique bin with the same motif set
                rp = rule != ""
                row.update(placed=int(rp.sum()), placement_precision=float((rp & missing & (rule == true)).sum() / max(rp.sum(), 1)),
                           put_back=float((rp & missing & (rule == true)).sum() / max(missing.sum(), 1)), no_bin_placed=int((rp & ~missing).sum()))
            row.update(_curve_row(s, ok, missing, iso, float(target), n_boot))
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


ABL_METRICS = ("contam_recall", "contam_precision", "inclusion_recall", "placement_precision")


def ablation_table(out: Path, scen: list[dict], D: pl.DataFrame, n_boot: int) -> pl.DataFrame:
    """Each setting against the default on the same scenarios: pooled metrics (all planted errors in the
    denominator), differences with intervals from a paired bootstrap over scenarios, and calls changed."""
    rows = []
    key = ["scenario", "contig"]
    for name in ABLATIONS:
        parts = [calls(sc, out / "runs" / slug(name) / sc["name"], out) for sc in scen]
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        A = pl.concat(parts)
        for label, groups in (("standard", ["standard"]), ("harder", sorted(set(A["group"]) - {"standard"}))):
            a = A.filter(pl.col("group").is_in(groups))
            if not a.height:
                continue
            d = D.join(a.select(key), on=key, how="semi")
            j = d.select(*key, "flagged", "final_bin").join(a.select(*key, pl.col("flagged").alias("f2"), pl.col("final_bin").alias("b2")), on=key)
            ca = a.group_by("scenario", maintain_order=True).agg([pl.col(k).sum() for k in em.COUNTS]).sort("scenario")
            cd = d.group_by("scenario", maintain_order=True).agg([pl.col(k).sum() for k in em.COUNTS]).sort("scenario")
            both = np.hstack([ca.select(em.COUNTS).to_numpy(), cd.select(em.COUNTS).to_numpy()]).astype(float)
            k = len(em.COUNTS)
            ra, rd = em.rates(_sum(a)), em.rates(_sum(d))
            row = {"setting": name, "scenarios": label, "n_scenarios": ca.height,
                   "flags_changed": int((j["flagged"] != j["f2"]).sum()), "bins_changed": int((j["final_bin"] != j["b2"]).sum())}
            for m in ABL_METRICS:
                lo, hi = em.cluster_bootstrap(both, lambda r, m=m: em.rates(dict(zip(em.COUNTS, r[:k])))[m] - em.rates(dict(zip(em.COUNTS, r[k:])))[m], n_boot)
                row.update({f"{m}_default": rd[m], f"{m}_delta": ra[m] - rd[m], f"{m}_ci": f"{100 * lo:+.1f} to {100 * hi:+.1f}"})
            sa, sd = _sum(a), _sum(d)
            row.update(false_flags=f"{sd['fp']} -> {sa['fp']}", wrong_placements=f"{sd['new_wrong']} -> {sa['new_wrong']}",
                       no_bin_placed=f"{sd['stay_wrong']} -> {sa['stay_wrong']}")
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def md(df: pl.DataFrame, cols: dict) -> str:
    """Markdown table of the chosen columns (name -> header); rates as %, other floats to 3 digits."""
    def fmt(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            pct = any(s in c for s in ("recall", "precision", "put_back", "EpiRefine")) or c in BASELINES
            if c.endswith("_delta"):
                return f"{100 * v:+.1f}"
            return f"{100 * v:.1f}" if pct and "average" not in c else f"{v:.3f}"
        return str(v)
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |" for r in df.iter_rows(named=True))


def report(a, out: Path, scen: list[dict]) -> None:
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    D = pl.concat([c for c in (calls(sc, out / "runs" / "default" / sc["name"], out) for sc in scen) if c is not None])
    D.write_parquet(tables / "calls_default.parquet")
    # the per-contig tally must give evaluate()'s numbers (undecided contigs counted)
    for sc in scen[:2]:
        d = out / "runs" / "default" / sc["name"]
        r = evaluate(sc["truth"], d / "bin_contamination.tsv", d / "new_contig_bin.tsv")
        c = _sum(D.filter(pl.col("scenario") == sc["name"]))
        assert (c["tp"] + c["fp"], c["included_ok"], c["stay_wrong"]) == (r["flagged"], r["included_correct"], r["placed_wrongly"]), sc["name"]
    con, lens, cells = contamination_tables(D, out, a.n_boot)
    inc = inclusion_table(D, out, a.n_boot)
    abl = ablation_table(out, scen, D, 2000)
    for name, t in (("contamination", con), ("contamination_by_length", lens), ("contamination_cells", cells), ("inclusion", inc), ("ablations", abl)):
        t.write_csv(tables / f"{name}.tsv", separator="\t", float_precision=4)
    planted = [pl.read_csv(Path(sc["truth"]).parent / "planting.tsv", separator="\t").with_columns(pl.lit(sc["name"]).alias("scenario"))
               for sc in scen if sc["group"] != "standard"]
    if planted:
        P = pl.concat(planted).join(D.select("scenario", "contig", "group", "decided", "tp", "coverage"), on=["scenario", "contig"])
        rel = (P.group_by("group", "relation", maintain_order=True)
               .agg(pl.len().alias("contaminants"), pl.col("decided").sum().alias("decided"), pl.col("tp").sum().alias("flagged"),
                    pl.col("thin").median().alias("median_thin"), pl.col("coverage").median().alias("median_coverage")).sort("group", "relation"))
        rel.write_csv(tables / "planting_summary.tsv", separator="\t", float_precision=3)
    inert = abl.group_by("setting", maintain_order=True).agg((pl.col("flags_changed") + pl.col("bins_changed")).sum().alias("calls_changed")) if abl.height else None
    txt = ["# Evidence tables (development data only)\n", "## Contamination: EpiRefine and trivial baselines\n",
           "All binned contigs are scored; a contaminant EpiRefine did not judge is a miss. AP = area under the precision-recall "
           "curve. 'recall at EpiRefine precision' picks each baseline's threshold with the truth in hand. Intervals: bootstrap over isolates.\n",
           md(con, {"group": "scenarios", "coverage": "coverage", "tool": "tool", "contaminants": "contaminants", "flags": "flags",
                    "precision": "precision %", "precision_ci": "95% CI", "recall_all": "recall % (all planted)", "recall_all_ci": "95% CI",
                    "recall_decided": "recall % (decided only)", "average_precision": "AP", "average_precision_ci": "AP 95% CI",
                    "recall_at_epirefine_precision": "recall % at EpiRefine precision", "recall_at_epirefine_precision_ci": "95% CI",
                    "recall_at_p95": "recall % at 95% precision", "recall_at_p90": "recall % at 90% precision"}),
           "\n## Contaminants found, by contaminant length (%)\n",
           md(lens, {"group": "scenarios", "length": "length", "contaminants": "contaminants", "undecided": "undecided by EpiRefine",
                     "EpiRefine": "EpiRefine", **{b: b for b in BASELINES}}),
           "\n## What methylation adds over coverage + composition (cov_tetra at its 90%-precision threshold)\n",
           md(cells, {c: c for c in cells.columns}),
           "\n## Inclusion of unbinned contigs\n",
           md(inc, {"group": "scenarios", "tool": "tool", "missing": "to put back", "no_bin_contigs": "no-bin contigs", "placed": "placed",
                    "placement_precision": "placement precision %", "put_back": "put back % (all)", "no_bin_placed": "no-bin placed",
                    "average_precision": "AP", "average_precision_ci": "AP 95% CI", "recall_at_epirefine_precision": "recall % at EpiRefine precision",
                    "recall_at_p90": "recall % at 90% precision"}),
           "\n## Rule ablations (difference to the default, percentage points; 95% interval from a paired bootstrap over scenarios)\n",
           md(abl, {"setting": "setting", "scenarios": "scenarios", "n_scenarios": "n", "flags_changed": "flags changed", "bins_changed": "final bins changed",
                    **{k: v for m in ABL_METRICS for k, v in ((f"{m}_delta", m.replace("_", " ")), (f"{m}_ci", "95% CI"))},
                    "false_flags": "false flags", "wrong_placements": "wrong placements", "no_bin_placed": "no-bin placed"}) if abl.height else "(no ablation runs)",
           "\n## Settings that changed no call in any scenario\n",
           "\n".join(f"- {r['setting']}" for r in inert.iter_rows(named=True) if r["calls_changed"] == 0) if inert is not None else "",
           "\n## Rules that cannot be switched from the command line (need a switch)\n", "\n".join(f"- {x}" for x in NEEDS_SWITCH), ""]
    (out / "report.md").write_text("\n".join(txt))
    print(f"report: {out / 'report.md'}; tables: {tables}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="working directory: caches, scenarios, runs, tables")
    ap.add_argument("--suite", default=str(REPO / "benchmarks/suite_monocultures.json"))
    ap.add_argument("--data-root", default=None, help="benchmark data root (default: EPIREFINE_DATA)")
    ap.add_argument("--only", nargs="+", help="base scenarios to use (default: all that are not optional)")
    ap.add_argument("--taxonomy", help="genome, genus, family, order (TSV) for the 'related' variant")
    ap.add_argument("--stage", nargs="+", default=["build", "run", "baselines", "report"], choices=["build", "run", "baselines", "report"])
    ap.add_argument("--ablation-groups", nargs="+", default=["standard", "nearest", "short", "cov5"],
                    help="scenario groups on which the ablations are run (the default settings run on all)")
    ap.add_argument("--jobs", type=int, default=4, help="EpiRefine runs at once")
    ap.add_argument("--workers", type=int, default=1, help="processes per run")
    ap.add_argument("--min-sites", type=int, default=10, help="sites a motif group needs on a contig in the naive methylation rule")
    ap.add_argument("--n-boot", type=int, default=300, help="bootstrap draws for curve statistics")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scen = build(a, out) if "build" in a.stage else json.loads((out / "scenarios.json").read_text())
    if "baselines" in a.stage:      # before the runs: the report's coverage classes come from here
        baselines(a, out, scen)
    if "run" in a.stage:
        run_all(a, out, scen)
    if "report" in a.stage:
        report(a, out, scen)


if __name__ == "__main__":
    main()
