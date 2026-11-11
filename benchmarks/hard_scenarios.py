"""Harder planted contamination than perturb_bins.py's, built from an existing scenario.

perturb_bins.py moves a random 6% of contigs into a uniformly random other bin, whatever their length, coverage
or relatedness. A binner errs the other way: it misplaces contigs that look like the bin. This script takes a
scenario made by perturb_bins.py (its truth_actions.tsv: the orphan genomes and the unbinned members are kept),
puts its planted contaminants back, and plants the same number again under one of these variants:

  related    only between genomes of the same genus, else family, else order (--taxonomy; genus defaults to the
             genome name before the first "-"); communities without such a pair give no scenario
  nearest    into the bin whose tetranucleotide composition is nearest to the contaminant's genome, with the
             contaminant's reads thinned to that bin's coverage (the composition- and coverage-nearest error)
  covmatch   random other bin, reads thinned to that bin's coverage
  short      random other bin, contaminants of 2-5 kb only
  cov5/cov20 random other bin, the contaminant's reads thinned to 5x / 20x
  multi      five contigs of one genome into one bin, bin after bin

Thinning is the binomial thinning of monoculture_sets.py, applied to the contaminant's rows of the scenario's
site cache (a copy; the pileup itself is not rewritten, and the run must be given the same assembly, pileup and
motifs so that the copied cache is accepted). Coverage is the pipeline's: median reads per motif site.

Writes <out>/<scenario>__<variant>/ with truth_actions.tsv, contig_bin.tsv, planting.tsv (contig, source, bin,
relation, thin factor) and, for thinned variants, cache/.

    python benchmarks/hard_scenarios.py --truth scen/truth_actions.tsv --cache scen/pbcache --variant nearest --out hard
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from trivial_baselines import load_features  # noqa: E402

VARIANTS = {
    "related":  dict(dest="related"),
    "nearest":  dict(dest="nearest", coverage="match"),
    "covmatch": dict(coverage="match"),
    "short":    dict(length=(2_000, 5_000)),
    "cov5":     dict(coverage=5.0),
    "cov20":    dict(coverage=20.0),
    "multi":    dict(group=5),
}
RANKS = ("genus", "family", "order")


def reset(truth: pl.DataFrame) -> pl.DataFrame:
    """The scenario before its contaminants were planted (members back in their bin, orphans unbinned)."""
    t = truth.with_columns(pl.col("true_bin", "given_bin").fill_null(""))
    return t.with_columns(
        pl.when(pl.col("category") == "contaminant").then(pl.col("true_bin"))
          .when(pl.col("category") == "orphan_contaminant").then(pl.lit("")).otherwise(pl.col("given_bin")).alias("given_bin"),
        pl.col("category").replace({"contaminant": "member", "orphan_contaminant": "orphan"}))


def relation(a: str, b: str, taxonomy: dict) -> str:
    """Lowest shared rank of two genomes ('' when none)."""
    for r in RANKS:
        if taxonomy.get(a, {}).get(r) and taxonomy.get(a, {}).get(r) == taxonomy.get(b, {}).get(r):
            return r
    return ""


def plant(truth: pl.DataFrame, genome: dict, n: int, rng, *, dest: str = "random", coverage=None, length=None,
          group: int = 1, genome_cov: dict | None = None, dist: dict | None = None, taxonomy: dict | None = None,
          min_keep: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Plant about n contaminants into a clean scenario (reset()).

    genome: contig -> genome of origin (a kept bin's name, or an orphan genome). genome_cov: genome -> coverage
    (for coverage='match' or a number: only sources with enough reads to be thinned to the target are used).
    dist: (genome, bin) -> composition distance (dest='nearest'). Returns the new truth and the planting table."""
    contig, cat = truth["contig"].to_list(), dict(zip(truth["contig"], truth["category"]))
    given, size = dict(zip(truth["contig"], truth["given_bin"])), dict(zip(truth["contig"], truth["length"]))
    kept = sorted(b for b in set(given.values()) if b)
    left = {b: sum(1 for c in contig if given[c] == b) for b in kept}
    lo, hi = length or (0, 10**12)
    pool = [c for c in contig if cat[c] in ("member", "orphan") and lo <= size[c] <= hi]
    taxonomy, genome_cov = taxonomy or {}, genome_cov or {}

    def targets(g):
        """Bins a contig of genome g may be planted into, best first."""
        ds = [b for b in kept if b != g]
        if coverage == "match":
            ds = [b for b in ds if genome_cov[g] >= genome_cov[b]]
        elif coverage is not None and genome_cov[g] < 0.85 * coverage:
            ds = []
        if dest == "related":
            rank = {b: RANKS.index(relation(g, b, taxonomy)) for b in ds if relation(g, b, taxonomy)}
            ds = [b for b in rank if rank[b] == min(rank.values())]
        if dest == "nearest" and ds:
            ds = [min(ds, key=lambda b: dist[(g, b)])]
        return ds

    rows = []

    def move(c, b):
        g = genome[c]
        if cat[c] == "member":
            left[g] -= 1
        factor = 1.0
        if coverage == "match":
            factor = genome_cov[b] / genome_cov[g]
        elif coverage is not None:
            factor = min(1.0, coverage / genome_cov[g])
        rows.append({"contig": c, "source": g, "bin": b, "relation": relation(g, b, taxonomy), "thin": factor})
        given[c] = b
        cat[c] = "contaminant" if cat[c] == "member" else "orphan_contaminant"

    def free(c):   # a member may leave only while its bin keeps min_keep contigs
        return cat[c] == "orphan" or (cat[c] == "member" and left[genome[c]] > min_keep)

    if group > 1:   # several contigs of one genome into one bin
        by_genome = {}
        for c in rng.permutation(pool):
            by_genome.setdefault(genome[c], []).append(c)
        pairs = [(g, b) for g in sorted(by_genome) for b in targets(g)]
        for i in rng.permutation(len(pairs)):
            if len(rows) >= n:
                break
            g, b = pairs[i]
            take = [c for c in by_genome[g] if free(c)][:group]
            if len(take) == group:
                for c in take:
                    move(c, b)
    else:
        for c in rng.permutation(pool):
            if len(rows) >= n:
                break
            ds = targets(genome[c]) if free(c) else []
            if ds:
                move(c, str(rng.choice(ds)))
    out = truth.with_columns(pl.col("contig").replace_strict(given).alias("given_bin"),
                             pl.col("contig").replace_strict(cat).alias("category"))
    schema = {"contig": pl.Utf8, "source": pl.Utf8, "bin": pl.Utf8, "relation": pl.Utf8, "thin": pl.Float64}
    return out, pl.DataFrame(rows, schema=schema)


def thin_cache(src: Path, dst: Path, factors: dict[str, float], seed: int) -> None:
    """Copy a site cache with the reads of some contigs thinned (each read kept with probability f)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("sites"))
    (dst / "sites").mkdir()
    names = json.loads((src / "asm" / "assembly.json").read_text())["names"]
    f = np.ones(len(names))
    for c, x in factors.items():
        f[names.index(c)] = x
    rng = np.random.default_rng(seed)
    for part in sorted((src / "sites").glob("*.parquet")):
        d = pl.read_parquet(part)
        p = f[d["contig"].to_numpy()]
        if (p < 1).any():
            k, n = d["k"].to_numpy().astype(np.int64), d["n"].to_numpy().astype(np.int64)
            k2 = rng.binomial(k, p)
            n2 = k2 + rng.binomial(n - k, p)
            d = d.with_columns(pl.Series("k", k2, dtype=d.schema["k"]), pl.Series("n", n2, dtype=d.schema["n"])).filter(pl.col("n") > 0)
        d.write_parquet(dst / "sites" / part.name)


def genome_stats(cache: Path, genome: dict, kept: list[str]) -> tuple[dict, dict]:
    """Per genome: coverage (median over its contigs of the median reads per site) and the distance of its
    tetranucleotide composition to each kept bin's (length-weighted mean frequencies)."""
    f = load_features(cache)
    g = np.array([genome.get(c, "") for c in f["names"]])
    cov, comp = {}, {}
    for x in sorted(set(g) - {""}):
        m = g == x
        cov[x] = float(np.median(f["cov_median"][m]))
        comp[x] = (f["length"][m][:, None] * f["tnf"][m]).sum(0) / f["length"][m].sum()
    dist = {(a, b): float(np.linalg.norm(comp[a] - comp[b])) for a in comp for b in kept if b in comp}
    return cov, dist


def build(truth_path, cache, variant: str, out, seed: int = 1, taxonomy: dict | None = None, genome: dict | None = None):
    """Write one variant of one scenario; returns its directory, or None when nothing could be planted."""
    truth = pl.read_csv(truth_path, separator="\t")
    n = int(truth["category"].is_in(["contaminant", "orphan_contaminant"]).sum())
    clean = reset(truth)
    genome = genome or {c: c.split("__")[0] for c in clean["contig"]}   # monoculture_sets.py's contig names
    kept = sorted(b for b in set(clean["given_bin"]) if b)
    spec = VARIANTS[variant]
    cov, dist = genome_stats(Path(cache), genome, kept) if (spec.get("coverage") or spec.get("dest") == "nearest") else ({}, {})
    if taxonomy is None:
        taxonomy = {g: {"genus": g.split("-")[0]} for g in set(genome.values())}
    new, planted = plant(clean, genome, n, np.random.default_rng(seed), genome_cov=cov, dist=dist, taxonomy=taxonomy, **spec)
    if not planted.height:
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    new.write_csv(out / "truth_actions.tsv", separator="\t")
    new.filter(pl.col("given_bin") != "").select("contig", "given_bin").write_csv(out / "contig_bin.tsv", separator="\t", include_header=False)
    planted.write_csv(out / "planting.tsv", separator="\t", float_precision=4)
    thin = {c: x for c, x in zip(planted["contig"], planted["thin"]) if x < 1}
    if spec.get("coverage"):
        thin_cache(Path(cache), out / "cache", thin, seed)
    return out


def read_taxonomy(path) -> dict:
    t = pl.read_csv(path, separator="\t", infer_schema_length=0)
    return {r["genome"]: {k: r.get(k) for k in RANKS} for r in t.iter_rows(named=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="truth_actions.tsv of a perturb_bins.py scenario")
    ap.add_argument("--cache", required=True, help="site cache of that scenario (read; thinned variants get a copy)")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--taxonomy", help="genome, genus, family, order (TSV) for --variant related")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    d = build(a.truth, a.cache, a.variant, a.out, a.seed, read_taxonomy(a.taxonomy) if a.taxonomy else None)
    print(f"{a.variant}: " + (f"written to {d}" if d else "no contig could be planted"))


if __name__ == "__main__":
    main()
