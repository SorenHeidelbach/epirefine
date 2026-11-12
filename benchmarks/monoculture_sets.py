"""Training, test and validation benchmark sets built from monoculture isolates.

Every isolate is sequenced alone, so the origin of every contig and every read is known. This builds
synthetic communities from them with realistic problems, in three disjoint sets:

  * isolates are split by GENUS into train / test / validation (default 60/20/20, seeded), so close
    relatives never sit on both sides of a split; the E. coli wild type and its dam-/dcm- knockout
    (same genome, different methylome) stay together;
  * each community draws isolates from its split and THINS every isolate's pileup to a target median
    coverage drawn log-uniformly from [--cov-min, --cov-max] (binomial thinning of every count
    column), so low-coverage genomes and the contig coverage cutoff are exercised;
  * every isolate's assembly is cut once into pieces of log-uniform length in [--min-len, --max-len]
    (benchmarks/fragment_assembly.py's cutting) with the pileup remapped to piece coordinates;
    contig names become <isolate>__<contig>__p<i>;
  * one bin per isolate; benchmarks/perturb_bins.py then plants the errors: whole genomes left
    unbinned (their pieces should stay unbinned), members unbinned (should be put back), pieces
    moved into a wrong bin (should be flagged);
  * motifs are Nanomotif's calls (bin-motifs.tsv, with complements) of the BINNED isolates only, as
    Nanomotif run on the bins would see them; pileups are modkit pileups. Both come from the isolate
    folders or, where missing, from the workflow repository's workflow/monocultures.smk (modkit pileup,
    nanomotif motif_discovery).

    python benchmarks/monoculture_sets.py --isolates <data root>/benchmarks/mono_bench/isolates_inputs.tsv \
        --data-root <data root> --suite benchmarks/suite_monocultures.json

The data root is --data-root or the EPIREFINE_DATA environment variable; the sets are written to
<data root>/benchmarks/mono_bench (or --out), and the suite file lists them relative to that directory.

Reproducibility of the piece cuts. Each isolate is cut with its own generator, seeded from the isolate's name by
isolate_seed() (SHA-256, the same in every process). Up to 0.11.0 the seed was Python's built-in hash() of the
name, which is salted per process unless PYTHONHASHSEED is set, so the cuts of the sets stored under
<data root>/benchmarks/mono_bench (built 2026-10-27) cannot be regenerated from code: no PYTHONHASHSEED from 0 to
299 reproduces them, and the default salt is random. The cuts are kept as data instead: isolates/<isolate>.pieces.tsv
(and every community's pieces.tsv) hold each piece's parent, start and end, and fragment_isolate() reuses the cached
pieces of an isolate when they exist, so rebuilding into the same directory keeps the stored cuts. Everything after
the cuts (genus split, community draws, thinning, planted errors) comes from --seed and was reproducible before.
Building into a new directory with this version gives different cuts than the stored sets.

Writes, per community, assembly.fasta, pileup.bed, contig_bin.tsv, bin-motifs.tsv, truth_actions.tsv
(perturb_bins.py's columns) and pieces.tsv (piece, isolate, parent contig, plasmid, coverage factor),
plus isolates.tsv (isolate, genus, split, coverage) and a suite file for benchmark_suite.py with groups
mono-train, mono-test, mono-val.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from fragment_assembly import cut_points  # noqa: E402

from epirefine import io as pio  # noqa: E402

COLS = [f"c{i}" for i in range(1, 19)]  # modkit bedMethyl, 18 columns
COUNTS = ["c12", "c13", "c14", "c15", "c16", "c17", "c18"]  # Nmod Ncanonical Nother Ndelete Nfail Ndiff Nnocall
SUITE_ROOT = "benchmarks/mono_bench"  # where the sets live below the data root; recorded as the suite file's "root"


def isolate_seed(name: str) -> int:
    """Seed of an isolate's cutting generator: the first 4 bytes of the SHA-256 of its name. The same in every
    process (the built-in hash() used before is salted per process; see the module docstring)."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def discover(table: str) -> pl.DataFrame:
    """isolate, assembly, pileup (modkit), motifs (Nanomotif bin-motifs.tsv), plasmids (geNomad, optional)."""
    iso = pl.read_csv(table, separator="\t", infer_schema_length=0).with_columns(pl.col("plasmids").fill_null(""))
    return iso.with_columns(pl.col("isolate").str.split("-").list.first().alias("genus"))


def split_genera(iso: pl.DataFrame, fractions, rng) -> dict[str, str]:
    genera = sorted(iso["genus"].unique().to_list())
    rng.shuffle(genera)
    n = len(genera)
    n_train = int(round(fractions[0] * n))
    n_test = int(round(fractions[1] * n))
    names = ["train"] * n_train + ["test"] * n_test + ["val"] * (n - n_train - n_test)
    return dict(zip(genera, names))


def read_pileup(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", has_header=False, new_columns=COLS, infer_schema_length=10000,
                       schema_overrides={"c1": pl.Utf8, "c4": pl.Utf8, "c6": pl.Utf8, "c9": pl.Utf8})


def fragment_isolate(row: dict, cache: Path, lo: int, hi: int, rng) -> tuple[Path, pl.DataFrame]:
    """Cut the isolate's assembly once; cache pieces (fasta) and the remapped pileup (parquet)."""
    iso = row["isolate"]
    fa, pq, pt = cache / f"{iso}.fasta", cache / f"{iso}.pileup.parquet", cache / f"{iso}.pieces.tsv"
    if fa.exists() and pq.exists() and pt.exists():
        return pq, pl.read_csv(pt, separator="\t")
    seqs = pio.read_fasta(row["assembly"])
    plasmids = set()
    if row["plasmids"]:
        plasmids = set(pl.read_csv(row["plasmids"], separator="\t")["seq_name"].to_list())
    pieces = []
    with open(fa, "w") as fh:
        for c, s in seqs.items():
            cuts = cut_points(len(s), lo, hi, rng)
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                name = f"{iso}__{c}__p{i}"
                fh.write(f">{name}\n{s[a:b]}\n")
                pieces.append({"contig": name, "parent": c, "start": a, "end": b, "length": b - a,
                               "isolate": iso, "plasmid": c in plasmids})
    pdf = pl.DataFrame(pieces)
    pdf.write_csv(pt, separator="\t")
    p = read_pileup(row["pileup"])
    p = p.with_columns(pl.col("c2").cast(pl.Int64)).sort("c1", "c2")
    ends = pdf.select(pl.col("parent").alias("c1"), pl.col("start").alias("c2"), pl.col("contig").alias("piece"),
                      pl.col("start").alias("p0")).sort("c1", "c2")
    p = p.join_asof(ends, on="c2", by="c1", strategy="backward")
    p = (p.filter(pl.col("piece").is_not_null())
         .with_columns((pl.col("c2") - pl.col("p0")).alias("c2"), (pl.col("c3").cast(pl.Int64) - pl.col("p0")).alias("c3"),
                       (pl.col("c7").cast(pl.Int64) - pl.col("p0")).alias("c7"), (pl.col("c8").cast(pl.Int64) - pl.col("p0")).alias("c8"))
         .drop("c1", "p0").rename({"piece": "c1"}).select(COLS))
    p.write_parquet(pq)
    return pq, pdf


def cut_isolate(row: dict, cache: Path, lo: int, hi: int) -> tuple[Path, pl.DataFrame]:
    """fragment_isolate with the isolate's own generator (isolate_seed): the cuts depend on the name only."""
    return fragment_isolate(row, cache, lo, hi, np.random.default_rng(isolate_seed(row["isolate"])))


def thin(p: pl.DataFrame, f: float, rng) -> pl.DataFrame:
    """Keep each read's call with probability f, independently per count column."""
    if f >= 1:
        return p
    cols = {c: rng.binomial(p[c].to_numpy().astype(np.int64), f) for c in COUNTS}
    p = p.with_columns(*[pl.Series(c, v) for c, v in cols.items()])
    nvalid = pl.col("c12") + pl.col("c13") + pl.col("c14")
    p = p.with_columns(nvalid.alias("c10"), nvalid.alias("c5")).filter(pl.col("c10") > 0)
    return p.with_columns((100 * pl.col("c12") / pl.col("c10")).round(2).alias("c11"))


def median_coverage(pq: Path) -> float:
    return float(pl.scan_parquet(pq).select(pl.col("c10").cast(pl.Int64).median()).collect().item())


def motif_table(rows: list[dict]) -> pl.DataFrame:
    """Nanomotif bin-motifs.tsv -> bin, motif, mod_position, mod_type (perturb_bins' --genome-motifs);
    a non-palindromic motif's complement is added as its own row."""
    out = []
    for r in rows:
        m = pl.read_csv(r["motifs"], separator="\t", infer_schema_length=0)
        parts = [m.select("motif", "mod_position", "mod_type")]
        if {"motif_complement", "mod_position_complement"} <= set(m.columns):
            parts.append(m.filter(pl.col("motif_complement").is_not_null() & (pl.col("motif_complement") != ""))
                         .select(pl.col("motif_complement").alias("motif"),
                                 pl.col("mod_position_complement").alias("mod_position"), "mod_type"))
        out.append(pl.concat(parts).unique().with_columns(pl.lit(r["isolate"]).alias("bin"))
                   .select("bin", "motif", "mod_position", "mod_type"))
    return pl.concat(out) if out else pl.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isolates", required=True, help="isolate, assembly, pileup, motifs, plasmids (TSV)")
    ap.add_argument("--data-root", default=os.environ.get("EPIREFINE_DATA"),
                    help="benchmark data root (default: the EPIREFINE_DATA environment variable)")
    ap.add_argument("--out", default=None, help="output directory (default: <data root>/" + SUITE_ROOT + ")")
    ap.add_argument("--suite", default=str(Path(__file__).parent / "suite_monocultures.json"))
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    ap.add_argument("--communities", type=int, nargs=3, default=(6, 3, 3), help="per split: train test val")
    ap.add_argument("--isolates-per-community", type=int, default=8)
    ap.add_argument("--cov-min", type=float, default=3.0)
    ap.add_argument("--cov-max", type=float, default=60.0)
    ap.add_argument("--min-len", type=int, default=2_000)
    ap.add_argument("--max-len", type=int, default=100_000)
    ap.add_argument("--orphan-bins", type=int, default=1)
    ap.add_argument("--missing", type=float, default=0.15)
    ap.add_argument("--contaminants", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if not (a.out or a.data_root):
        sys.exit("output location not set: pass --data-root <dir> (or set EPIREFINE_DATA), or --out <dir>")
    out = Path(a.out) if a.out else Path(a.data_root) / SUITE_ROOT
    # the suite lists paths relative to <data root>/<root>; with an --out outside the data root they stay absolute
    in_root = bool(a.data_root) and out.resolve() == (Path(a.data_root) / SUITE_ROOT).resolve()
    rel = (lambda p: str(p.relative_to(out))) if in_root else str
    cache = out / "isolates"
    cache.mkdir(parents=True, exist_ok=True)

    iso = discover(a.isolates)
    split_of = split_genera(iso, a.fractions, rng)
    iso = iso.with_columns(pl.col("genus").replace_strict(split_of).alias("split"))
    print(iso.group_by("split").agg(pl.len(), pl.col("isolate")).sort("split"))

    covs, frag = {}, {}
    for row in iso.iter_rows(named=True):
        pq, pieces = cut_isolate(row, cache, a.min_len, a.max_len)
        frag[row["isolate"]] = (pq, pieces)
        covs[row["isolate"]] = median_coverage(pq)
        print(f"  {row['isolate']}: {pieces.height} pieces, median coverage {covs[row['isolate']]:.0f}x")
    iso = iso.with_columns(pl.col("isolate").replace_strict(covs, return_dtype=pl.Float64).alias("coverage"))
    iso.write_csv(out / "isolates.tsv", separator="\t")

    scenarios = []
    for split, n_comm, label in zip(("train", "test", "val"), a.communities, ("mono-train", "mono-test", "mono-val")):
        members = iso.filter(pl.col("split") == split)
        for k in range(n_comm):
            name = f"mono_{split}_{k + 1}"
            d = out / split / name
            d.mkdir(parents=True, exist_ok=True)
            pick = members.sample(min(a.isolates_per_community, members.height), seed=int(rng.integers(1 << 31)))
            target = np.exp(rng.uniform(np.log(a.cov_min), np.log(a.cov_max), pick.height))
            rows, pieces_all = [], []
            with open(d / "assembly.fasta", "w") as fa, open(d / "pileup.bed", "w") as pf:
                for r, t in zip(pick.iter_rows(named=True), target):
                    pq, pieces = frag[r["isolate"]]
                    f = min(1.0, t / max(covs[r["isolate"]], 1e-9))
                    fa.write((cache / f"{r['isolate']}.fasta").read_text())
                    thin(pl.read_parquet(pq), f, rng).sort("c1", "c2").write_csv(pf, separator="\t", include_header=False)
                    pieces_all.append(pieces.with_columns(pl.lit(f).alias("coverage_factor"),
                                                          pl.lit(covs[r["isolate"]] * f).alias("target_coverage")))
                    rows.append(r)
            pieces_all = pl.concat(pieces_all)
            pieces_all.write_csv(d / "pieces.tsv", separator="\t")
            pieces_all.select("contig", "length", pl.col("isolate").alias("expected")).write_csv(d / "truth.tsv", separator="\t")
            motif_table(rows).write_csv(d / "genome_motifs.tsv", separator="\t")
            subprocess.run([sys.executable, str(Path(__file__).parent / "perturb_bins.py"), "--truth", str(d / "truth.tsv"),
                            "--out", str(d), "--orphan-bins", str(a.orphan_bins), "--missing", str(a.missing),
                            "--contaminants", str(a.contaminants), "--genome-motifs", str(d / "genome_motifs.tsv"),
                            "--seed", str(int(rng.integers(1 << 31)))], check=True)
            scenarios.append({"name": name, "group": label, "assembly": rel(d / "assembly.fasta"),
                              "pileup": rel(d / "pileup.bed"), "contig_bin": rel(d / "contig_bin.tsv"),
                              "motifs": rel(d / "bin-motifs.tsv"), "truth": rel(d / "truth_actions.tsv"),
                              "cache": rel(d / "pbcache"), "optional": label != "mono-train"})
            print(f"{name}: {pick.height} isolates, coverage targets " + ", ".join(f"{x:.0f}x" for x in target))
    Path(a.suite).write_text(json.dumps({
        "description": "Monoculture benchmark sets (benchmarks/monoculture_sets.py). Tune on mono-train, select on "
                       "mono-test, report mono-val once. Isolates split by genus; pileups thinned to 3-60x. "
                       "Paths are relative to <data root>/<root>; the data root is benchmark_suite.py's --data-root "
                       "or the EPIREFINE_DATA environment variable.",
        **({"root": SUITE_ROOT} if in_root else {}), "scenarios": scenarios}, indent=1))
    print(f"wrote {len(scenarios)} scenarios to {a.suite}")


if __name__ == "__main__":
    main()
