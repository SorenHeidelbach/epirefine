"""Introduce known binning errors into a perfect binning, for contamination / inclusion benchmarks.

Starting from truth (contig -> expected bin), the perfect binning puts every contig with a known
bin into it. Then:
  * orphan bins:  whole bins are dropped; their contigs become unbinned and must stay unbinned
                  (their genome has no bin to join);
  * missing:      a fraction of the remaining bins' contigs is unbinned; each should be included
                  back into its bin;
  * contaminants: a fraction of contigs is moved into another (wrong) bin, including some contigs
                  of orphan genomes; each should be removed, and re-included into its true bin when
                  that bin exists.
Bins keep at least --min-keep correct contigs, so none is emptied.

    python benchmarks/perturb_bins.py --truth frag/truth.tsv --out scen1 --seed 1

Writes contig_bin.tsv (no header, as binners write it) and truth_actions.tsv with columns
contig, length, true_bin ("" if its genome has no bin), given_bin ("" if unbinned), category.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="contig, length, expected")
    ap.add_argument("--out", required=True)
    ap.add_argument("--orphan-bins", type=int, default=2)
    ap.add_argument("--missing", type=float, default=0.15, help="fraction of member contigs unbinned")
    ap.add_argument("--contaminants", type=float, default=0.06, help="fraction of binned contigs moved")
    ap.add_argument("--min-keep", type=int, default=3)
    ap.add_argument("--genome-motifs", help="bin, motif, mod_position, mod_type per true bin (simulations): "
                                             "writes bin-motifs.tsv for the bins kept in the scenario")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t = pl.read_csv(a.truth, separator="\t").select("contig", "length", "expected")
    known = t.filter(pl.col("expected") != "unknown")
    bins = sorted(known["expected"].unique().to_list())
    orphans = set(rng.choice(bins, a.orphan_bins, replace=False).tolist()) if a.orphan_bins else set()
    kept = [b for b in bins if b not in orphans]

    contig = t["contig"].to_list()
    true_bin = {c: (e if e in kept else "") for c, e in zip(contig, t["expected"])}
    given = {c: (e if e in kept else "") for c, e in zip(contig, t["expected"])}
    category = {c: ("member" if e in kept else "orphan" if e in orphans else "unknown")
                for c, e in zip(contig, t["expected"])}

    members = {b: [c for c in contig if given[c] == b] for b in kept}
    for b in kept:  # unbin a fraction of each bin, keeping min_keep
        pool = members[b]
        k = min(int(round(a.missing * len(pool))), max(0, len(pool) - a.min_keep))
        for c in rng.choice(pool, k, replace=False) if k else []:
            given[c] = ""
            category[c] = "missing"
    # contaminants: from the remaining members and from orphan genomes, into a random other kept bin
    still = [c for c in contig if category[c] == "member"]
    orphan_pool = [c for c in contig if category[c] == "orphan"]
    n_cont = int(round(a.contaminants * len(still)))
    n_orph = min(len(orphan_pool), max(1, n_cont // 4)) if orphan_pool else 0
    left = {b: sum(1 for c in still if given[c] == b) for b in kept}
    moved = 0
    for c in rng.permutation(still):
        if moved >= n_cont - n_orph:
            break
        if left[given[c]] <= a.min_keep:
            continue
        left[given[c]] -= 1
        given[c] = str(rng.choice([b for b in kept if b != true_bin[c]]))
        category[c] = "contaminant"
        moved += 1
    for c in rng.choice(orphan_pool, n_orph, replace=False) if n_orph else []:
        given[c] = str(rng.choice(kept))
        category[c] = "orphan_contaminant"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    act = t.select("contig", "length").with_columns(
        pl.col("contig").replace_strict(true_bin).alias("true_bin"),
        pl.col("contig").replace_strict(given).alias("given_bin"),
        pl.col("contig").replace_strict(category).alias("category"))
    act.write_csv(out / "truth_actions.tsv", separator="\t")
    act.filter(pl.col("given_bin") != "").select("contig", "given_bin").write_csv(
        out / "contig_bin.tsv", separator="\t", include_header=False)
    if a.genome_motifs:
        gm = pl.read_csv(a.genome_motifs, separator="\t")
        gm.filter(pl.col("bin").is_in(kept)).select("bin", "motif", "mod_position", "mod_type").write_csv(
            out / "bin-motifs.tsv", separator="\t")
    print(f"orphan bins {sorted(orphans)}; " + ", ".join(
        f"{k} {v}" for k, v in act.group_by("category").len().sort("category").iter_rows()))


if __name__ == "__main__":
    main()
