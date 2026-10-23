"""Cut an assembly into realistic pieces and remap its modkit pileup, keeping the reference truth.

Benchmarks for contamination detection and contig inclusion need many contigs of known origin and
varied length. This cuts every contig longer than --max-len into consecutive pieces whose lengths are
log-uniform in [--min-len, --max-len], writes the pieces as a new assembly, rewrites the pileup in
piece coordinates (same 18 columns), and gives every piece the `expected` bin of its parent contig.

    python benchmarks/fragment_assembly.py --assembly asm.fasta --pileup pileup.bed \
        --truth truth_all_contigs.tsv --out zymo_frag

Writes: assembly.fasta, pileup.bed, truth.tsv (contig, parent, start, length, expected, species).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pbinner import io as pio


def cut_points(length: int, lo: int, hi: int, rng) -> list[int]:
    if length <= hi:
        return [0, length]
    cuts = [0]
    while length - cuts[-1] > hi:
        step = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        cuts.append(cuts[-1] + step)
    if length - cuts[-1] < lo:  # fold a short tail into the previous piece
        cuts.pop()
    cuts.append(length)
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", required=True)
    ap.add_argument("--pileup", required=True)
    ap.add_argument("--truth", required=True, help="columns contig, expected, species")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=5_000)
    ap.add_argument("--max-len", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    seqs = pio.read_fasta(a.assembly)
    truth = pl.read_csv(a.truth, separator="\t", infer_schema_length=0)
    exp = dict(zip(truth["contig"], truth["expected"]))
    sp = dict(zip(truth["contig"], truth["species"].fill_null("")))
    rows = []
    with open(out / "assembly.fasta", "w") as fa:
        for c, s in seqs.items():
            cuts = cut_points(len(s), a.min_len, a.max_len, rng)
            for k, (b, e) in enumerate(zip(cuts[:-1], cuts[1:])):
                name = c if len(cuts) == 2 else f"{c}__p{k}"
                fa.write(f">{name}\n")
                for i in range(b, e, 80):
                    fa.write(s[i:i + 80] + "\n")
                rows.append({"contig": name, "parent": c, "start": b, "length": e - b,
                             "expected": exp.get(c, "unknown"), "species": sp.get(c, "")})
    pieces = pl.DataFrame(rows)
    pieces.write_csv(out / "truth.tsv", separator="\t")

    # remap the pileup: as-of join of each row onto the piece that starts at or before it
    pile = pl.scan_csv(a.pileup, separator="\t", has_header=False, infer_schema=False)
    cols = pile.collect_schema().names()  # positional: chrom, start, end, ..., thickStart, thickEnd, ...
    chrom, start, end, tstart, tend = cols[0], cols[1], cols[2], cols[6], cols[7]
    starts = pieces.select(pl.col("parent").alias(chrom), pl.col("start").alias("pstart"),
                           pl.col("contig").alias("piece")).sort(chrom, "pstart")
    pile = pile.with_columns(pl.col(start).cast(pl.Int64)).sort(chrom, start)
    shifted = (pile.join_asof(starts.lazy(), left_on=start, right_on="pstart", by=chrom)
               .with_columns(*[(pl.col(c).cast(pl.Int64) - pl.col("pstart")).cast(pl.Utf8).alias(c)
                               for c in (start, end, tstart, tend)],
                             pl.col("piece").alias(chrom))
               .select(cols))
    shifted.sink_csv(out / "pileup.bed", separator="\t", include_header=False)
    n = pieces.height
    print(f"{len(seqs)} contigs -> {n} pieces; median length {int(pieces['length'].median())} bp -> {out}")


if __name__ == "__main__":
    main()
