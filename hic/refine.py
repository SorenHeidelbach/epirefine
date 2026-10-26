"""Hi-C-only bin refinement, as a reference point: what the bins look like when Hi-C decides.

From confident Hi-C calls (hic/contacts.py): a binned contig whose Hi-C bin is another bin is moved
there; an unbinned contig with a confident Hi-C bin is added to it. Everything else keeps its bin.
Writes new_contig_bin.tsv (contig<TAB>bin, no header, like pbinner and Nanomotif) so the workflow can
score it with CheckM2 next to the methylation-based methods.

    python hic/refine.py --hic-calls hic/hic_calls.tsv --contig-bin contig_bin.tsv --out hic/new_contig_bin.tsv
"""
from __future__ import annotations

import argparse

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hic-calls", required=True)
    ap.add_argument("--contig-bin", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    calls = (pl.read_csv(a.hic_calls, separator="\t", infer_schema_length=0)
             .filter(pl.col("confident") == "true").select("contig", "hic_bin"))
    cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "bin"],
                     infer_schema_length=0)
    new = (cb.join(calls, on="contig", how="full", coalesce=True)
           .with_columns(pl.coalesce("hic_bin", "bin").alias("new_bin")))
    moved = new.filter(pl.col("bin").is_not_null() & pl.col("hic_bin").is_not_null() & (pl.col("bin") != pl.col("hic_bin")))
    added = new.filter(pl.col("bin").is_null())
    new.select("contig", "new_bin").sort("new_bin", "contig").write_csv(a.out, separator="\t", include_header=False)
    print(f"Hi-C refinement: {moved.height} binned contigs moved to their Hi-C bin, {added.height} unbinned contigs added")


if __name__ == "__main__":
    main()
