"""Bin quality from Hi-C, per bin and bin set: an alternative to CheckM2's marker-gene estimates.

CheckM2 sees a bin through its marker genes: a foreign contig is contamination only when it brings a
second copy of a marker, and a missing contig lowers completeness only when it carries markers the
bin lacks. Hi-C sees every contig with contacts. Per bin and bin set:

  cis_share     contacts between the bin's own contigs / all inter-contig contacts of its contigs
                (contacts to unbinned contigs count against it: lower = less complete or less pure)
  foreign_bp    sequence in the bin whose confident Hi-C bin is another bin
  missing_bp    sequence outside the bin (unbinned or in another bin) whose confident Hi-C bin is this one

The Hi-C calls name the bins of the original binning, so this works for any refinement that keeps
bin names. A bin set built from the Hi-C calls (hic/refine.py) scores well on foreign_bp and
missing_bp by construction; cis_share and CheckM2 are the fair comparisons for it.

    python hic/bin_quality.py --pairs hic/contig_pairs.parquet --hic-calls hic/hic_calls.tsv \
        --lengths contig_lengths.tsv --set original contig_bin.tsv --set pbinner pbinner/new_contig_bin.tsv \
        [--checkm2 original checkm2/original/quality_report.tsv ...] --out hic_quality/
Writes bins.tsv (one row per bin and set) and summary.tsv (one row per set).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def read_bins(path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", has_header=False, infer_schema_length=0)
    df = df.select(df.columns[:2])
    df.columns = ["contig", "bin"]
    return df.filter(~pl.col("contig").is_in(["contig", "contig_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--hic-calls", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--set", nargs=2, action="append", metavar=("NAME", "CONTIG_BIN"), required=True)
    ap.add_argument("--checkm2", nargs=2, action="append", metavar=("NAME", "REPORT"), default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(a.pairs)
    lengths = pl.read_csv(a.lengths, separator="\t")
    calls = (pl.read_csv(a.hic_calls, separator="\t", infer_schema_length=0)
             .filter(pl.col("confident") == "true").select("contig", "hic_bin"))
    ends = pl.concat([pairs.select(pl.col("a").alias("contig"), "n"), pairs.select(pl.col("b").alias("contig"), "n")])
    ends = ends.group_by("contig").agg(pl.col("n").sum().alias("contacts"))
    checkm2 = {n: pl.read_csv(p, separator="\t").select(pl.col("Name").alias("bin"), "Completeness", "Contamination")
               for n, p in a.checkm2}
    rows = []
    for name, path in a.set:
        cb = read_bins(path)
        intra = (pairs.join(cb.rename({"contig": "a", "bin": "ba"}), on="a")
                 .join(cb.rename({"contig": "b", "bin": "bb"}), on="b")
                 .filter(pl.col("ba") == pl.col("bb")).group_by("ba").agg(pl.col("n").sum().alias("intra"))
                 .rename({"ba": "bin"}))
        tot = cb.join(ends, on="contig").group_by("bin").agg(pl.col("contacts").sum())
        withlen = cb.join(lengths, on="contig", how="left")
        size = withlen.group_by("bin").agg(pl.col("length").sum().alias("bin_bp"), pl.len().alias("contigs"))
        foreign = (withlen.join(calls, on="contig").filter(pl.col("hic_bin") != pl.col("bin"))
                   .group_by("bin").agg(pl.col("length").sum().alias("foreign_bp")))
        missing = (calls.join(cb, on="contig", how="left").filter(pl.col("bin").is_null() | (pl.col("bin") != pl.col("hic_bin")))
                   .join(lengths, on="contig").group_by("hic_bin").agg(pl.col("length").sum().alias("missing_bp"))
                   .rename({"hic_bin": "bin"}))
        b = (size.join(tot, on="bin", how="left").join(intra, on="bin", how="left")
             .join(foreign, on="bin", how="left").join(missing, on="bin", how="left")
             .with_columns(pl.col("contacts", "intra", "foreign_bp", "missing_bp").fill_null(0))
             .with_columns((2 * pl.col("intra") / pl.col("contacts")).alias("cis_share"), pl.lit(name).alias("set")))
        if name in checkm2:
            b = b.join(checkm2[name], on="bin", how="left")
        rows.append(b)
    bins = pl.concat(rows, how="diagonal_relaxed").sort("set", "bin")
    bins.write_csv(out / "bins.tsv", separator="\t", float_precision=4)
    agg = [pl.len().alias("bins"), pl.col("bin_bp").sum().alias("binned_bp"),
           pl.col("cis_share").filter(pl.col("contacts") >= 100).median().alias("median_cis_share"),
           (2 * pl.col("intra").sum() / pl.col("contacts").sum()).alias("pooled_cis_share"),
           pl.col("foreign_bp").sum(), (pl.col("foreign_bp") > 0).sum().alias("bins_with_foreign"),
           pl.col("missing_bp").sum(), (pl.col("missing_bp") > 0).sum().alias("bins_with_missing")]
    if "Completeness" in bins.columns:
        agg += [pl.col("Completeness").median().alias("median_completeness"),
                pl.col("Contamination").median().alias("median_contamination")]
    summary = bins.group_by("set", maintain_order=True).agg(agg)
    summary.write_csv(out / "summary.tsv", separator="\t", float_precision=4)
    with pl.Config(tbl_width_chars=250, tbl_cols=20):
        print(summary)


if __name__ == "__main__":
    main()
