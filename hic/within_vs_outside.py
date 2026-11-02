"""Hi-C contacts inside vs outside the bin, per contig and per bin, next to epirefine's decisions.

Per binned contig (original binning): contacts with other contigs of its own bin (within), with
contigs of other bins (outside) and with unbinned contigs, together with epirefine's status and
P(belongs to its bin) = 1 - p_contaminant. A genuine member has most of its binned contacts within
its bin; a misbinned contig has them outside.

Per bin and bin set (e.g. original, epirefine, hic): within-bin share = contacts between the bin's own
contigs / contacts of its contigs with any binned contig (each counted from both ends). Refinement
that removes foreign contigs and adds members raises it.

    python hic/within_vs_outside.py --pairs hic/contig_pairs.parquet --contig-bin contig_bin.tsv \
        --decisions epirefine/contig_decisions.tsv --set original contig_bin.tsv --set epirefine epirefine/new_contig_bin.tsv \
        --out hic_within/
Writes contigs.tsv and bins.tsv.
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
    ap.add_argument("--pairs", required=True, help="contig_pairs.parquet (hic/contacts.py)")
    ap.add_argument("--contig-bin", required=True, help="original contig -> bin table")
    ap.add_argument("--decisions", required=True, help="epirefine contig_decisions.tsv")
    ap.add_argument("--set", nargs=2, action="append", metavar=("NAME", "CONTIG_BIN"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(a.pairs)
    both = pl.concat([pairs.select(pl.col("a").alias("c"), pl.col("b").alias("o"), "n"),
                      pairs.select(pl.col("b").alias("c"), pl.col("a").alias("o"), "n")])

    # per contig, original binning
    cb = read_bins(a.contig_bin)
    x = (both.join(cb.rename({"contig": "c", "bin": "cbin"}), on="c")
         .join(cb.rename({"contig": "o", "bin": "obin"}), on="o", how="left")
         .with_columns(pl.when(pl.col("obin").is_null()).then(pl.lit("unbinned"))
                       .when(pl.col("obin") == pl.col("cbin")).then(pl.lit("within"))
                       .otherwise(pl.lit("outside")).alias("kind")))
    per = (x.group_by("c", "cbin").agg(*[pl.col("n").filter(pl.col("kind") == k).sum().alias(k)
                                         for k in ("within", "outside", "unbinned")],
                                       pl.col("o").filter(pl.col("kind") == "within").n_unique().alias("within_partners"))
           .with_columns((pl.col("within") / (pl.col("within") + pl.col("outside"))).alias("within_share"))
           .rename({"c": "contig", "cbin": "bin"}))
    d = (pl.read_csv(a.decisions, separator="\t", infer_schema_length=0)
         .select("contig", "length", "status", (1 - pl.col("p_contaminant").cast(pl.Float64)).alias("p_own")))
    per = per.join(d, on="contig", how="left")
    per.write_csv(out / "contigs.tsv", separator="\t", float_precision=4)

    # per bin and set
    rows = []
    for name, path in a.set:
        s = read_bins(path)
        y = (both.join(s.rename({"contig": "c", "bin": "cbin"}), on="c")
             .join(s.rename({"contig": "o", "bin": "obin"}), on="o")
             .group_by("cbin").agg(pl.col("n").filter(pl.col("obin") == pl.col("cbin")).sum().alias("within"),
                                   pl.col("n").filter(pl.col("obin") != pl.col("cbin")).sum().alias("outside"))
             .with_columns((pl.col("within") / (pl.col("within") + pl.col("outside"))).alias("within_share"),
                           pl.lit(name).alias("set"))
             .rename({"cbin": "bin"}))
        rows.append(y)
    bins = pl.concat(rows).sort("set", "bin")
    bins.write_csv(out / "bins.tsv", separator="\t", float_precision=4)
    with pl.Config(tbl_width_chars=200):
        print(per.filter(pl.col("within") + pl.col("outside") >= 10).group_by("status").agg(
            pl.len().alias("contigs"), pl.col("within_share").median().alias("median_within_share"),
            (pl.col("within_share") < 0.5).mean().alias("share_mostly_outside")).sort("contigs", descending=True))
        print(bins.group_by("set", maintain_order=True).agg(
            pl.len().alias("bins"), (2 * pl.col("within").sum() / (2 * pl.col("within").sum() + pl.col("outside").sum())).alias("pooled"),
            pl.col("within_share").median().alias("median_within_share")))


if __name__ == "__main__":
    main()
