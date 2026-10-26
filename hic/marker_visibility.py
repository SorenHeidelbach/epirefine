"""Can CheckM2 see the contigs Hi-C says are misbinned?

CheckM2 predicts completeness and contamination from the KEGG orthologs (KOs) its DIAMOND search finds
on a bin's proteins. Contamination shows mostly as extra copies of genes that a genome normally has
once; a foreign contig with no such gene can't raise it, and one whose single-copy genes are new to
the bin raises completeness instead.

Single-copy markers are derived from the sample itself: KOs present in at least `--min-prevalence`
of the high-quality bins (CheckM2 completeness >= 90, contamination < 5) and found in exactly one
copy in at least `--min-single` of the bins that have them. For each binned contig with a confident
Hi-C call (original binning and its CheckM2 run):

  markers       single-copy marker genes on the contig
  markers_dup   of these, markers also found elsewhere in the same bin (would read as contamination)
  visibility    "no marker" (CheckM2 can't see it), "only new markers" (reads as completeness),
                "duplicated marker" (reads as contamination)
  class         member (Hi-C agrees with the bin) or foreign (Hi-C bin is another bin)

    python hic/marker_visibility.py --diamond checkm2/original/diamond_output \
        --checkm2 checkm2/original/quality_report.tsv --hic-calls hic/hic_calls.tsv \
        --contig-bin contig_bin.tsv --lengths contig_lengths.tsv --out hic_quality/marker_visibility.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diamond", required=True, help="CheckM2 diamond_output directory")
    ap.add_argument("--checkm2", required=True, help="CheckM2 quality_report.tsv of the same bins")
    ap.add_argument("--hic-calls", required=True)
    ap.add_argument("--contig-bin", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--min-prevalence", type=float, default=0.8)
    ap.add_argument("--min-single", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(a.diamond).glob("DIAMOND_RESULTS*.tsv"))
    # query: <bin>Ω<contig>_<gene>; target: UniRef100_<id>~<KO>; one KO per gene (best hit)
    genes = (pl.concat([pl.read_csv(f, separator="\t", has_header=False, columns=[0, 1])
                        .select(pl.nth(0).alias("q"), pl.nth(1).alias("t")) for f in files])
             .unique("q", keep="first", maintain_order=True)
             .with_columns(pl.col("q").str.split_exact("Ω", 1).struct.rename_fields(["bin", "gene"]).alias("s"),
                           pl.col("t").str.split("~").list.last().alias("ko"))
             .unnest("s").with_columns(pl.col("gene").str.replace(r"_\d+$", "").alias("contig"))
             .select("bin", "contig", "gene", "ko"))
    q = pl.read_csv(a.checkm2, separator="\t")
    hq = q.filter((pl.col("Completeness") >= 90) & (pl.col("Contamination") < 5))["Name"].to_list()
    copies = genes.filter(pl.col("bin").is_in(hq)).group_by("bin", "ko").agg(pl.len().alias("copies"))
    markers = (copies.group_by("ko").agg(pl.len().alias("bins"), (pl.col("copies") == 1).mean().alias("single"))
               .filter((pl.col("bins") >= a.min_prevalence * len(hq)) & (pl.col("single") >= a.min_single)))["ko"]
    print(f"{len(hq)} high-quality bins; {len(markers)} single-copy marker KOs")
    mg = genes.filter(pl.col("ko").is_in(markers.implode()))
    copies_in_bin = mg.group_by("bin", "ko").agg(pl.col("contig").n_unique().alias("k"))
    per_contig = (mg.join(copies_in_bin, on=["bin", "ko"])
                  .group_by("contig").agg(pl.len().alias("markers"), (pl.col("k") > 1).sum().alias("markers_dup")))
    cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "bin"], infer_schema_length=0)
    calls = (pl.read_csv(a.hic_calls, separator="\t", infer_schema_length=0)
             .filter(pl.col("confident") == "true").select("contig", "hic_bin"))
    lengths = pl.read_csv(a.lengths, separator="\t")
    x = (cb.join(calls, on="contig").join(lengths, on="contig", how="left").join(per_contig, on="contig", how="left")
         .with_columns(pl.col("markers", "markers_dup").fill_null(0),
                       pl.when(pl.col("hic_bin") == pl.col("bin")).then(pl.lit("member")).otherwise(pl.lit("foreign")).alias("class"))
         .with_columns(pl.when(pl.col("markers") == 0).then(pl.lit("no marker"))
                       .when(pl.col("markers_dup") == 0).then(pl.lit("only new markers"))
                       .otherwise(pl.lit("duplicated marker")).alias("visibility")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    x.write_csv(a.out, separator="\t")
    s = (x.group_by("class", "visibility").agg(pl.len().alias("contigs"), pl.col("length").sum().alias("bp"))
         .with_columns((pl.col("contigs") / pl.col("contigs").sum().over("class")).alias("share_contigs"),
                       (pl.col("bp") / pl.col("bp").sum().over("class")).alias("share_bp"))
         .sort("class", "visibility"))
    s.write_csv(Path(a.out).with_name("marker_visibility_summary.tsv"), separator="\t", float_precision=4)
    with pl.Config(tbl_width_chars=200):
        print(s)


if __name__ == "__main__":
    main()
