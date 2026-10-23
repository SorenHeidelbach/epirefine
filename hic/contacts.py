"""Contig-level Hi-C contacts and a per-contig Hi-C bin call, for validating bin refinement.

Input: a .pairs file (chromap --preset hic) and a contig -> bin table. Contacts between different
contigs are counted per contig pair; for each contig and bin:

    links[c, h]     = contacts between c and the members of bin h (c itself excluded)
    expected[c, h]  = total inter-contig contacts of c x share of all contact ends that fall on bin h
                      (random ligation between cells: a contig contacts each bin in proportion to
                      that bin's overall Hi-C signal, i.e. abundance x restriction sites)
    enrichment      = links / expected

Hi-C call for a contig: the bin with the highest enrichment, when it has at least `min_links`
contacts and takes at least `min_share` of the contig's contacts to bins. A binned contig whose
Hi-C bin is another bin is contamination evidence; one whose Hi-C bin is its own is a member.

    python hic_contacts.py --pairs out.pairs --contig-bin contig_bin.tsv --out hic/
Writes contig_pairs.parquet (c1, c2, n), contig_bin_links.tsv and hic_calls.tsv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def read_pairs(path) -> pl.LazyFrame:
    # 4DN .pairs: readID chr1 pos1 chr2 pos2 strand1 strand2 [...]; header lines start with '#'
    return (pl.scan_csv(path, separator="\t", has_header=False, comment_prefix="#",
                        new_columns=["read", "c1", "p1", "c2", "p2", "s1", "s2"],
                        schema_overrides={"c1": pl.Utf8, "c2": pl.Utf8}, truncate_ragged_lines=True)
            .select("c1", "c2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--contig-bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-links", type=int, default=10)
    ap.add_argument("--min-share", type=float, default=0.6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = (read_pairs(a.pairs).filter(pl.col("c1") != pl.col("c2"))
             .select(pl.min_horizontal("c1", "c2").alias("a"), pl.max_horizontal("c1", "c2").alias("b"))
             .group_by("a", "b").agg(pl.len().alias("n")).collect(engine="streaming"))
    pairs.write_parquet(out / "contig_pairs.parquet")
    print(f"{pairs['n'].sum()} inter-contig contacts between {pairs.height} contig pairs")

    cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "bin"])
    cb = cb.filter(~pl.col("contig").is_in(["contig", "contig_id"]))
    both = pl.concat([pairs.select(pl.col("a").alias("c"), pl.col("b").alias("o"), "n"),
                      pairs.select(pl.col("b").alias("c"), pl.col("a").alias("o"), "n")])
    total = both.group_by("c").agg(pl.col("n").sum().alias("total"))
    links = (both.join(cb.rename({"contig": "o"}), on="o").group_by("c", "bin").agg(pl.col("n").sum().alias("links")))
    # share of all contact ends on each bin
    bin_ends = links.group_by("bin").agg(pl.col("links").sum().alias("ends"))
    all_ends = float(both["n"].sum())
    links = (links.join(total, on="c").join(bin_ends, on="bin")
             .with_columns((pl.col("total") * pl.col("ends") / all_ends).alias("expected"))
             .with_columns((pl.col("links") / pl.col("expected")).alias("enrichment"),
                           (pl.col("links") / pl.col("links").sum().over("c")).alias("share")))
    links.write_csv(out / "contig_bin_links.tsv", separator="\t", float_precision=4)

    best = links.sort("enrichment", descending=True).group_by("c", maintain_order=True).first()
    calls = (best.with_columns(((pl.col("links") >= a.min_links) & (pl.col("share") >= a.min_share)).alias("confident"))
             .join(cb.rename({"contig": "c", "bin": "given_bin"}), on="c", how="left")
             .rename({"c": "contig", "bin": "hic_bin"}))
    calls.write_csv(out / "hic_calls.tsv", separator="\t", float_precision=4)
    conf = calls.filter("confident")
    binned = conf.filter(pl.col("given_bin").is_not_null())
    print(f"{calls.height} contigs with contacts to bins; {conf.height} confident Hi-C calls; of these "
          f"{binned.height} binned: {int((binned['hic_bin'] == binned['given_bin']).sum())} agree with their bin")


if __name__ == "__main__":
    main()
