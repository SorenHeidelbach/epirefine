"""Contig-level Hi-C contacts and a per-contig Hi-C bin call, for validating bin refinement.

Input: a .pairs file (chromap --preset hic) and a contig -> bin table. Contacts between different
contigs are counted per contig pair; for each contig and bin:

    links[c, h]     = contacts between c and the members of bin h (c itself excluded)
    expected[c, h]  = total inter-contig contacts of c x share of all contact ends that fall on bin h
                      (random ligation between cells: a contig contacts each bin in proportion to
                      that bin's overall Hi-C signal, i.e. abundance x restriction sites)
    enrichment      = links / expected

Hi-C call for a contig: the bin with the most contacts (ties: higher enrichment). A call is
confident only when the mapping supports it from several sides:

    links >= min_links              contacts to the called bin
    share >= min_share              of the contig's contacts to binned contigs
    partners >= min_partners        distinct contigs of the called bin contacted
    links - top_partner >= min_links_wo_top
                                    still supported without its strongest single partner
    P(Binomial(n, b_h) >= links) <= max_p_random
                                    not explained by random ligation (b_h: the bin's share of all
                                    contact ends)

Measured on ZymoFecal (contigs of CheckM2 high-quality bins, which should be called to their own
bin): the plain rule (>= 10 links, >= 0.6 share) disagreed with the bin for 33 of 343 calls (9.6%);
almost all of these had every contact on ONE partner contig (a repeat, an overlapping contig end or
a mapping artefact). With the defaults below: 0 of 148. Bins with fewer than min_partners other
contigs can't receive calls. (Ranking by enrichment instead favours tiny bins, whose expected count
is near zero.) A binned contig whose
Hi-C bin is another bin is contamination evidence; one whose Hi-C bin is its own is a member.

    python hic/contacts.py --pairs out.pairs --out hic/                       # count: contig_pairs.parquet
    python hic/contacts.py --contig-pairs hic/contig_pairs.parquet --contig-bin contig_bin.tsv --out hic/
Counting writes contig_pairs.parquet (a, b, n); calling writes contig_bin_links.tsv and hic_calls.tsv.
Both steps run together when --pairs and --contig-bin are given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import binom


def read_pairs(path) -> pl.LazyFrame:
    """Contig pairs of uniquely mapped read pairs.

    4DN .pairs: readID chr1 pos1 chr2 pos2 strand1 strand2 [pair_type ...]; header lines start with
    '#'. chromap writes pair_type as an 8th column; when it is there only UU (both ends unique) is
    kept, so reads from repeats shared between genomes can't create contacts between bins.
    """
    lf = pl.scan_csv(path, separator="\t", has_header=False, comment_prefix="#", infer_schema=False,
                     truncate_ragged_lines=True)
    cols = lf.collect_schema().names()
    lf = lf.rename({cols[1]: "c1", cols[3]: "c2"})
    if len(cols) > 7:
        lf = lf.filter(pl.col(cols[7]) == "UU")
    return lf.select("c1", "c2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help=".pairs file to count (writes contig_pairs.parquet)")
    ap.add_argument("--contig-pairs", help="contig_pairs.parquet from an earlier count")
    ap.add_argument("--contig-bin", help="contig<TAB>bin: call each contig's Hi-C bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-links", type=int, default=20)
    ap.add_argument("--min-share", type=float, default=0.8)
    ap.add_argument("--min-partners", type=int, default=3)
    ap.add_argument("--min-links-wo-top", type=int, default=10)
    ap.add_argument("--max-p-random", type=float, default=1e-6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.pairs:
        pairs = (read_pairs(a.pairs).filter(pl.col("c1") != pl.col("c2"))
                 .select(pl.min_horizontal("c1", "c2").alias("a"), pl.max_horizontal("c1", "c2").alias("b"))
                 .group_by("a", "b").agg(pl.len().alias("n")).sort("a", "b").collect(engine="streaming"))
        pairs.write_parquet(out / "contig_pairs.parquet")
    elif a.contig_pairs:
        pairs = pl.read_parquet(a.contig_pairs)
    else:
        ap.error("give --pairs or --contig-pairs")
    print(f"{pairs['n'].sum()} inter-contig contacts between {pairs.height} contig pairs")
    if not a.contig_bin:
        return

    cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "bin"])
    cb = cb.filter(~pl.col("contig").is_in(["contig", "contig_id"]))
    both = pl.concat([pairs.select(pl.col("a").alias("c"), pl.col("b").alias("o"), "n"),
                      pairs.select(pl.col("b").alias("c"), pl.col("a").alias("o"), "n")])
    total = both.group_by("c").agg(pl.col("n").sum().alias("total"))
    links = (both.join(cb.rename({"contig": "o"}), on="o").group_by("c", "bin")
             .agg(pl.col("n").sum().alias("links"), pl.col("o").n_unique().alias("partners"),
                  pl.col("n").max().alias("top_partner")))
    # share of all contact ends on each bin
    bin_ends = links.group_by("bin").agg(pl.col("links").sum().alias("ends"))
    all_ends = float(both["n"].sum())
    links = (links.join(total, on="c").join(bin_ends, on="bin")
             .with_columns((pl.col("total") * pl.col("ends") / all_ends).alias("expected"))
             .with_columns((pl.col("links") / pl.col("expected")).alias("enrichment"),
                           (pl.col("links") / pl.col("links").sum().over("c")).alias("share")))
    links.write_csv(out / "contig_bin_links.tsv", separator="\t", float_precision=4)

    best = (links.sort(["c", "links", "enrichment"], descending=[False, True, True])
            .group_by("c", maintain_order=True).first())
    to_bins = links.group_by("c").agg(pl.col("links").sum().alias("to_bins"))
    best = best.join(to_bins, on="c").with_columns((pl.col("ends") / pl.col("ends").sum()).alias("b"))
    b_all = bin_ends["ends"].to_numpy() / bin_ends["ends"].sum()
    bmap = dict(zip(bin_ends["bin"].to_list(), b_all))
    bh = np.array([bmap[x] for x in best["bin"]])
    p_random = binom.sf(best["links"].to_numpy() - 1, best["to_bins"].to_numpy(), bh)
    best = best.drop("b").with_columns(pl.Series("p_random", p_random),
                                       (pl.col("links") - pl.col("top_partner")).alias("links_wo_top"))
    calls = (best.with_columns(((pl.col("links") >= a.min_links) & (pl.col("share") >= a.min_share)
                                & (pl.col("partners") >= a.min_partners)
                                & (pl.col("links_wo_top") >= a.min_links_wo_top)
                                & (pl.col("p_random") <= a.max_p_random)).alias("confident"))
             .join(cb.rename({"contig": "c", "bin": "given_bin"}), on="c", how="left")
             .rename({"c": "contig", "bin": "hic_bin"}))
    calls.write_csv(out / "hic_calls.tsv", separator="\t", float_precision=4)
    with open(out / "hic_call_rule.txt", "w") as fh:
        fh.write(f"min_links={a.min_links} min_share={a.min_share} min_partners={a.min_partners} "
                 f"min_links_wo_top={a.min_links_wo_top} max_p_random={a.max_p_random}\n")
    conf = calls.filter("confident")
    binned = conf.filter(pl.col("given_bin").is_not_null())
    print(f"{calls.height} contigs with contacts to bins; {conf.height} confident Hi-C calls; of these "
          f"{binned.height} binned: {int((binned['hic_bin'] == binned['given_bin']).sum())} agree with their bin")


if __name__ == "__main__":
    main()
