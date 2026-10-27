"""Hi-C membership calls by linkage density, the way ProxiMeta and similar tools judge host links.

Raw contact counts favour abundant genomes and long contigs: on ZymoFecal, 48% of binned contigs at
>= 10x had most of their contacts outside their bin, but only 16% once contact density was compared
with the bin's own internal density. So instead of counts, this compares densities (links per kb^2):

    intra density of bin B   = links between B's contigs / sum over pairs of B's contigs of L_i L_j
    density of contig c to B = links(c, B without c) / (L_c * L_{B without c})
    ratio(c, B)              = density(c, B) / intra density(B)

A ratio near 1 means c contacts B as densely as B's contigs contact each other; ProxiMeta calls a
host link real at an inter/intra ratio >= 0.1 with >= 2 links and >= 10 links inside the MAG
(Press et al. 2017; Uritskiy et al. 2021). Calls:

  binned contig    member        own ratio >= min_ratio with >= min_links links
                   foreign       own ratio < min_ratio, and another eligible bin has ratio >= min_ratio
                                 with >= min_links links and >= min_partners partner contigs
                   not linked    own ratio < min_ratio although the contig has >= min_total links to
                                 binned contigs and no other bin qualifies: Hi-C says it doesn't belong
                                 to its bin, and its own genome has no bin (an orphan contaminant)
                   undetermined  otherwise (too few links, the contig is its bin's only contig, or its
                                 bin has < min_intra internal links)
  unbinned contig  placed        best eligible bin has ratio >= min_ratio, >= min_links links,
                                 >= min_partners partners and >= min_fold x the second best ratio
                   not placed    otherwise

min_ratio is ProxiMeta's 0.1. The in-sample distribution of clear members' ratios to other bins
(>= --min-links links) is reported as a diagnostic, not used as the floor: on ZymoFecal only 40 of
1,328 clear members reach 5 links to another bin, and those ratios are high (99th percentile 17.9),
i.e. real shared signal (split bins of one genome, related strains, shared elements), not ligation
noise; the link and partner minimums are what filter the spurious contacts. Contacts are inter-contig, unique,
MAPQ-filtered pairs (hic/contacts.py).

    python hic/density.py --pairs hic/contig_pairs.parquet --contig-bin contig_bin.tsv \
        --lengths contig_lengths.tsv --out hic_density/
Writes calls.tsv (one row per contig with contacts), bins.tsv (intra density per bin) and params.tsv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--contig-bin", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--min-ratio", type=float, default=0.1)
    ap.add_argument("--min-links", type=int, default=5)
    ap.add_argument("--min-partners", type=int, default=3)
    ap.add_argument("--min-intra", type=int, default=10)
    ap.add_argument("--min-fold", type=float, default=3.0)
    ap.add_argument("--min-total", type=int, default=20, help="links to binned contigs needed for 'not linked'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = pl.read_parquet(a.pairs)
    cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "bin"], infer_schema_length=0)
    cb = cb.filter(~pl.col("contig").is_in(["contig", "contig_id"]))
    L = pl.read_csv(a.lengths, separator="\t", schema_overrides={"contig": pl.Utf8}).select("contig", "length")
    kb = L.with_columns((pl.col("length") / 1000.0).alias("kb"))

    # per bin: total kb, sum of squared kb, internal links -> intra density
    bl = cb.join(kb, on="contig").group_by("bin").agg(pl.col("kb").sum().alias("bin_kb"), (pl.col("kb") ** 2).sum().alias("kb2"),
                                                       pl.len().alias("contigs"))
    both = pl.concat([pairs.select(pl.col("a").alias("c"), pl.col("b").alias("o"), "n"),
                      pairs.select(pl.col("b").alias("c"), pl.col("a").alias("o"), "n")])
    x = both.join(cb.rename({"contig": "c", "bin": "cbin"}), on="c", how="left").join(cb.rename({"contig": "o", "bin": "obin"}), on="o")
    intra = (x.filter(pl.col("cbin") == pl.col("obin")).group_by("obin").agg((pl.col("n").sum() / 2).alias("intra_links"))
             .rename({"obin": "bin"}))
    bins = (bl.join(intra, on="bin", how="left").with_columns(pl.col("intra_links").fill_null(0))
            .with_columns(((pl.col("bin_kb") ** 2 - pl.col("kb2")) / 2).alias("pair_kb2"))
            .with_columns((pl.col("intra_links") / pl.col("pair_kb2")).alias("intra_density")))
    bins.write_csv(out / "bins.tsv", separator="\t", float_precision=6)

    # per contig and bin: links, partners, density ratio (the contig's own length removed from its bin)
    per = (x.group_by("c", "obin").agg(pl.col("n").sum().alias("links"), pl.col("o").n_unique().alias("partners"))
           .rename({"obin": "bin"})
           .join(cb.rename({"contig": "c", "bin": "cbin"}), on="c", how="left")
           .join(kb.rename({"contig": "c"}).select("c", pl.col("kb").alias("c_kb")), on="c", how="left")
           .join(bins.select("bin", "bin_kb", "intra_links", "intra_density"), on="bin")
           .with_columns(pl.when(pl.col("bin") == pl.col("cbin")).then(pl.col("bin_kb") - pl.col("c_kb")).otherwise(pl.col("bin_kb")).alias("rest_kb"))
           .with_columns((pl.col("links") / (pl.col("c_kb") * pl.col("rest_kb")) / pl.col("intra_density")).alias("ratio"),
                         (pl.col("intra_links") >= a.min_intra).alias("eligible")))

    own = per.filter(pl.col("bin") == pl.col("cbin")).select("c", pl.col("links").alias("own_links"),
                                                              pl.col("partners").alias("own_partners"), pl.col("ratio").alias("own_ratio"))
    other = (per.filter((pl.col("bin") != pl.col("cbin")) | pl.col("cbin").is_null()).filter("eligible")
             .sort(["c", "ratio"], descending=[False, True]))
    best = other.group_by("c", maintain_order=True).agg(
        pl.col("bin").first().alias("best_bin"), pl.col("ratio").first().alias("best_ratio"),
        pl.col("links").first().alias("best_links"), pl.col("partners").first().alias("best_partners"),
        pl.col("ratio").slice(1, 1).first().alias("second_ratio"))

    # in-sample floor: how densely clear members contact other bins (spurious inter-genome contacts)
    clear = own.filter((pl.col("own_ratio") >= 0.5) & (pl.col("own_links") >= 20)).select("c")
    spur = other.join(clear, on="c").filter(pl.col("links") >= a.min_links)["ratio"]
    floor = float(spur.quantile(0.99)) if spur.len() else 0.0
    share_spur = spur.len() / max(clear.height, 1)
    thr = a.min_ratio

    total = per.group_by("c").agg(pl.col("links").sum().alias("total_links"))
    contigs = (pl.concat([per.select("c").unique()]).join(cb.rename({"contig": "c", "bin": "cbin"}), on="c", how="left")
               .join(total, on="c", how="left")
               .join(own, on="c", how="left").join(best, on="c", how="left")
               .join(bins.select(pl.col("bin").alias("cbin"), pl.col("contigs").alias("bin_contigs"),
                                 pl.col("intra_links").alias("bin_intra_links")), on="cbin", how="left")
               .with_columns(pl.col("own_links", "own_partners", "best_links", "best_partners").fill_null(0),
                             pl.col("own_ratio", "best_ratio", "second_ratio").fill_null(0.0)))
    alt_ok = (pl.col("best_ratio") >= thr) & (pl.col("best_links") >= a.min_links) & (pl.col("best_partners") >= a.min_partners)
    contigs = contigs.with_columns(
        pl.when(pl.col("cbin").is_null())
        .then(pl.when(alt_ok & (pl.col("best_ratio") >= a.min_fold * pl.col("second_ratio"))).then(pl.lit("placed")).otherwise(pl.lit("not placed")))
        .when((pl.col("bin_contigs") < 2) | (pl.col("bin_intra_links").fill_null(0) < a.min_intra)).then(pl.lit("undetermined"))
        .when((pl.col("own_ratio") >= thr) & (pl.col("own_links") >= a.min_links)).then(pl.lit("member"))
        .when(alt_ok & (pl.col("best_ratio") > pl.col("own_ratio"))).then(pl.lit("foreign"))
        .when((pl.col("own_ratio") < thr) & (pl.col("total_links") >= a.min_total)).then(pl.lit("not linked"))
        .otherwise(pl.lit("undetermined")).alias("hic_call")).rename({"c": "contig", "cbin": "given_bin"})
    contigs.write_csv(out / "calls.tsv", separator="\t", float_precision=5)
    pl.DataFrame({"parameter": ["min_ratio_requested", "members_ratio_to_other_bins_p99", "min_ratio_used", "min_links", "min_partners", "min_intra", "min_fold"],
                  "value": [a.min_ratio, floor, thr, a.min_links, a.min_partners, a.min_intra, a.min_fold]}).write_csv(out / "params.tsv", separator="\t")
    print(f"clear members with >= {a.min_links} links to another bin: {share_spur:.1%}, their ratio p99 {floor:.2f}; "
          f"ratio threshold {thr:g}")
    print(contigs.group_by(pl.col("given_bin").is_null().alias("unbinned"), "hic_call").len().sort("unbinned", "hic_call"))


if __name__ == "__main__":
    main()
