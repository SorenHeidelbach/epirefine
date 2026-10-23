"""pbinner's assignment probabilities against Hi-C contact support, per contig.

Hi-C support of contig c for bin h = contacts between c and h's contigs / c's contacts to all binned
contigs (share in contig_bin_links.tsv). It is independent of methylation, so if pbinner's
probabilities are calibrated, contigs given probability p for bin h should have Hi-C pointing to h
about a fraction p of the time. Three comparisons:

  own        binned contigs: P(belongs to its bin) = 1 - p_contaminant  vs  support for its bin
  alt        contigs pbinner could move or include: p_best_alternative  vs  support for that bin
  map        every contig: p_map (most probable bin, or "unknown")       vs  support for that bin

Only contigs with at least --min-links contacts to binned contigs are scored. Per comparison and
probability interval, calibration.tsv gives the mean probability, the mean Hi-C support, and the
share of contigs whose confident Hi-C call is that bin (with a Wilson 95% interval).

    python hic/probability_vs_hic.py --decisions pbinner/contig_decisions.tsv \
        --links hic/contig_bin_links.tsv --hic-calls hic/hic_calls.tsv --out hic_probability/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

EDGES = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0001]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--links", required=True)
    ap.add_argument("--hic-calls", required=True)
    ap.add_argument("--min-links", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pl.read_csv(a.decisions, separator="\t", infer_schema_length=0).with_columns(
        pl.col("length").cast(pl.Int64), pl.col("n_sites").cast(pl.Int64),
        *[pl.col(c).cast(pl.Float64) for c in ("p_contaminant", "p_best_alternative", "p_map", "p_unknown")])
    links = (pl.read_csv(a.links, separator="\t", schema_overrides={"c": pl.Utf8, "bin": pl.Utf8})
             .select(pl.col("c").alias("contig"), "bin", "links", "share"))
    total = links.group_by("contig").agg(pl.col("links").sum().alias("hic_links"))
    calls = (pl.read_csv(a.hic_calls, separator="\t", infer_schema_length=0)
             .select("contig", "hic_bin", (pl.col("confident") == "true").alias("hic_confident")))
    base = d.join(total, on="contig", how="left").join(calls, on="contig", how="left").with_columns(
        pl.col("hic_links").fill_null(0), pl.col("hic_confident").fill_null(False))

    def compare(name, rows, bin_col, p_expr):
        x = (rows.with_columns(pl.col(bin_col).alias("target"), p_expr.alias("p"), pl.lit(name).alias("comparison"))
             .join(links.rename({"bin": "target"}), on=["contig", "target"], how="left")
             .with_columns(pl.col("share").fill_null(0.0).alias("hic_support"), pl.col("links").fill_null(0).alias("hic_links_target"),
                           (pl.col("hic_confident") & (pl.col("hic_bin") == pl.col("target"))).fill_null(False).alias("hic_agrees")))
        return x.select("comparison", "contig", "length", "n_sites", "status", "given_bin", "target", "p",
                        "hic_links", "hic_links_target", "hic_support", "hic_bin", "hic_confident", "hic_agrees")

    parts = [
        compare("own", base.filter((pl.col("given_bin") != "") & pl.col("p_contaminant").is_not_nan()),
                "given_bin", 1 - pl.col("p_contaminant")),
        compare("alt", base.filter(pl.col("best_alternative") != "unknown"), "best_alternative", pl.col("p_best_alternative")),
        compare("map", base.filter(pl.col("map") != "unknown"), "map", pl.col("p_map")),
    ]
    x = pl.concat(parts).filter(pl.col("hic_links") >= a.min_links)
    x.write_csv(out / "contigs.tsv", separator="\t", float_precision=4)
    rows = []
    for (name,), g in x.group_by("comparison", maintain_order=True):
        idx = np.digitize(g["p"].to_numpy(), EDGES) - 1
        for i in range(len(EDGES) - 1):
            s = g.filter(pl.Series(idx == i))
            n = s.height
            k = int(s["hic_agrees"].sum()) if n else 0
            lo, hi = wilson(k, n)
            rows.append({"comparison": name, "p_from": EDGES[i], "p_to": min(EDGES[i + 1], 1.0), "contigs": n,
                         "mean_p": s["p"].mean() if n else None, "mean_hic_support": s["hic_support"].mean() if n else None,
                         "median_hic_support": s["hic_support"].median() if n else None,
                         "hic_agree": k / n if n else None, "hic_agree_lo": lo if n else None, "hic_agree_hi": hi if n else None})
    cal = pl.DataFrame(rows)
    cal.write_csv(out / "calibration.tsv", separator="\t", float_precision=4)
    corr = x.group_by("comparison", maintain_order=True).agg(
        pl.len().alias("contigs"), pl.corr("p", "hic_support", method="spearman").alias("spearman"))
    corr.write_csv(out / "correlation.tsv", separator="\t", float_precision=4)
    with pl.Config(tbl_width_chars=200, tbl_rows=40):
        print(corr)
        print(cal)


if __name__ == "__main__":
    main()
