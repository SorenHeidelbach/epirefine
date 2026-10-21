"""Compare pbinner assignments and Nanomotif include_contigs to a reference truth table.

    python scripts/compare_nanomotif.py --truth truth.tsv --pbinner run/assignments.tsv \
        --nanomotif nanomotif_include/include_contigs.tsv --out comparison.tsv
truth: columns contig, expected (bin or "unknown"), binned_as (empty if unbinned).
"""
import argparse

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pbinner", required=True)
    ap.add_argument("--nanomotif", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = pl.read_csv(a.truth, separator="\t").with_columns(pl.col("binned_as").fill_null(""))
    p = pl.read_csv(a.pbinner, separator="\t").select(
        "contig", "n_sites", "status", pl.col("assignment").alias("pb_map"), pl.col("posterior").alias("pb_post"),
        pl.col("credible_set_95").alias("pb_set"))
    nm = (pl.read_csv(a.nanomotif, separator="\t").group_by("contig")
          .agg(pl.col("assigned_bin").first().alias("nm_bin"), pl.col("mean_prob").first().alias("nm_prob"),
               pl.col("confidence").first().alias("nm_conf")))
    d = t.join(p, on="contig", how="left").join(nm, on="contig", how="left")
    d.write_csv(a.out, separator="\t")
    u = d.filter((pl.col("binned_as") == "") & (pl.col("expected") != "unknown"))
    n = u.height
    pb_conf = u.filter(pl.col("status") == "assigned")
    nm_any = u.filter(pl.col("nm_bin").is_not_null())
    nm_inc = u.filter(pl.col("nm_conf").is_in(["high_confidence"]) | (pl.col("nm_prob") >= 0.8))
    rows = [
        ("pbinner MAP (any posterior)", n, int((u["pb_map"] == u["expected"]).sum())),
        ("pbinner 'assigned' (single-bin 95% set)", pb_conf.height, int((pb_conf["pb_map"] == pb_conf["expected"]).sum())),
        ("pbinner truth inside 95% credible set", n, int(sum(e in s.split(",") for e, s in zip(u["expected"], u["pb_set"])))),
        ("Nanomotif any proposal (all confidences)", nm_any.height, int((nm_any["nm_bin"] == nm_any["expected"]).sum())),
        ("Nanomotif included (mean_prob >= 0.8)", nm_inc.height, int((nm_inc["nm_bin"] == nm_inc["expected"]).sum())),
    ]
    print(f"unbinned contigs with a known host: {n}")
    for name, k, ok in rows:
        print(f"  {name:<45} called {k:>3}   correct {ok:>3}")


if __name__ == "__main__":
    main()
