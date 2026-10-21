"""Score contamination detection and contig inclusion against truth_actions.tsv.

Works on any tool that writes Nanomotif-style outputs: a contamination table (column `contig`) and
a refined contig -> bin table (`new_contig_bin.tsv`, header contig, bin).

    python scripts/evaluate_refinement.py --truth scen1/truth_actions.tsv \
        --contamination out/bin_contamination.tsv --new-bins out/new_contig_bin.tsv --name pbinner

Prints one row of metrics and appends it to --table if given.
Contamination: precision / recall of flagged contigs (contaminant or orphan_contaminant = positive).
Inclusion: of the contigs that should be (re)included (missing members, flagged contaminants of a
binned genome), how many end up in their true bin; of the contigs that should stay unbinned (orphans,
unknowns), how many are wrongly placed in a bin.
Bins: bp-weighted completeness and contamination of all bins, before and after refinement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def bin_quality(assign: pl.DataFrame, truth: pl.DataFrame) -> tuple[float, float]:
    """Mean bp completeness and bp contamination over the kept bins, CheckM-style but exact."""
    d = assign.join(truth.select("contig", "length", "true_bin"), on="contig")
    size = truth.filter(pl.col("true_bin") != "").group_by("true_bin").agg(pl.col("length").sum().alias("genome"))
    per = d.group_by("bin").agg(
        pl.col("length").filter(pl.col("true_bin") == pl.col("bin")).sum().alias("own"),
        pl.col("length").sum().alias("total"))
    per = per.join(size, left_on="bin", right_on="true_bin", how="left")
    comp = (per["own"] / per["genome"]).mean()
    cont = ((per["total"] - per["own"]) / per["total"]).mean()
    return float(comp), float(cont)


def evaluate(truth_path, contamination_path, new_bins_path, name="") -> dict:
    t = pl.read_csv(truth_path, separator="\t").with_columns(pl.col("true_bin", "given_bin").fill_null(""))
    flagged = set()
    if contamination_path and Path(contamination_path).exists():
        cdf = pl.read_csv(contamination_path, separator="\t", infer_schema_length=0)
        flagged = set(cdf["contig"].to_list()) if cdf.height else set()
    nb = pl.read_csv(new_bins_path, separator="\t", infer_schema_length=0)
    nb = nb.rename(dict(zip(nb.columns[:2], ["contig", "bin"])))
    final = dict(zip(nb["contig"], nb["bin"]))
    t = t.with_columns(pl.col("contig").is_in(list(flagged)).alias("flagged"),
                       pl.col("contig").replace_strict(final, default="").alias("final_bin"))

    binned = t.filter(pl.col("given_bin") != "")
    pos = binned["category"].is_in(["contaminant", "orphan_contaminant"])
    tp = int((binned["flagged"] & pos).sum()); fp = int((binned["flagged"] & ~pos).sum())
    fn = int((~binned["flagged"] & pos).sum())
    # should end up in their true bin: missing members, and contaminants whose genome has a bin
    want = t.filter(pl.col("category").is_in(["missing", "contaminant"]))
    included_ok = int((want["final_bin"] == want["true_bin"]).sum())
    new_in = t.filter((pl.col("final_bin") != "") & (pl.col("final_bin") != pl.col("given_bin")))
    wrong_in = int((new_in["final_bin"] != new_in["true_bin"]).sum())
    stay = t.filter(pl.col("category").is_in(["orphan", "unknown", "orphan_contaminant"]))
    stay_wrong = int((stay["final_bin"] != "").sum())

    given = t.filter(pl.col("given_bin") != "").select("contig", pl.col("given_bin").alias("bin"))
    after = t.filter(pl.col("final_bin") != "").select("contig", pl.col("final_bin").alias("bin"))
    c0, k0 = bin_quality(given, t)
    c1, k1 = bin_quality(after, t)
    return {
        "tool": name, "contaminants": tp + fn, "flagged": tp + fp,
        "contam_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "contam_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "to_include": want.height, "included_correct": included_ok,
        "inclusion_recall": included_ok / want.height if want.height else float("nan"),
        "new_placements": new_in.height, "new_placements_wrong": wrong_in,
        "should_stay_unbinned": stay.height, "placed_wrongly": stay_wrong,
        "completeness_before": c0, "completeness_after": c1,
        "contamination_before": k0, "contamination_after": k1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--contamination")
    ap.add_argument("--new-bins", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--table", help="append the row to this TSV")
    a = ap.parse_args()
    row = evaluate(a.truth, a.contamination, a.new_bins, a.name)
    df = pl.DataFrame([row])
    if a.table:
        p = Path(a.table)
        if p.exists():
            df = pl.concat([pl.read_csv(p, separator="\t"), df], how="diagonal_relaxed")
        df.write_csv(p, separator="\t", float_precision=4)
    for k, v in row.items():
        print(f"{k:<24} {v:.4f}" if isinstance(v, float) else f"{k:<24} {v}")


if __name__ == "__main__":
    main()
