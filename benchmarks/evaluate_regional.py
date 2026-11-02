"""Score epirefine on chimera scenarios (benchmarks/chimera_sets.py): long chimeras, long members, long contaminants.

    python benchmarks/evaluate_regional.py --truth chim_1/truth_actions.tsv --decisions run/contig_decisions.tsv

Per kind of planted long contig: how many epirefine flagged as a whole (status contaminant / moved),
called mixed, kept, or left undecided; for chimeras also how well the reported foreign segments match
the true foreign part (base-pair Jaccard of the two intervals' union). Right answers: chimera -> not
flagged as a whole (ideally mixed, with the foreign segment found); long member -> kept; long
contaminant -> flagged; long missing / unbinned chimera -> included into the true bin; unbinned foreign ->
its true bin, or left unbinned when it has none.
"""
from __future__ import annotations

import argparse

import polars as pl


def segments_bp(s: str) -> list[tuple[int, int]]:
    if not s:
        return []
    return [tuple(int(v) for v in seg.split("-")) for seg in s.split(";") if seg]


def jaccard(pred: list[tuple[int, int]], lo: int, hi: int) -> float:
    inter = sum(max(0, min(e, hi) - max(s, lo)) for s, e in pred)
    union = sum(e - s for s, e in pred) + (hi - lo) - inter
    return inter / union if union > 0 else 0.0


def evaluate(truth_path, decisions_path) -> pl.DataFrame:
    t = pl.read_csv(truth_path, separator="\t", infer_schema_length=0).filter(pl.col("kind") != "")
    d = pl.read_csv(decisions_path, separator="\t", infer_schema_length=0)
    seg = d["foreign_segments"] if "foreign_segments" in d.columns else pl.Series([""] * d.height)
    d = d.with_columns(seg.fill_null("").alias("foreign_segments"))
    x = t.join(d.select("contig", "status", pl.col("new_bin").fill_null(""), "foreign_segments"), on="contig", how="left").with_columns(
        pl.col("foreign_start", "foreign_end").cast(pl.Int64))
    x = x.with_columns(pl.struct("foreign_segments", "foreign_start", "foreign_end").map_elements(
        lambda r: jaccard(segments_bp(r["foreign_segments"]), r["foreign_start"], r["foreign_end"]) if r["foreign_start"] >= 0 else None,
        return_dtype=pl.Float64).alias("segment_jaccard"))
    out = x.with_columns(pl.when(pl.col("given_bin").fill_null("") == "")
                         .then(pl.when(pl.col("status") != "included").then(pl.lit("not included"))
                               .when(pl.col("new_bin") == pl.col("true_bin").fill_null("")).then(pl.lit("included, right bin"))
                               .otherwise(pl.lit("included, wrong bin")))
                         .when(pl.col("status").is_in(["contaminant", "moved"])).then(pl.lit("flagged"))
                         .when(pl.col("status") == "mixed").then(pl.lit("mixed"))
                         .when(pl.col("status").is_in(["low_coverage", "too_short", "no_methylation_evidence"])).then(pl.lit("undecided"))
                         .otherwise(pl.lit("kept")).alias("outcome"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, nargs="+")
    ap.add_argument("--decisions", required=True, nargs="+")
    a = ap.parse_args()
    x = pl.concat([evaluate(t, d) for t, d in zip(a.truth, a.decisions)])
    print(x.group_by("kind", "outcome").len().sort("kind", "outcome"))
    ch = x.filter(pl.col("kind") == "chimera")
    print(f"chimeras: segment Jaccard median {ch['segment_jaccard'].median():.2f} "
          f"(mixed only: {ch.filter(pl.col('outcome') == 'mixed')['segment_jaccard'].median()})")


if __name__ == "__main__":
    main()
