"""Compare assignments.tsv with a truth table that has columns contig and expected (bin name or 'unknown')."""
from __future__ import annotations

import polars as pl


def evaluate(assignments, truth) -> str:
    a = pl.read_csv(assignments, separator="\t", infer_schema_length=10000)
    t = pl.read_csv(truth, separator="\t", infer_schema_length=10000)
    df = a.join(t.select([c for c in t.columns if c not in a.columns or c == "contig"]), on="contig", how="inner")
    df = df.with_columns(
        (pl.col("assignment") == pl.col("expected")).alias("correct"),
        pl.col("credible_set_95").str.split(",").list.contains(pl.col("expected")).alias("in_set95"),
    )
    lines = [f"{df.height} contigs evaluated",
             f"accuracy (MAP)            {df['correct'].mean():.3f}",
             f"expected in 95% set       {df['in_set95'].mean():.3f}",
             f"mean posterior of MAP     {df['posterior'].mean():.3f}",
             f"confident (>=0.95) wrong  {df.filter((pl.col('posterior') >= 0.95) & ~pl.col('correct')).height}"]
    cols = [c for c in ("contig", "length", "n_sites", "expected", "assignment", "posterior", "p_unknown",
                        "credible_set_95", "correct") if c in df.columns]
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=250):
        lines.append(str(df.select(cols).sort("contig")))
    return "\n".join(lines)
