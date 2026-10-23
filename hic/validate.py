"""Score bin refinement against Hi-C: an independent, physical signal of which contigs share a cell.

Inputs: hic_calls.tsv (hic/contacts.py), the original contig -> bin table, and for each method its
refined contig -> bin table (new_contig_bin.tsv) and contamination table (bin_contamination.tsv).
Only contigs with a confident Hi-C call are scored:

  binned contigs   Hi-C bin == given bin -> member;  Hi-C bin != given bin -> contaminant
                   flagged member = false flag, flagged contaminant = confirmed, unflagged contaminant = missed
  unbinned contigs placed in the Hi-C bin = correct inclusion, in another bin = wrong inclusion,
                   not placed = missed inclusion

    python hic/validate.py --hic-calls hic/hic_calls.tsv --contig-bin contig_bin.tsv \
        --method pbinner pbinner/new_contig_bin.tsv pbinner/bin_contamination.tsv \
        --method nanomotif nanomotif/new_contig_bin.tsv nanomotif/bin_contamination.tsv --out hic_validation/
Writes summary.tsv (one row per method) and contigs.tsv (one row per scored contig and method).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def read_table(path, names=("contig", "bin")) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", has_header=False, infer_schema_length=0, truncate_ragged_lines=True)
    df = df.select(df.columns[:2])
    df.columns = list(names)
    return df.filter(~pl.col("contig").is_in(["contig", "contig_id", "seq_name"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hic-calls", required=True)
    ap.add_argument("--contig-bin", required=True)
    ap.add_argument("--method", nargs=3, action="append", metavar=("NAME", "NEW_CONTIG_BIN", "CONTAMINATION"),
                    required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calls = (pl.read_csv(a.hic_calls, separator="\t").filter(pl.col("confident"))
             .select("contig", "hic_bin", "links", "share", "enrichment"))
    given = read_table(a.contig_bin, ("contig", "given_bin"))
    base = calls.join(given, on="contig", how="left").with_columns(pl.col("given_bin").fill_null(""))
    rows, per_contig = [], []
    for name, new_path, cont_path in a.method:
        new = read_table(new_path, ("contig", "new_bin"))
        flagged = set(pl.read_csv(cont_path, separator="\t", infer_schema_length=0)["contig"].to_list()) \
            if Path(cont_path).exists() and Path(cont_path).stat().st_size > 0 else set()
        x = (base.join(new, on="contig", how="left").with_columns(pl.col("new_bin").fill_null(""),
             pl.col("contig").is_in(list(flagged)).alias("flagged"))
             .with_columns(pl.when(pl.col("given_bin") != "")
                           .then(pl.when(pl.col("hic_bin") == pl.col("given_bin"))
                                 .then(pl.when(pl.col("flagged")).then(pl.lit("false flag")).otherwise(pl.lit("member kept")))
                                 .otherwise(pl.when(pl.col("flagged")).then(pl.lit("contaminant confirmed")).otherwise(pl.lit("contaminant missed"))))
                           .otherwise(pl.when(pl.col("new_bin") == pl.col("hic_bin")).then(pl.lit("inclusion correct"))
                                      .when(pl.col("new_bin") != "").then(pl.lit("inclusion wrong"))
                                      .otherwise(pl.lit("inclusion missed"))).alias("outcome"),
                           pl.lit(name).alias("method")))
        per_contig.append(x)
        c = dict(x.group_by("outcome").len().iter_rows())
        get = lambda k: c.get(k, 0)  # noqa: E731
        flags = get("contaminant confirmed") + get("false flag")
        cont = get("contaminant confirmed") + get("contaminant missed")
        placed = get("inclusion correct") + get("inclusion wrong")
        rows.append({"method": name, "scored_binned": get("member kept") + get("false flag") + cont,
                     "hic_contaminants": cont, "flagged": flags,
                     "flag_precision": get("contaminant confirmed") / flags if flags else None,
                     "contaminant_recall": get("contaminant confirmed") / cont if cont else None,
                     "false_flags": get("false flag"),
                     "scored_unbinned": placed + get("inclusion missed"), "placed": placed,
                     "inclusion_precision": get("inclusion correct") / placed if placed else None,
                     "inclusion_recall": get("inclusion correct") / (placed + get("inclusion missed"))
                     if placed + get("inclusion missed") else None})
    summary = pl.DataFrame(rows)
    summary.write_csv(out / "summary.tsv", separator="\t", float_precision=4)
    pl.concat(per_contig).write_csv(out / "contigs.tsv", separator="\t", float_precision=4)
    with pl.Config(tbl_width_chars=200, tbl_cols=20):
        print(summary)


if __name__ == "__main__":
    main()
