"""Re-derive pbinner's refinement at other decision thresholds and score each against the truth.

The thresholds only turn posteriors into decisions, so no rerun is needed: contig_decisions.tsv holds
P(contaminant) and the best alternative bin with its probability for every contig.

    python scripts/sweep_thresholds.py --run scen1/pbinner --truth scen1/truth_actions.tsv \
        --include 0.5 0.8 0.9 0.95 --contamination 0.5 --out sweep.tsv
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_refinement import evaluate  # noqa: E402


def decisions(dec: pl.DataFrame, t_cont: float, t_inc: float):
    d = dec.with_columns(pl.col("given_bin").fill_null(""), pl.col("p_contaminant").fill_nan(None))
    cont = (d["given_bin"] != "") & (d["p_contaminant"].fill_null(0) >= t_cont) & (d["status"] != "sole_contig_of_bin")
    cand = (d["given_bin"] == "") | cont
    inc = cand & (d["best_alternative"] != "unknown") & (d["p_best_alternative"] >= t_inc)
    new = pl.when(inc).then(d["best_alternative"]).when((d["given_bin"] != "") & ~cont).then(d["given_bin"]).otherwise(pl.lit(""))
    nb = d.select("contig", new.alias("bin")).filter(pl.col("bin") != "")
    return d.filter(cont).select("contig"), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--truth", nargs="+", required=True)
    ap.add_argument("--include", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--contamination", type=float, nargs="+", default=[0.5])
    ap.add_argument("--label", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, (run, truth) in enumerate(zip(a.run, a.truth)):
            dec = pl.read_csv(Path(run) / "contig_decisions.tsv", separator="\t")
            for tc in a.contamination:
                for ti in a.include:
                    c, nb = decisions(dec, tc, ti)
                    cp, np_ = Path(tmp) / "c.tsv", Path(tmp) / "n.tsv"
                    c.write_csv(cp, separator="\t"); nb.write_csv(np_, separator="\t")
                    r = evaluate(truth, cp, np_, "pbinner")
                    rows.append({"run": a.label[k] if a.label else run, "contamination_threshold": tc,
                                 "include_threshold": ti, **r})
    df = pl.DataFrame(rows)
    if a.out:
        df.write_csv(a.out, separator="\t", float_precision=4)
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(df.select("run", "contamination_threshold", "include_threshold", "contam_recall", "contam_precision",
                        "inclusion_recall", "new_placements_wrong", "placed_wrongly", "contamination_after",
                        "completeness_after"))


if __name__ == "__main__":
    main()
