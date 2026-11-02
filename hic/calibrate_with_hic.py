"""Calibrate epirefine's contamination probability against Hi-C.

epirefine's P(contaminant) measures how strongly a contig's methylation disagrees with its bin. Whether
that disagreement means a foreign genome (or a plasmid, phage or strain variant that lives in the same
cells) is something methylation can't tell; Hi-C can. For binned contigs with enough contacts to
binned contigs (>= --min-contacts), the Hi-C label is "foreign" when fewer than half of those contacts
are with other contigs of its own bin (hic/within_vs_outside.py). A monotone (isotonic, pool adjacent
violators) fit of that label on epirefine's log-odds gives, for every binned contig, the share of
contigs with that much methylation disagreement that Hi-C finds outside their bin:

    p_hic_calibrated = P(Hi-C foreign | epirefine P(contaminant))

Contigs without enough contacts get the fitted value for their epirefine probability too, so the
calibrated probability is defined for every binned contig. The fit is sample-specific.

    python hic/calibrate_with_hic.py --within hic_within/contigs.tsv --decisions epirefine/contig_decisions.tsv \
        --out hic_calibration/
Writes contigs.tsv (epirefine p, calibrated p, Hi-C label where known), curve.tsv (the step function)
and thresholds.tsv (epirefine p needed for a given Hi-C precision of the flags).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Non-decreasing least-squares fit of y on x (pool adjacent violators); returns sorted x, fit."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order].astype(float)
    ws = np.ones_like(ys) if w is None else w[order].astype(float)
    vals, wts, sizes = [], [], []
    for v, wt in zip(ys, ws):
        vals.append(v); wts.append(wt); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); sizes.append(s1 + s2)
    fit = np.repeat(vals, sizes)
    return xs, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--within", required=True, help="contigs.tsv from hic/within_vs_outside.py")
    ap.add_argument("--decisions", required=True, help="epirefine contig_decisions.tsv")
    ap.add_argument("--min-contacts", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = (pl.read_csv(a.decisions, separator="\t", infer_schema_length=0)
         .filter(pl.col("given_bin").is_not_null() & (pl.col("given_bin") != ""))
         .with_columns(pl.col("p_contaminant").cast(pl.Float64))
         .filter(pl.col("p_contaminant").is_not_nan())
         .select("contig", "given_bin", "status", "p_contaminant"))
    w = pl.read_csv(a.within, separator="\t").select("contig", "within", "outside")
    x = d.join(w, on="contig", how="left").with_columns(pl.col("within", "outside").fill_null(0))
    x = x.with_columns(((pl.col("within") + pl.col("outside")) >= a.min_contacts).alias("hic_informative"),
                       (pl.col("within") < pl.col("outside")).alias("hic_foreign"))
    p = np.clip(x["p_contaminant"].to_numpy(), 1e-12, 1 - 1e-12)
    lo = np.log(p) - np.log1p(-p)
    inf = x["hic_informative"].to_numpy()
    xs, fit = isotonic(lo[inf], x["hic_foreign"].to_numpy()[inf].astype(float))
    # step function: calibrated value for any log-odds (right-continuous, clamped at the ends)
    idx = np.clip(np.searchsorted(xs, lo, side="right") - 1, 0, len(xs) - 1)
    x = x.with_columns(pl.Series("p_hic_calibrated", fit[idx]))
    x.write_csv(out / "contigs.tsv", separator="\t", float_precision=4)

    # compact curve: one row per pooled block
    blocks = []
    for v in np.unique(fit):
        sel = fit == v
        blocks.append({"p_from": float(1 / (1 + np.exp(-xs[sel].min()))), "p_to": float(1 / (1 + np.exp(-xs[sel].max()))),
                       "contigs": int(sel.sum()), "hic_foreign_rate": float(v)})
    pl.DataFrame(blocks).sort("p_from").write_csv(out / "curve.tsv", separator="\t", float_precision=4)

    # epirefine threshold for a given Hi-C precision among flagged contigs (contigs with p >= threshold)
    rows = []
    xi = x.filter(pl.col("hic_informative"))
    for thr in (0.5, 0.9, 0.99, 0.999, 0.9999, 1 - 1e-6):
        s = xi.filter(pl.col("p_contaminant") >= thr)
        rows.append({"p_threshold": thr, "flagged_with_hic": s.height,
                     "hic_precision": s["hic_foreign"].mean() if s.height else None,
                     "hic_foreign_caught": int(s["hic_foreign"].sum()), "hic_foreign_total": int(xi["hic_foreign"].sum())})
    pl.DataFrame(rows).write_csv(out / "thresholds.tsv", separator="\t", float_precision=4)
    base = float(xi["hic_foreign"].mean())
    with pl.Config(tbl_width_chars=200, tbl_rows=30):
        print(f"{xi.height} binned contigs with >= {a.min_contacts} contacts; Hi-C foreign {base:.3f}")
        print(pl.DataFrame(blocks).sort("p_from"))
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
