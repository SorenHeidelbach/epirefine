"""pbinner's assignment probabilities against the confidence of the Hi-C contact map, per contig.

Hi-C probability: the contact map turned into a posterior over bins, so it reads on the same scale as
pbinner's. Of a contig's n contacts to binned contigs, n_h go to bin h. If the contig belongs to h, a
fraction f of its contacts are within its genome and the rest are random ligations between cells,
which land on bin k in proportion to k's share b_k of all contact ends; if its genome has no bin,
all contacts are random:

    P(n | h)    ∝ (f + (1-f) b_h)^n_h · Π_{k≠h} ((1-f) b_k)^n_k
    P(n | none) ∝ Π_k b_k^n_k
    => log P(n | h) / P(n | none) = n_h log((f + (1-f) b_h) / ((1-f) b_h)) + n log(1-f)

Contacts aren't independent (they cluster on related strains, mapping artefacts), so the counts are
Dirichlet-multinomial around that expectation, with concentration α fitted on the binned contigs.
Prior: P(none) = --prior-none, the rest uniform over bins. f is estimated from the binned contigs
(median of (share_own - b_h) / (1 - b_h) over contigs with >= --min-links contacts). The prior uses
neither methylation nor the binning, so the Hi-C probability is independent of pbinner's: few
contacts give a Hi-C probability near the prior, many contacts a confident one.

Three comparisons, each pairing a pbinner probability with the Hi-C probability of the same bin:

  own        binned contigs: P(belongs to its bin) = 1 - p_contaminant
  alt        p_best_alternative (the bin pbinner would move or include the contig into)
  map        p_map (pbinner's most probable bin; contigs whose MAP is "unknown" left out)

Only contigs with at least --min-links contacts to binned contigs are scored. Outputs:
  contigs.tsv       one row per comparison and contig: p (pbinner), hic_p (Hi-C), contact share, ...
  calibration.tsv   per comparison and pbinner probability interval: mean p, mean hic_p, share of
                    contigs whose Hi-C probability for the bin is >= 0.5 (Wilson 95% interval)
  correlation.tsv   Spearman correlation of p and hic_p per comparison
  params.tsv        f, prior, contig counts

    python hic/probability_vs_hic.py --decisions pbinner/contig_decisions.tsv \
        --links hic/contig_bin_links.tsv --contig-bin contig_bin.tsv --out hic_probability/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import gammaln, logsumexp

EDGES = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0001]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _g(a, n):
    """log Γ(a + n) - log Γ(a): one category's term of a Dirichlet-multinomial log-likelihood."""
    return gammaln(a + n) - gammaln(a)


def hic_posterior(links: pl.DataFrame, given: pl.DataFrame, min_links: int, prior_none: float):
    """Posterior over bins (and 'none') from contacts, with a Dirichlet-multinomial noise model.

    A contig's contacts to bins are multinomial with probabilities θ ~ Dirichlet(α m), m the expected
    split under the hypothesis (above). A plain multinomial (α = ∞) treats every contact as
    independent, so one contact to a rare bin decides; real contacts cluster (shared sequence between
    strains, mapping artefacts), and α, fitted on the binned contigs with their own bin as the
    hypothesis, measures how much. Returns the long table (contig, bin, links, n, hic_p), P(none), f, α.
    """
    bins = links.select("bin", "ends").unique("bin").sort("bin")
    b = (bins["ends"] / bins["ends"].sum()).to_numpy()
    bidx = {x: i for i, x in enumerate(bins["bin"].to_list())}
    H = len(b)
    x = (links.join(links.group_by("contig").agg(pl.col("links").sum().alias("n")), on="contig")
         .with_columns(pl.col("bin").replace_strict(bidx, return_dtype=pl.Int64).alias("i")).sort("contig", "i"))
    # f from binned contigs: share of contacts to their own bin above the random expectation
    own = x.join(given, on="contig").filter((pl.col("bin") == pl.col("given_bin")) & (pl.col("n") >= min_links))
    bo = b[own["i"].to_numpy()]
    f = float(np.clip(np.median((own["links"].to_numpy() / own["n"].to_numpy() - bo) / (1 - bo)), 0.01, 0.99))
    # α by maximum likelihood on binned contigs, hypothesis = own bin
    fit = x.join(given, on="contig").filter(pl.col("n") >= min_links)
    fb = b[fit["i"].to_numpy()]
    is_own = (fit["bin"] == fit["given_bin"]).to_numpy()
    m = np.where(is_own, f + (1 - f) * fb, (1 - f) * fb)
    nk = fit["links"].to_numpy()
    tot = fit.group_by("contig", maintain_order=True).agg(pl.col("n").first())["n"].to_numpy()
    grid = np.logspace(-1, 4, 51)
    ll = [(_g(al * m, nk).sum() - _g(al, tot).sum()) for al in grid]
    alpha = float(grid[int(np.argmax(ll))])
    bh = b[x["i"].to_numpy()]
    n_k = x["links"].to_numpy()
    rand = _g((1 - f) * alpha * bh, n_k)
    x = x.with_columns(pl.Series("d_rand", rand - _g(alpha * bh, n_k)),
                       pl.Series("d_own", _g(alpha * (f + (1 - f) * bh), n_k) - rand))
    # log P(n | h) - log P(n | none) = S0 + d_own[h], S0 = Σ_k d_rand[k]; bins without contacts: S0
    x = x.with_columns(pl.col("d_rand").sum().over("contig").alias("S0"))
    lp_bin = np.log((1 - prior_none) / H)
    x = x.with_columns((pl.col("S0") + pl.col("d_own") + lp_bin).alias("lp"))
    per = x.group_by("contig").agg(pl.col("lp"), pl.col("S0").first(), pl.len().alias("k"))
    norm = {}
    for c, lp, s0, k in per.iter_rows():
        rest = s0 + lp_bin + np.log(H - k) if H > k else -np.inf
        norm[c] = float(logsumexp(np.r_[lp, rest, np.log(prior_none)]))
    x = x.with_columns((pl.col("lp") - pl.col("contig").replace_strict(norm, return_dtype=pl.Float64)).exp().alias("hic_p"))
    none = pl.DataFrame({"contig": list(norm), "hic_p_none": [float(np.exp(np.log(prior_none) - v)) for v in norm.values()]})
    return x.select("contig", "bin", "links", "n", "hic_p"), none, f, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--links", required=True, help="contig_bin_links.tsv from hic/contacts.py")
    ap.add_argument("--contig-bin", required=True, help="the original contig -> bin table (to estimate f)")
    ap.add_argument("--min-links", type=int, default=10)
    ap.add_argument("--prior-none", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pl.read_csv(a.decisions, separator="\t", infer_schema_length=0).with_columns(
        pl.col("length").cast(pl.Int64), pl.col("n_sites").cast(pl.Int64), pl.col("given_bin").fill_null(""),
        *[pl.col(c).cast(pl.Float64) for c in ("p_contaminant", "p_best_alternative", "p_map", "p_unknown")])
    links = (pl.read_csv(a.links, separator="\t", schema_overrides={"c": pl.Utf8, "bin": pl.Utf8})
             .select(pl.col("c").alias("contig"), "bin", "links", "ends"))
    given = pl.read_csv(a.contig_bin, separator="\t", has_header=False, new_columns=["contig", "given_bin"],
                        infer_schema_length=0)
    post, none, f, alpha = hic_posterior(links, given, a.min_links, a.prior_none)
    print(f"Hi-C within-genome contact fraction f = {f:.3f}; Dirichlet-multinomial α = {alpha:.3g}; "
          f"prior P(none) = {a.prior_none}")
    hic_map = (post.sort(["contig", "hic_p"], descending=[False, True]).group_by("contig", maintain_order=True).first()
               .select("contig", pl.col("bin").alias("hic_map"), pl.col("hic_p").alias("hic_p_map"), pl.col("n").alias("hic_links")))
    base = d.join(hic_map, on="contig", how="inner").join(none, on="contig", how="left")

    def compare(name, rows, bin_col, p_expr):
        return (rows.with_columns(pl.col(bin_col).alias("target"), p_expr.alias("p"), pl.lit(name).alias("comparison"))
                .join(post.select("contig", pl.col("bin").alias("target"), pl.col("links").alias("hic_links_target"), "hic_p"),
                      on=["contig", "target"], how="left")
                .with_columns(pl.col("hic_links_target").fill_null(0), pl.col("hic_p").fill_null(0.0))
                .with_columns((pl.col("hic_links_target") / pl.col("hic_links")).alias("hic_share"))
                .select("comparison", "contig", "length", "n_sites", "status", "given_bin", "target", "p", "hic_p",
                        "hic_share", "hic_links", "hic_links_target", "hic_map", "hic_p_map", "hic_p_none"))

    x = pl.concat([
        compare("own", base.filter((pl.col("given_bin") != "") & pl.col("p_contaminant").is_not_nan()),
                "given_bin", 1 - pl.col("p_contaminant")),
        compare("alt", base.filter(pl.col("best_alternative") != "unknown"), "best_alternative", pl.col("p_best_alternative")),
        compare("map", base.filter(pl.col("map") != "unknown"), "map", pl.col("p_map")),
    ]).filter(pl.col("hic_links") >= a.min_links)
    x.write_csv(out / "contigs.tsv", separator="\t", float_precision=4)

    rows = []
    for (name,), g in x.group_by("comparison", maintain_order=True):
        idx = np.digitize(g["p"].to_numpy(), EDGES) - 1
        for i in range(len(EDGES) - 1):
            s = g.filter(pl.Series(idx == i))
            n = s.height
            k = int((s["hic_p"] >= 0.5).sum()) if n else 0
            lo, hi = wilson(k, n)
            rows.append({"comparison": name, "p_from": EDGES[i], "p_to": min(EDGES[i + 1], 1.0), "contigs": n,
                         "mean_p": s["p"].mean() if n else None, "mean_hic_p": s["hic_p"].mean() if n else None,
                         "median_hic_p": s["hic_p"].median() if n else None,
                         "mean_hic_share": s["hic_share"].mean() if n else None,
                         "hic_supports": k / n if n else None, "hic_supports_lo": lo, "hic_supports_hi": hi})
    pl.DataFrame(rows).write_csv(out / "calibration.tsv", separator="\t", float_precision=4)
    corr = x.group_by("comparison", maintain_order=True).agg(
        pl.len().alias("contigs"), pl.corr("p", "hic_p", method="spearman").alias("spearman_p_vs_hic_p"),
        pl.corr("p", "hic_share", method="spearman").alias("spearman_p_vs_share"))
    corr.write_csv(out / "correlation.tsv", separator="\t", float_precision=4)
    pl.DataFrame({"parameter": ["f", "alpha", "prior_none", "min_links", "contigs_with_hic"],
                  "value": [f, alpha, a.prior_none, a.min_links, base.height]}).write_csv(out / "params.tsv", separator="\t")
    with pl.Config(tbl_width_chars=220, tbl_rows=40, tbl_cols=20):
        print(corr)
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
