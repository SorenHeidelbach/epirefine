"""pbinner's assignment probabilities against the confidence of the Hi-C contact map, per contig.

Hi-C probability: the contact map turned into a posterior over bins (model in hic_posterior), so
it reads on the same scale as pbinner's. Its prior uses neither methylation nor the binning: P(no
bin) = --prior-none, the rest uniform over bins. Few contacts give a Hi-C probability near the
prior, many contacts a confident one; a contig that is its whole bin gets no Hi-C evidence.

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
from scipy.special import logsumexp

EDGES = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0001]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def hic_posterior(links: pl.DataFrame, given: pl.DataFrame, lengths: pl.DataFrame, min_links: int,
                  prior_none: float, grid: int = 64):
    """Posterior over bins (and 'none') for each contig with contacts to bins.

    Only contacts between different contigs are counted. Of a contig's contacts, within-genome ones
    reach other contigs of its genome with probability g = (genome outside the contig) / genome:
    for c in bin h, g = (L_h - L_c) / L_h; for c outside h, g = L_h / (L_h + L_c). With f the
    within-genome share of all its ligations, a contact (to a binned contig) lands on bin h with

        q_h = (f g + (1 - f) b_h) / (f g + 1 - f)                  if c belongs to h
        b_h                                                         if c belongs to no bin

    and the contacts to the other bins are spread in proportion to b either way, so

        log P(n | h) / P(n | none) = log E_f[q_h^n_h (1 - q_h)^(n - n_h)] - n_h log b_h - (n - n_h) log(1 - b_h)

    f varies between contigs (coverage, cell lysis, how much of the genome is binned): f ~ Beta,
    fitted by moments to per-contig estimates from binned contigs with >= min_links contacts. A
    contig that is its whole bin has g = 0 and gets no Hi-C evidence for it (posterior = prior).
    Returns the long table (contig, bin, links, n, hic_p), P(none) per contig, and the Beta (a, b).
    """
    bins = links.select("bin", "ends").unique("bin").sort("bin")
    names = bins["bin"].to_list()
    b = (bins["ends"] / bins["ends"].sum()).to_numpy()
    H = len(b)
    bidx = {x: i for i, x in enumerate(names)}
    L = dict(zip(lengths["contig"], lengths["length"]))
    gl = given.with_columns(pl.col("contig").replace_strict(L, default=0, return_dtype=pl.Int64).alias("length"))
    Lbin = dict(gl.group_by("given_bin").agg(pl.col("length").sum()).iter_rows())
    LH = np.array([Lbin.get(x, 0) for x in names], float)
    gmap = dict(given.iter_rows())
    tot = links.group_by("contig").agg(pl.col("links").sum().alias("n"))

    # f per binned contig from its share to its own bin: share = (f g + (1-f) b) / (f g + 1 - f)
    own = (links.join(given, on="contig").filter(pl.col("bin") == pl.col("given_bin")).join(tot, on="contig")
           .filter(pl.col("n") >= min_links))
    oi = np.array([bidx[x] for x in own["bin"]])
    oL = np.array([L.get(c, 0) for c in own["contig"]], float)
    og = np.clip((LH[oi] - oL) / np.maximum(LH[oi], 1), 0, 1)
    sh = own["links"].to_numpy() / own["n"].to_numpy()
    ok = og > 0.05
    # solve share (f g + 1 - f) = f g + (1-f) b  =>  f = (share - b) / (g - b + share (1 - g))
    fc = np.clip((sh[ok] - b[oi][ok]) / (og[ok] - b[oi][ok] + sh[ok] * (1 - og[ok])), 0.001, 0.999)
    mu, var = fc.mean(), fc.var()
    k = max(mu * (1 - mu) / max(var, 1e-6) - 1, 0.5)
    ab = (mu * k, (1 - mu) * k)
    xs = (np.arange(grid) + 0.5) / grid
    from scipy.stats import beta as beta_dist
    w = beta_dist.pdf(xs, *ab)
    lw = np.log(w / w.sum())

    rows_c, rows_b, rows_p, rows_n, rows_g, none_p = [], [], [], [], [], {}
    lprior = np.log((1 - prior_none) / H)
    nmap = dict(tot.iter_rows())
    for (c,), g in links.group_by("contig", maintain_order=True):
        n = nmap[c]
        nk = np.zeros(H)
        idx = np.array([bidx[x] for x in g["bin"]])
        nk[idx] = g["links"].to_numpy()
        Lc = L.get(c, 0)
        own_b = bidx.get(gmap.get(c), -1)
        gg = LH / np.maximum(LH + Lc, 1)
        if own_b >= 0:
            gg[own_b] = max(LH[own_b] - Lc, 0) / max(LH[own_b], 1)
        fx = xs[None, :] * gg[:, None]                                     # [H, grid]
        q = np.clip((fx + (1 - xs[None, :]) * b[:, None]) / (fx + 1 - xs[None, :]), 1e-12, 1 - 1e-12)
        ll = logsumexp(lw[None, :] + nk[:, None] * np.log(q) + (n - nk[:, None]) * np.log1p(-q), axis=1)
        llr = ll - nk * np.log(b) - (n - nk) * np.log1p(-b)
        lp = np.r_[llr + lprior, np.log(prior_none)]
        post = np.exp(lp - logsumexp(lp))
        rows_c += list(np.repeat(c, H)); rows_b += names; rows_p += list(post[:H]); rows_n += [n] * H
        rows_g += list(gg)
        none_p[c] = post[-1]
    post = (pl.DataFrame({"contig": rows_c, "bin": rows_b, "n": rows_n, "hic_p": rows_p, "genome_outside": rows_g})
            .join(links.select("contig", "bin", "links"), on=["contig", "bin"], how="left")
            .with_columns(pl.col("links").fill_null(0)))
    none = pl.DataFrame({"contig": list(none_p), "hic_p_none": list(none_p.values())})
    return post.select("contig", "bin", "links", "n", "hic_p", "genome_outside"), none, ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--links", required=True, help="contig_bin_links.tsv from hic/contacts.py")
    ap.add_argument("--contig-bin", required=True, help="the original contig -> bin table (to estimate f)")
    ap.add_argument("--lengths", required=True, help="contig<TAB>length with header (workflow contig_lengths.tsv)")
    ap.add_argument("--min-links", type=int, default=10)
    ap.add_argument("--prior-none", type=float, default=0.5)
    ap.add_argument("--min-outside", type=float, default=0.1,
                    help="score a contig and bin only when this share of the genome lies outside the contig")
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
    lengths = pl.read_csv(a.lengths, separator="\t", schema_overrides={"contig": pl.Utf8})
    post, none, (fa, fb) = hic_posterior(links, given, lengths, a.min_links, a.prior_none)
    print(f"Hi-C within-genome share f ~ Beta({fa:.2f}, {fb:.2f}), mean {fa / (fa + fb):.3f}; prior P(none) = {a.prior_none}")
    hic_map = (post.filter(pl.col("genome_outside") >= a.min_outside).sort(["contig", "hic_p"], descending=[False, True]).group_by("contig", maintain_order=True).first()
               .select("contig", pl.col("bin").alias("hic_map"), pl.col("hic_p").alias("hic_p_map"), pl.col("n").alias("hic_links")))
    base = d.join(hic_map, on="contig", how="inner").join(none, on="contig", how="left")

    def compare(name, rows, bin_col, p_expr):
        return (rows.with_columns(pl.col(bin_col).alias("target"), p_expr.alias("p"), pl.lit(name).alias("comparison"))
                .join(post.select("contig", pl.col("bin").alias("target"), pl.col("links").alias("hic_links_target"), "hic_p",
                                  "genome_outside"), on=["contig", "target"], how="left")
                .with_columns(pl.col("hic_links_target").fill_null(0), pl.col("hic_p").fill_null(0.0))
                .with_columns((pl.col("hic_links_target") / pl.col("hic_links")).alias("hic_share"))
                .select("comparison", "contig", "length", "n_sites", "status", "given_bin", "target", "p", "hic_p",
                        "hic_share", "hic_links", "hic_links_target", "hic_map", "hic_p_map", "hic_p_none", "genome_outside"))

    x = pl.concat([
        compare("own", base.filter((pl.col("given_bin") != "") & pl.col("p_contaminant").is_not_nan()),
                "given_bin", 1 - pl.col("p_contaminant")),
        compare("alt", base.filter(pl.col("best_alternative") != "unknown"), "best_alternative", pl.col("p_best_alternative")),
        compare("map", base.filter(pl.col("map") != "unknown"), "map", pl.col("p_map")),
    ]).filter(pl.col("hic_links") >= a.min_links)
    # Hi-C can only judge a bin when enough of the genome lies outside the contig to be contacted
    blind = x.filter(pl.col("genome_outside") < a.min_outside)
    print("left out (contig is >= {:.0%} of the bin's genome; Hi-C can't judge):".format(1 - a.min_outside),
          dict(blind.group_by("comparison").len().iter_rows()))
    x = x.filter(pl.col("genome_outside") >= a.min_outside)
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
    pl.DataFrame({"parameter": ["f_beta_a", "f_beta_b", "f_mean", "prior_none", "min_links", "contigs_with_hic"],
                  "value": [fa, fb, fa / (fa + fb), a.prior_none, a.min_links, base.height]}).write_csv(out / "params.tsv", separator="\t")
    with pl.Config(tbl_width_chars=220, tbl_rows=40, tbl_cols=20):
        print(corr)
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
