"""Trivial baselines for bin refinement: what a few lines of arithmetic achieve without EpiRefine's model.

They answer the referee's question "would a 20-line script do as well?". All read the same inputs as a run (the
binning, and the run's site cache for sequence and read counts) and give every contig a score, so they can be
compared as precision-recall curves and at EpiRefine's own precision (benchmarks/evidence_metrics.py).

Contamination (a score per binned contig; higher = more likely foreign to its bin):
  coverage      robust z of log coverage against the bin (median, MAD, plus the contig's own counting noise
                1 / reads, so that a short contig with a handful of reads is not an outlier by chance);
                coverage is the mean read count at the contig's motif sites in the site cache
  tetra         distance of the contig's tetranucleotide frequencies (136 strand-symmetric 4-mers) to the bin's
                length-weighted centroid without the contig, as a robust z among bin contigs of similar length
                (short contigs have noisy frequencies, so distances are compared within length classes)
  gc            the same for |GC - bin GC|
  cov_tetra     coverage or tetra: sqrt(z_cov^2 + max(z_tetra, 0)^2)
  cov_and_tetra coverage and tetra: min(z_cov, z_tetra), an outlier in both (as RefineM's strictest setting asks)
  naive_meth    "the contig lacks a motif the rest of its bin has methylated": over motif groups with at least
                `min_sites` sites on the contig and a pooled methylated read fraction >= 0.5 in the rest of the
                bin, 1 - the contig's lowest methylated fraction (so the rule "fraction < x" is score > 1 - x)
  naive_meth2   two-sided: the largest |bin fraction - contig fraction| over groups with >= min_sites sites

Inclusion (a bin and a score per unbinned contig):
  naive_meth    the bin whose methylated fractions the contig matches best (smallest worst-group difference);
                score = how much worse the second-best bin is. `rule_bin` is the plain rule "the unique bin whose
                methylated / unmethylated motif set equals the contig's" ("" when none or several)
  cov_tetra     the bin nearest in coverage and tetranucleotide z; score = -distance

    python benchmarks/trivial_baselines.py --cache <site cache> --truth truth_actions.tsv --out scores.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from epirefine.sites import EncodedAssembly

LENGTH_EDGES = np.array([5_000, 10_000, 20_000, 50_000])


def _canonical_map() -> np.ndarray:
    """4-mer index (base 4) -> index of its strand-symmetric class (136 classes)."""
    idx = np.arange(256)
    digits = np.stack([(idx >> s) & 3 for s in (6, 4, 2, 0)], 1)
    rc = ((3 - digits[:, ::-1]) * np.array([64, 16, 4, 1])).sum(1)
    canon = np.minimum(idx, rc)
    return np.unique(canon, return_inverse=True)[1]


def composition(asm: EncodedAssembly) -> tuple[np.ndarray, np.ndarray]:
    """Tetranucleotide frequencies [contigs, 136] and GC share per contig, from the encoded assembly."""
    cmap = _canonical_map()
    tnf = np.zeros((len(asm.names), 136))
    gc = np.zeros(len(asm.names))
    for i in range(len(asm.names)):
        s = np.asarray(asm.seq(i)).astype(np.int64)
        ok = s < 4
        gc[i] = ((s == 1) | (s == 2)).sum() / max(ok.sum(), 1)
        if len(s) < 4:
            continue
        k = s[:-3] * 64 + s[1:-2] * 16 + s[2:-1] * 4 + s[3:]
        k = k[ok[:-3] & ok[1:-2] & ok[2:-1] & ok[3:]]
        c = np.bincount(cmap[k], minlength=136)
        tnf[i] = c / max(c.sum(), 1)
    return tnf, gc


def site_tables(parts: list[str], n_contigs: int) -> dict:
    """Per contig and motif group: methylated reads K, reads N, sites S; per contig mean and median reads per site."""
    lf = pl.scan_parquet(parts)
    g = lf.group_by("contig", "group").agg(pl.col("k").sum(), pl.col("n").sum(), pl.len().alias("s")).collect()
    n_groups = int(g["group"].max()) + 1 if g.height else 0
    K, N, S = (np.zeros((n_contigs, n_groups)) for _ in range(3))
    ci, gi = g["contig"].to_numpy(), g["group"].to_numpy()
    K[ci, gi], N[ci, gi], S[ci, gi] = g["k"].to_numpy(), g["n"].to_numpy(), g["s"].to_numpy()
    c = lf.group_by("contig").agg(pl.col("n").mean().alias("mean"), pl.col("n").median().alias("median")).collect()
    mean, median = np.full(n_contigs, np.nan), np.zeros(n_contigs)
    mean[c["contig"].to_numpy()] = c["mean"].to_numpy()
    median[c["contig"].to_numpy()] = c["median"].to_numpy()   # the pipeline's coverage (0 without sites)
    return {"K": K, "N": N, "S": S, "cov_mean": mean, "cov_median": median}


def load_features(cache: str | Path) -> dict:
    """Everything the baselines need, from a run's site cache (encoded assembly + site read counts)."""
    cache = Path(cache)
    asm = EncodedAssembly(cache / "asm")
    man = json.loads((cache / "manifest.json").read_text())
    parts = [str(cache / "sites" / Path(f).name) for f in man["parts"]]
    tnf, gc = composition(asm)
    return {"names": list(asm.names), "length": np.asarray(asm.lengths), "tnf": tnf, "gc": gc,
            **site_tables(parts, len(asm.names))}


def _robust_z(x: np.ndarray, ref: np.ndarray, floor: float, extra_var=0.0) -> np.ndarray:
    """(x - median(ref)) / spread; spread = MAD-based sd of ref (at least floor), plus a per-item variance."""
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        return np.full(len(x), np.nan)
    med = np.median(ref)
    return (x - med) / np.sqrt(max(1.4826 * np.median(np.abs(ref - med)), floor) ** 2 + extra_var)


def _z_by_length(x, length, members, targets, floor, min_ref=8):
    """Robust z of x[targets] against x[members] of the same length class (the whole bin when a class is small)."""
    cls = np.searchsorted(LENGTH_EDGES, length, side="right")
    out = np.full(len(targets), np.nan)
    for k in np.unique(cls[targets]):
        ref = members[cls[members] == k]
        ref = ref if len(ref) >= min_ref else members
        sel = cls[targets] == k
        out[sel] = _robust_z(x[targets[sel]], x[ref], floor)
    return out


def contamination_scores(f: dict, given_bin: dict, min_sites: int = 10) -> pl.DataFrame:
    """One row per binned contig with every baseline's contamination score (nan-free: no evidence = 0)."""
    names, L = f["names"], f["length"].astype(float)
    bins = sorted(set(given_bin.values()))
    b_of = np.array([bins.index(given_bin[c]) if c in given_bin else -1 for c in names])
    logcov = np.log(np.where(f["cov_mean"] > 0, f["cov_mean"], np.nan))
    frac = np.divide(f["K"], f["N"], out=np.full(f["K"].shape, np.nan), where=f["N"] > 0)
    noise = 1 / np.maximum(f["N"].sum(1), 1)          # variance of the log of a Poisson mean: 1 / reads
    cols = {k: np.zeros(len(names)) for k in ("coverage", "tetra", "gc", "cov_tetra", "cov_and_tetra", "naive_meth", "naive_meth2")}
    for b in range(len(bins)):
        m = np.nonzero(b_of == b)[0]
        if len(m) < 2:
            continue
        w = L[m][:, None]
        tot, W = (w * f["tnf"][m]).sum(0), w.sum()
        centroid = (tot - w * f["tnf"][m]) / (W - w)                      # the bin without the contig
        d = np.full(len(names), np.nan)
        d[m] = np.log(np.linalg.norm(f["tnf"][m] - centroid, axis=1) + 1e-6)
        dg = np.full(len(names), np.nan)
        gc_rest = ((L[m] * f["gc"][m]).sum() - L[m] * f["gc"][m]) / (W - L[m])
        dg[m] = np.abs(f["gc"][m] - gc_rest)
        z_cov = np.abs(np.nan_to_num(_robust_z(logcov[m], logcov[m], 0.05, noise[m])))
        z_tet = np.nan_to_num(_z_by_length(d, L, m, m, 0.05))
        cols["coverage"][m] = z_cov
        cols["tetra"][m] = z_tet
        cols["gc"][m] = np.nan_to_num(_z_by_length(dg, L, m, m, 0.002))
        cols["cov_tetra"][m] = np.sqrt(z_cov ** 2 + np.maximum(z_tet, 0) ** 2)
        cols["cov_and_tetra"][m] = np.minimum(z_cov, np.maximum(z_tet, 0))
        Kb, Nb = f["K"][m].sum(0), f["N"][m].sum(0)
        rest = np.divide(Kb - f["K"][m], Nb - f["N"][m], out=np.full(f["K"][m].shape, np.nan), where=(Nb - f["N"][m]) > 0)
        enough = f["S"][m] >= min_sites
        lacks = np.where(enough & (rest >= 0.5), 1 - frac[m], 0.0)
        cols["naive_meth"][m] = np.nan_to_num(lacks).max(1) if lacks.shape[1] else 0.0
        diff = np.where(enough, np.abs(rest - frac[m]), 0.0)
        cols["naive_meth2"][m] = np.nan_to_num(diff).max(1) if diff.shape[1] else 0.0
    keep = b_of >= 0
    return pl.DataFrame({"contig": [c for c, k in zip(names, keep) if k], **{k: v[keep] for k, v in cols.items()}})


def inclusion_scores(f: dict, given_bin: dict, min_sites: int = 10) -> pl.DataFrame:
    """One row per unbinned contig and baseline: the bin it would be given and the score of that choice."""
    names, L = f["names"], f["length"].astype(float)
    bins = sorted(set(given_bin.values()))
    b_of = np.array([bins.index(given_bin[c]) if c in given_bin else -1 for c in names])
    u = np.nonzero(b_of < 0)[0]
    logcov = np.log(np.where(f["cov_mean"] > 0, f["cov_mean"], np.nan))
    frac = np.divide(f["K"], f["N"], out=np.full(f["K"].shape, np.nan), where=f["N"] > 0)
    noise = 1 / np.maximum(f["N"].sum(1), 1)
    D = np.full((len(u), len(bins)), np.inf)          # coverage + tetranucleotide distance to each bin
    M = np.full((len(u), len(bins)), np.inf)          # worst-group methylation difference to each bin
    same = np.zeros((len(u), len(bins)), bool)        # same methylated / unmethylated set as the bin
    enough = f["S"][u] >= min_sites
    for b in range(len(bins)):
        m = np.nonzero(b_of == b)[0]
        if len(m) < 2:
            continue
        w = L[m][:, None]
        tot, W = (w * f["tnf"][m]).sum(0), w.sum()
        d = np.full(len(names), np.nan)
        d[m] = np.log(np.linalg.norm(f["tnf"][m] - (tot - w * f["tnf"][m]) / (W - w), axis=1) + 1e-6)
        d[u] = np.log(np.linalg.norm(f["tnf"][u] - tot / W, axis=1) + 1e-6)
        z_tet = np.maximum(_z_by_length(d, L, m, u, 0.05), 0)
        z_cov = _robust_z(logcov[u], logcov[m], 0.05, noise[u])
        D[:, b] = np.nan_to_num(np.sqrt(z_cov ** 2 + z_tet ** 2), nan=np.inf)
        Nb = f["N"][m].sum(0)
        level = np.divide(f["K"][m].sum(0), Nb, out=np.full(len(Nb), np.nan), where=Nb > 0)
        use = enough & np.isfinite(level)[None, :]
        diff = np.where(use, np.abs(frac[u] - level[None, :]), 0.0)
        any_use = use.any(1)
        M[:, b] = np.where(any_use, np.nan_to_num(diff).max(1) if diff.shape[1] else 0.0, np.inf)
        agree = np.where(use, (frac[u] >= 0.5) == (level[None, :] >= 0.5), True)
        same[:, b] = any_use & agree.all(1)
    rows = []
    for name, dist in (("cov_tetra", D), ("naive_meth", M)):
        order = np.argsort(dist, axis=1)
        best = dist[np.arange(len(u)), order[:, 0]]
        second = dist[np.arange(len(u)), order[:, 1]] if len(bins) > 1 else np.full(len(u), np.inf)
        if name == "cov_tetra":
            score = -best
        else:
            score = np.where(np.isfinite(best), np.minimum(second, 1.0) - best, -np.inf)
        rule = [bins[int(np.argmax(s))] if s.sum() == 1 else "" for s in same] if name == "naive_meth" else [""] * len(u)
        rows.append(pl.DataFrame({"contig": [names[i] for i in u], "baseline": name,
                                  "bin": [bins[j] if np.isfinite(x) else "" for j, x in zip(order[:, 0], best)],
                                  "score": np.where(np.isfinite(best), score, -np.inf), "rule_bin": rule}))
    return pl.concat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="site cache of an epirefine run on these inputs")
    ap.add_argument("--truth", help="truth_actions.tsv (its given_bin is the binning)")
    ap.add_argument("--contig-bin", help="contig<TAB>bin without header (instead of --truth)")
    ap.add_argument("--min-sites", type=int, default=10)
    ap.add_argument("--out", required=True, help="prefix: writes <out>.contamination.tsv and <out>.inclusion.tsv")
    a = ap.parse_args()
    if a.truth:
        t = pl.read_csv(a.truth, separator="\t").with_columns(pl.col("given_bin").fill_null(""))
        given = {c: b for c, b in zip(t["contig"], t["given_bin"]) if b}
    else:
        cb = pl.read_csv(a.contig_bin, separator="\t", has_header=False, infer_schema_length=0)
        given = dict(zip(cb[cb.columns[0]], cb[cb.columns[1]]))
    f = load_features(a.cache)
    contamination_scores(f, given, a.min_sites).write_csv(a.out + ".contamination.tsv", separator="\t", float_precision=4)
    inclusion_scores(f, given, a.min_sites).write_csv(a.out + ".inclusion.tsv", separator="\t", float_precision=4)


if __name__ == "__main__":
    main()
