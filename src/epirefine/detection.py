"""Empirical-Bayes model of the modification caller, per motif group.

A motif site is either methylated (s=1) or not (s=0). Read counts at a site:

    k | n, s=0 ~ BetaBinomial(n, mu0, kappa0)   false-positive calls; mu0 = per-motif FP level
    k | n, s=1 ~ BetaBinomial(n, mu1, kappa1)   true calls; mu1 = per-motif detection ceiling

Parameters are fitted on *all* sites of the group in the sample (all bins, all contigs) as a
two-component mixture. Mod-type level parameters are fitted first by pooling groups of the same
mod type, then each group is fitted with a Gaussian prior (in transformed space) centred on its
mod-type fit. Groups that are methylated nowhere (or everywhere) therefore borrow the missing
component from the mod type.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit


def bb_logpmf(k, n, mu, kappa):
    """Beta-binomial log pmf without the binomial coefficient (constant across hypotheses)."""
    a, b = mu * kappa, (1.0 - mu) * kappa
    return betaln(k + a, n - k + b) - betaln(a, b)


@dataclass
class DetectionParams:
    mu0: float
    kappa0: float
    mu1: float
    kappa1: float
    pi1: float = 0.5
    n_sites: int = 0

    def to_theta(self):
        return np.array([logit(2 * self.mu0), np.log(self.kappa0), logit(2 * self.mu1 - 1), np.log(self.kappa1)])

    @staticmethod
    def from_theta(t, pi1=0.5, n_sites=0):
        return DetectionParams(0.5 * expit(t[0]), float(np.exp(t[1])), 0.5 + 0.5 * expit(t[2]),
                               float(np.exp(t[3])), float(pi1), int(n_sites))


DEFAULT_PRIOR = DetectionParams(mu0=0.05, kappa0=10.0, mu1=0.85, kappa1=10.0)
_BOUNDS = [(-12, 8), (np.log(0.3), np.log(1e4)), (-12, 12), (np.log(0.3), np.log(1e4)), (-12, 12)]


def fit_mixture(n, k, w, prior: DetectionParams, prior_sd: float) -> DetectionParams:
    """MAP fit of the two-component mixture.

    The likelihood can have a degenerate local optimum where the "methylated" component becomes a
    broad junk component (mu1 near 0.5, kappa1 < 1). L-BFGS is started from the prior centre and
    from clearly-methylated components, and the best objective wins; inputs are sorted so the
    result does not depend on row order.
    """
    n, k, w = (np.asarray(x, float) for x in (n, k, w))
    order = np.lexsort((k, n))
    n, k, w = n[order], k[order], w[order]
    centre = prior.to_theta()

    def nll(theta):
        p = DetectionParams.from_theta(theta[:4])
        lp1 = np.log(expit(theta[4]))
        lp0 = np.log(expit(-theta[4]))
        ll = np.logaddexp(lp1 + bb_logpmf(k, n, p.mu1, p.kappa1), lp0 + bb_logpmf(k, n, p.mu0, p.kappa0))
        return -(w * ll).sum() + 0.5 * (((theta[:4] - centre) / prior_sd) ** 2).sum()

    starts = [np.concatenate([centre, [logit(pi)]]) for pi in (0.1, 0.5, 0.9)]
    for mu1 in (0.8, 0.95):
        for pi in (0.1, 0.5):
            t = DetectionParams(prior.mu0, prior.kappa0, mu1, 30.0).to_theta()
            starts.append(np.concatenate([t, [logit(pi)]]))
    best = None
    for x0 in starts:
        x0 = np.clip(x0, [b[0] for b in _BOUNDS], [b[1] for b in _BOUNDS])
        res = minimize(nll, x0, method="L-BFGS-B", bounds=_BOUNDS)
        if best is None or res.fun < best.fun:
            best = res
    return DetectionParams.from_theta(best.x[:4], expit(best.x[4]), w.sum())


def _unique_counts(df: pl.DataFrame) -> pl.DataFrame:
    w = pl.col("w").sum() if "w" in df.columns else pl.len()
    return df.group_by("n", "k").agg(w.alias("w"))


def _fit_one(args):
    n, k, w, prior, sd = args
    return fit_mixture(n, k, w, prior, sd)


POOL_MIN_ROWS = 60_000  # count patterns over all jobs below which the fits run in this process


def _job_rows(job) -> int:
    """Count patterns (rows of n, k, w) a fit job holds, at any nesting of its arguments."""
    if isinstance(job, np.ndarray):
        return len(job) // 3 if job.ndim == 1 else 0
    return sum(_job_rows(x) for x in job) if isinstance(job, (tuple, list)) else 0


def _map(fn, jobs, workers):
    """fn over jobs, on `workers` processes when the work is worth starting them.

    Spawning the pool costs about 3 s; the fits of a 20-group monoculture set (10,000 count patterns)
    take 0.5 s in one process. Each fit depends on its own job only, so where it runs changes nothing.
    """
    if workers <= 1 or len(jobs) <= 1 or sum(_job_rows(j) for j in jobs) < POOL_MIN_ROWS:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from .runinfo import note_worker_peaks
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=get_context("spawn")) as ex:
        out = list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
        note_worker_peaks()
    return out


def fit_detection(sites: pl.DataFrame, group_modtype: dict[int, str],
                  type_prior_sd: float = 3.0, group_prior_sd: float = 1.0, workers: int = 1) -> dict[int, DetectionParams]:
    """`sites` has columns group, n, k: one row per site, or per count pattern with a count column w.

    Mod-type fits and then group fits are independent of each other and run on `workers` processes.
    """
    out: dict[int, DetectionParams] = {}
    sites = sites.with_columns(pl.col("group").replace_strict(group_modtype, return_dtype=pl.Utf8).alias("mt"))
    types, jobs = [], []
    for (mt,), df in sites.group_by("mt"):
        u = _unique_counts(df)
        types.append(mt)
        jobs.append((u["n"].to_numpy(), u["k"].to_numpy(), u["w"].to_numpy(), DEFAULT_PRIOR, type_prior_sd))
    type_fit = dict(zip(types, _map(_fit_one, jobs, workers)))
    w = pl.col("w").sum() if "w" in sites.columns else pl.len()
    per_group = {g: df for (g,), df in sites.group_by("group", "n", "k").agg(w.alias("w")).group_by("group")}
    groups, jobs = [], []
    for g, mt in group_modtype.items():
        if g not in per_group:
            out[g] = DetectionParams(**{**asdict(type_fit.get(mt, DEFAULT_PRIOR)), "n_sites": 0})
            continue
        u = per_group[g]
        groups.append(g)
        jobs.append((u["n"].to_numpy(), u["k"].to_numpy(), u["w"].to_numpy(), type_fit[mt], group_prior_sd))
    out.update(zip(groups, _map(_fit_one, jobs, workers)))
    return dict(sorted(out.items()))


def detection_table(params: dict[int, DetectionParams]) -> pl.DataFrame:
    return pl.DataFrame([{"group": g, **asdict(p)} for g, p in sorted(params.items())])


# ---------------------------------------------------------------------------------------------
# fit using the bins: methylation is a property of a genome, so a motif group is (nearly) all
# methylated in some bins and unmethylated in the rest


def _fit_bb(n, k, w, mu_bounds, prior: tuple[float, float] | None = None, prior_sd: float = 2.0):
    """MLE (or MAP with a normal prior on (logit mu, log kappa)) of one beta-binomial."""
    n, k, w = (np.asarray(x, float) for x in (n, k, w))
    lo, hi = mu_bounds

    def unpack(t):
        return lo + (hi - lo) * expit(t[0]), float(np.exp(t[1]))

    def nll(t):
        mu, kap = unpack(t)
        pen = 0.0 if prior is None else 0.5 * (((t - np.array(prior)) / prior_sd) ** 2).sum()
        return -(w * bb_logpmf(k, n, mu, kap)).sum() + pen

    m = float(np.clip((w * k).sum() / max((w * n).sum(), 1e-9), lo + 1e-4, hi - 1e-4))
    best = None
    for kap0 in (3.0, 30.0):
        t0 = np.array([logit((m - lo) / (hi - lo)), np.log(kap0)])
        res = minimize(nll, t0, method="L-BFGS-B", bounds=[(-12, 12), (np.log(0.3), np.log(1e4))])
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x)


def _fit_meth(n, k, w, mu0, kappa0, prior: tuple[float, float] | None = None):
    """Methylated component of sites from methylated bins: mixture with the fixed unmethylated
    component (bins can be partly methylated). Returns mu1, kappa1, weight."""
    n, k, w = (np.asarray(x, float) for x in (n, k, w))
    l0 = bb_logpmf(k, n, mu0, kappa0)

    def unpack(t):
        return 0.5 + 0.5 * expit(t[0]), float(np.exp(t[1])), expit(t[2])

    def nll(t):
        mu1, k1, p = unpack(t)
        pen = 0.0 if prior is None else 0.5 * (((t[:2] - np.array(prior)) / 2.0) ** 2).sum()
        return -(w * np.logaddexp(np.log(p) + bb_logpmf(k, n, mu1, k1), np.log1p(-p) + l0)).sum() + pen

    best = None
    for mu_init in (0.8, 0.95):
        t0 = np.array([logit(2 * mu_init - 1), np.log(20.0), logit(0.9)])
        res = minimize(nll, t0, method="L-BFGS-B", bounds=[(-12, 12), (np.log(0.3), np.log(1e4)), (-12, 12)])
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x)


def _theta(mu, kap, lo, hi):
    return (float(logit((mu - lo) / (hi - lo))), float(np.log(kap)))


def _group_fit(args):
    g, s0, s1, type_p0, type_p1 = args
    if s0 is not None:
        mu0, k0 = _fit_bb(*s0, (0.0, 0.5), prior=type_p0)
    else:
        mu0, k0 = 0.5 * expit(type_p0[0]), float(np.exp(type_p0[1]))
    if s1 is not None:
        mu1, k1, _ = _fit_meth(*s1, mu0, k0, prior=type_p1)
    else:
        mu1, k1 = 0.5 + 0.5 * expit(type_p1[0]), float(np.exp(type_p1[1]))
    return g, mu0, k0, mu1, k1


def fit_detection_by_bin(counts: pl.DataFrame, group_modtype: dict[int, str], unmethylated_max: float = 0.2,
                         methylated_min: float = 0.5, min_sites: int = 50, workers: int = 1) -> dict[int, DetectionParams]:
    """Caller model per motif group, fitted with the bins as labels.

    `counts`: columns group, bin, n, k, w (w = number of sites with that pattern in that bin).
    Per (group, bin) the methylated read fraction r = sum(k) / sum(n) is computed; bins with
    r < unmethylated_max train the unmethylated component (beta-binomial), bins with r >
    methylated_min train the methylated one (mixture with the unmethylated component fixed, so a
    partly methylated bin is allowed). Bins in between, or with < min_sites sites, are not used.
    Groups without labelled bins borrow the component from their mod type (pooled fit).

    Pooling all sites instead (fit_detection) fails when a motif is methylated in only a few of
    many bins: the methylated component then collapses to a broad junk component (mu1 = 0.5).
    """
    per_bin = counts.group_by("group", "bin").agg(
        (pl.col("k") * pl.col("w")).sum().alias("sk"), (pl.col("n") * pl.col("w")).sum().alias("sn"),
        pl.col("w").sum().alias("sites"))
    per_bin = per_bin.with_columns((pl.col("sk") / pl.col("sn")).alias("r")).with_columns(
        pl.when((pl.col("sites") >= min_sites) & (pl.col("r") < unmethylated_max)).then(pl.lit(0))
        .when((pl.col("sites") >= min_sites) & (pl.col("r") > methylated_min)).then(pl.lit(1))
        .otherwise(pl.lit(-1)).alias("label"))
    lab = counts.join(per_bin.select("group", "bin", "label"), on=["group", "bin"]).filter(pl.col("label") >= 0)
    lab = lab.with_columns(pl.col("group").replace_strict(group_modtype, return_dtype=pl.Utf8).alias("mt"))
    agg = lab.group_by("mt", "group", "label", "n", "k").agg(pl.col("w").sum())

    def arrays(df):
        df = df.sort("n", "k")
        return df["n"].to_numpy(), df["k"].to_numpy(), df["w"].to_numpy()

    # mod-type fits (priors and fallbacks), on sites pooled over the type's groups
    type_p0, type_p1 = {}, {}
    for mt in set(group_modtype.values()):
        t = agg.filter(pl.col("mt") == mt)
        z = t.filter(pl.col("label") == 0).group_by("n", "k").agg(pl.col("w").sum())
        mu0, k0 = _fit_bb(*arrays(z), (0.0, 0.5)) if z.height else (DEFAULT_PRIOR.mu0, DEFAULT_PRIOR.kappa0)
        o = t.filter(pl.col("label") == 1).group_by("n", "k").agg(pl.col("w").sum())
        mu1, k1, _ = _fit_meth(*arrays(o), mu0, k0) if o.height else (DEFAULT_PRIOR.mu1, DEFAULT_PRIOR.kappa1, 1.0)
        type_p0[mt], type_p1[mt] = _theta(mu0, k0, 0.0, 0.5), _theta(mu1, k1, 0.5, 1.0)

    # the rows of each (group, label) taken out in one pass (a filter of the whole table per group and label was
    # 1,420 scans on ZymoFecal)
    parts = agg.partition_by("group", "label", as_dict=True)
    jobs = []
    for g, mt in group_modtype.items():
        z, o = parts.get((g, 0)), parts.get((g, 1))
        jobs.append((g, arrays(z) if z is not None and z.height else None, arrays(o) if o is not None and o.height else None,
                     type_p0[mt], type_p1[mt]))
    fits = _map(_group_fit, jobs, workers)
    n_sites = dict(counts.group_by("group").agg(pl.col("w").sum()).iter_rows())
    meth_frac = dict(lab.group_by("group").agg(
        (pl.col("w") * (pl.col("label") == 1)).sum() / pl.col("w").sum()).iter_rows())
    return {g: DetectionParams(mu0, k0, mu1, k1, float(meth_frac.get(g, 0.0)), int(n_sites.get(g, 0)))
            for g, mu0, k0, mu1, k1 in sorted(fits)}
