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


def _map(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=get_context("spawn")) as ex:
        return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


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
