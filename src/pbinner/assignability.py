"""Assignability: how well *could* a unit be assigned, given only where its sites are and their coverage?

Uses the "Asimov dataset": the observed read counts are replaced by their expectation under a
hypothesised true host, and the resulting log Bayes factors are expected log BFs (KL divergences
between predictive distributions). Because the expected per-site log-likelihood is linear in phi,

    E_phi[ log P(k_i | n_i, x_g) ] = phi * E1(n_i, g) + (1 - phi) * E0(n_i, g)

so per unit only two tables H1[u, m, g] and H0[u, m, g] are needed, and any host can be plugged in.

Two numbers per unit:
  * expected log BF of the MAP host over its nearest competitor (bin), and over "unknown".
  * detectability: max over bins h of the expected log BF of h vs unknown if h were the host.
    A unit called "unknown" with high detectability means "a binned host would have been
    recognised, so the host is probably not binned"; low detectability means "no information".
    With many bins the maximum is taken over a shortlist of bins (`top`) that are most distinct
    from the community at the unit's informative groups (cheap KL screen); the result is then a
    lower bound on the full maximum, exact whenever the best bin is on the shortlist.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import gammaln

from .detection import DetectionParams, bb_logpmf
from .grid import Grid
from concurrent.futures import ThreadPoolExecutor

from scipy.special import logsumexp

from .model import FittedModel, _n_threads, score_units


def expected_tables(sites: pl.DataFrame, n_units: int, n_groups: int, det: dict[int, DetectionParams],
                    grid: Grid, n_cap: int = 500, dtype=np.float64):
    """sites: integer columns unit, group, n (optional count w). Returns H1, H0 with shape [U, M, G]."""
    w = pl.col("w").sum() if "w" in sites.columns else pl.len()
    s = (sites.with_columns(pl.col("n").clip(upper_bound=n_cap))
         .group_by("unit", "group", "n").agg(w.alias("w")))
    table = {}
    for g, n in s.select("group", "n").unique().iter_rows():
        p = det[g]
        k = np.arange(n + 1, dtype=float)
        logc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        l1, l0 = bb_logpmf(k, n, p.mu1, p.kappa1), bb_logpmf(k, n, p.mu0, p.kappa0)
        p1, p0 = np.exp(logc + l1), np.exp(logc + l0)
        term = np.logaddexp(l1[:, None] + grid.logx, l0[:, None] + grid.log1mx)
        table[(g, n)] = (p1 / p1.sum()) @ term, (p0 / p0.sum()) @ term
    H1 = np.zeros((n_units, n_groups, grid.size), dtype)
    H0 = np.zeros_like(H1)
    if s.height == 0:
        return H1, H0
    kid = {k: i for i, k in enumerate(table)}
    T1 = np.stack([v[0] for v in table.values()]); T0 = np.stack([v[1] for v in table.values()])
    s = s.with_columns((pl.col("unit").cast(pl.Int64) * n_groups + pl.col("group")).alias("flat")).sort("flat")
    flat, w = s["flat"].to_numpy(), s["w"].to_numpy().astype(float)
    r = np.array([kid[k] for k in zip(s["group"].to_list(), s["n"].to_list())])
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    for T, out in ((T1, H1), (T0, H0)):
        out.reshape(-1, grid.size)[flat[starts]] = np.add.reduceat(T[r] * w[:, None], starts, axis=0)
    return H1, H0


def asimov(H1, H0, phi):
    """phi: [U, M] true methylation level per unit and group."""
    return phi[..., None] * H1 + (1 - phi[..., None]) * H0


def phi_means(model: FittedModel) -> np.ndarray:
    p = np.exp(model.log_prof)
    return (p * model.grid.x).sum(-1)  # [H, M]


def margin_vs_competitors(model: FittedModel, H1, H0, truth: np.ndarray, plasmid: bool, temperature: float,
                          batch: int = 2000):
    """Expected log BF of the true bin over (a) its nearest competing bin, (b) unknown."""
    U, H = len(truth), model.n_bins
    vs_bin, nearest, vs_unk = np.full(U, np.inf), np.zeros(U, int), np.zeros(U)
    phi_h = phi_means(model) * (model.delta if plasmid else 1.0)
    for s in range(0, U, batch):
        sl = slice(s, s + batch)
        t = truth[sl]
        Lp = model.marginal(asimov(H1[sl], H0[sl], phi_h[t]), plasmid)
        S = score_units(Lp, model.log_prof)
        Su = score_units(Lp, model.log_pi[None])[:, 0]
        idx = np.arange(len(t))
        own = S[idx, t]
        vs_unk[sl] = (own - Su) / temperature
        if H > 1:
            S[idx, t] = -np.inf
            nearest[sl] = S.argmax(1)
            vs_bin[sl] = (own - S[idx, nearest[sl]]) / temperature
    return vs_bin, nearest, vs_unk


def _bern_kl(p, q):
    p, q = np.clip(p, 1e-6, 1 - 1e-6), np.clip(q, 1e-6, 1 - 1e-6)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def detectability(model: FittedModel, H1, H0, plasmid: bool, temperature: float,
                  n_sites_group: np.ndarray | None = None, top: int = 20, batch: int = 250):
    """max_h expected log BF(h vs unknown) if bin h were the true host, and the argmax.

    With more than `top` bins and site counts given, h ranges over a per-unit shortlist: the bins
    with the largest sum over groups of min(sites, 30) x KL(Bernoulli(phi_hm) || Bernoulli(mean of pi_m)).
    """
    d = model.delta if plasmid else 1.0
    phi_all = phi_means(model) * d
    U, H = H1.shape[0], phi_all.shape[0]
    if n_sites_group is not None and H > top:
        pbar = (np.exp(model.log_pi) * model.grid.x).sum(-1) * d
        screen = np.minimum(n_sites_group, 30).astype(float) @ _bern_kl(phi_all, pbar[None]).T
        cand = np.argpartition(-screen, top - 1, axis=1)[:, :top]
    else:
        cand = np.broadcast_to(np.arange(H), (U, H))
    best, arg = np.full(U, -np.inf), np.zeros(U, int)

    def work(sl):  # batches are independent; numpy releases the GIL, so they run on threads
        for j in range(cand.shape[1]):
            h = cand[sl, j]
            Lp = model.marginal(asimov(H1[sl], H0[sl], phi_all[h]), plasmid).astype(np.float64)
            s_h = logsumexp(Lp + model.log_prof[h], -1).sum(-1)
            s_u = logsumexp(Lp + model.log_pi[None], -1).sum(-1)
            bf = (s_h - s_u) / temperature
            better = bf > best[sl]
            best[sl] = np.where(better, bf, best[sl])
            arg[sl] = np.where(better, h, arg[sl])

    slices = [slice(s, s + batch) for s in range(0, U, batch)]
    with ThreadPoolExecutor(min(len(slices), _n_threads()) or 1) as ex:
        list(ex.map(work, slices))
    return best, arg
