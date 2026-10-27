"""Expected evidence: how well *could* a unit be placed, given only where its sites are and their coverage?

Uses the "Asimov dataset": the observed read counts are replaced by their expectation under a
hypothesised true bin, and the resulting log Bayes factors are expected log BFs (KL divergences
between predictive distributions). Because the expected per-site log-likelihood is linear in phi,

    E_phi[ log P(k_i | n_i, x_g) ] = phi * E1(n_i, g) + (1 - phi) * E0(n_i, g)

so per unit only two tables H1[u, m, g] and H0[u, m, g] are needed, and any bin can be plugged in.
Calibration uses it to decide which pseudo-contigs could tell their bin from its nearest competitor
(dev log D12).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import gammaln

from .detection import DetectionParams, bb_logpmf
from .grid import Grid
from scipy.special import logsumexp

from .model import FittedModel, score_units


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


def margin_vs_competitors(model: FittedModel, H1, H0, truth: np.ndarray, temperature: float,
                          batch: int = 2000):
    """Expected log BF of the true bin over (a) its nearest competing bin, (b) unknown."""
    U, H = len(truth), model.n_bins
    vs_bin, nearest, vs_unk = np.full(U, np.inf), np.zeros(U, int), np.zeros(U)
    phi_h = phi_means(model)
    for s in range(0, U, batch):
        sl = slice(s, s + batch)
        t = truth[sl]
        Lp = model.marginal(asimov(H1[sl], H0[sl], phi_h[t]))
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


def contig_evidence(parts: list[str], unit_of: dict[int, int], truth: np.ndarray, model: FittedModel,
                    det: dict[int, DetectionParams], grid: Grid, temperature: float, min_cov: int = 1,
                    workers: int = 8):
    """Expected methylation evidence and coverage per target contig.

    Evidence: the expected log BF (tempered) that the contig's sites would give for bin truth[u] over a
    random genome of the community, had the contig come from that bin (assignability, D12). Low
    values mean its sites can't show where it belongs, whatever their reads say: too few sites, too
    few reads, or a bin without a distinctive methylome. Coverage: median reads per motif site (all
    sites, before min_cov).

    parts: site-cache parquet files (contig, group, pos, n, k); unit_of: cache contig id -> target
    index. Parts are processed on threads; contigs split across parts are evaluated together at the
    end. Returns (vs_unknown [U], vs_nearest_bin [U], coverage [U]); NaN where truth < 0 or no sites.
    """
    from concurrent.futures import ThreadPoolExecutor

    U = len(truth)
    vs_unk, vs_bin, cov = np.full(U, np.nan), np.full(U, np.nan), np.full(U, np.nan)
    want = pl.Series(list(unit_of), dtype=pl.Int32)
    seen: dict[int, int] = {}
    for f in parts:
        for c in pl.read_parquet(f, columns=["contig"])["contig"].unique().to_list():
            seen[c] = seen.get(c, 0) + 1
    split = pl.Series(sorted(c for c, k in seen.items() if k > 1), dtype=pl.Int32)

    def evaluate(sites: pl.DataFrame):
        sites = sites.filter(pl.col("contig").is_in(want))
        if not sites.height:
            return
        med = sites.group_by("contig").agg(pl.col("n").median())
        for c, m in med.iter_rows():
            cov[unit_of[c]] = m
        sites = sites.filter(pl.col("n") >= min_cov)
        contigs = [c for c in sites["contig"].unique().sort().to_list() if truth[unit_of[c]] >= 0]
        if not contigs:
            return
        units = np.array([unit_of[c] for c in contigs])
        inv = {c: i for i, c in enumerate(contigs)}
        s = (sites.filter(pl.col("contig").is_in(contigs))
             .select(pl.col("contig").replace_strict(inv, return_dtype=pl.Int64).alias("unit"), "group", "n"))
        H1, H0 = expected_tables(s, len(units), model.log_prof.shape[1], det, grid, dtype=np.float32)
        b, _, u = margin_vs_competitors(model, H1, H0, truth[units], temperature)
        vs_bin[units], vs_unk[units] = b, u   # each contig is evaluated in exactly one call

    def one(f):
        d = pl.read_parquet(f, columns=["contig", "group", "n"])
        if split.len():
            held = d.filter(pl.col("contig").is_in(split))
            d = d.filter(~pl.col("contig").is_in(split))
        else:
            held = None
        evaluate(d)
        return held

    with ThreadPoolExecutor(max(1, workers)) as ex:
        held = [h for h in ex.map(one, parts) if h is not None and h.height]
    if held:
        evaluate(pl.concat(held))
    return vs_unk, vs_bin, cov
