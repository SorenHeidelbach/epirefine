"""Hierarchical model linking read counts -> sites -> unit -> bin -> population.

    k_i | n_i, s_i          ~ BetaBinomial (caller model, see detection.py)
    s_i | phi_u             ~ Bernoulli(phi_u)              site i in unit u methylated?
    phi_um | h              ~ kernel around phi_hm          between-unit variation (Beta(tau) or
                                                            logit-normal(sigma), see grid.py)
    phi_hm                  ~ pi_m                          population prior per motif group (NPMLE)
    host chunk k            ~ (1 - eps) bin + eps foreign   robust bin profiles
    unbinned genome         ~ community draw (lam) or a relative of some bin (1 - lam), whose level
                              in each group comes from that bin with prob 1 - eta

A "unit" is any stretch of sequence scored as one entity: a contig, a chunk of a binned contig, or a
pseudo-contig fragment used for calibration. Everything is evaluated on a discrete phi grid.

Arrays:
    L[u, m, g]   log P(sites of unit u, group m | phi_u = x_g)
    Lp[u, m, g]  log P(sites of unit u, group m | phi_h = x_g)    (phi_u integrated out)
    ell[h, m, g] weighted sum of Lp over the host chunks of bin h
    A[u, h, m]   log P(sites of u in group m | bin h)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.special import expit, logsumexp

from .detection import DetectionParams, bb_logpmf
from .grid import Grid


def site_terms(keys: pl.DataFrame, det: dict[int, DetectionParams], grid: Grid, block: int = 65_536) -> np.ndarray:
    """log P(k | n, phi = x_g) for each row (group, n, k) of `keys`: one row per distinct count pattern.

    The [P, G] table is filled in row blocks: in one expression the two broadcast sums are held next
    to the result, three times the table (10.9 GB instead of 3.6 GB for 7.1 million patterns).
    """
    grp, n, k = (keys[c].to_numpy() for c in ("group", "n", "k"))
    n, k = n.astype(float), k.astype(float)
    p = [det[g] for g in range(max(det) + 1)]
    mu0, k0, mu1, k1 = (np.array([getattr(q, f) for q in p])[grp] for f in ("mu0", "kappa0", "mu1", "kappa1"))
    la, lb = bb_logpmf(k, n, mu1, k1), bb_logpmf(k, n, mu0, k0)
    out = np.empty((len(la), grid.size))
    for s in range(0, len(la), block):
        e = s + block
        np.logaddexp(la[s:e, None] + grid.logx[None, :], lb[s:e, None] + grid.log1mx[None, :], out=out[s:e])
    return out


def unit_loglik(sites: pl.DataFrame, n_units: int, n_groups: int, det: dict[int, DetectionParams],
                grid: Grid, chunk_rows: int = 200_000, dtype=np.float64) -> np.ndarray:
    """L[u, m, g] from a site table with integer columns unit, group, n, k (and optional count w).

    The per-site term depends only on (group, n, k), so it is computed once per distinct pattern
    and gathered, instead of evaluating transcendental functions for every site.
    """
    w = pl.col("w").sum() if "w" in sites.columns else pl.len()
    agg = (sites.group_by("unit", "group", "n", "k").agg(w.alias("w"))
           .with_columns((pl.col("unit").cast(pl.Int64) * n_groups + pl.col("group")).alias("flat")))
    out = np.zeros((n_units * n_groups, grid.size), dtype)
    if agg.height == 0:
        return out.reshape(n_units, n_groups, grid.size)
    keys = agg.select("group", "n", "k").unique().with_row_index("r")
    table = site_terms(keys, det, grid)
    agg = agg.join(keys, on=["group", "n", "k"]).sort("flat")
    for start in range(0, agg.height, chunk_rows):
        c = agg.slice(start, chunk_rows)
        r, wt, flat = c["r"].to_numpy(), c["w"].to_numpy().astype(float), c["flat"].to_numpy()
        term = table[r] * wt[:, None]
        starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
        out[flat[starts]] += np.add.reduceat(term, starts, axis=0)
    return out.reshape(n_units, n_groups, grid.size)


def marginalize_unit(L: np.ndarray, T, batch: int = 2048) -> np.ndarray:
    """Lp[..., i] = log sum_j T[i, j] exp(L[..., j]); keeps the dtype of L, works in unit batches.

    T is one [G, G] matrix for all groups, or a list of (group indices, [G, G] matrix) pairs when
    groups have different between-unit variation (per-group tau).
    """
    out = np.empty_like(L)
    parts = [(slice(None), T)] if isinstance(T, np.ndarray) else T
    tiny = np.finfo(L.dtype).tiny
    for ms, Tm in parts:
        Tt = Tm.T.astype(L.dtype)
        for s in range(0, L.shape[0], batch):
            x = L[s:s + batch, ms]
            mx = x.max(-1, keepdims=True)
            q = np.exp(x - mx) @ Tt
            out[s:s + batch, ms] = np.log(np.maximum(q, tiny)) + mx
    return out


def transition_spec(grid: Grid, tau, delta: float = 1.0):
    """One transition matrix for a scalar tau, or (groups, matrix) pairs for a per-group tau array."""
    if np.ndim(tau) == 0:
        return grid.transition(tau, delta)
    tau = np.asarray(tau, dtype=float)
    return [(np.flatnonzero(tau == t), grid.transition(float(t), delta)) for t in np.unique(tau)]


def bin_loglik(Lp_chunks: np.ndarray, chunk_bin: np.ndarray, n_bins: int, w: np.ndarray | None = None) -> np.ndarray:
    """ell[h] = sum of w * Lp over the chunks of bin h (float64)."""
    ell = np.zeros((n_bins,) + Lp_chunks.shape[1:])
    w = np.ones(len(chunk_bin)) if w is None else w
    order = np.argsort(chunk_bin, kind="stable")
    bounds = np.searchsorted(chunk_bin[order], np.arange(n_bins + 1))
    for h in range(n_bins):
        idx = order[bounds[h]:bounds[h + 1]]
        if len(idx):
            ell[h] = np.tensordot(w[idx], Lp_chunks[idx].astype(np.float64), 1)
    return ell


def npmle(ell: np.ndarray, iters: int = 300, tol: float = 1e-6, init: np.ndarray | None = None) -> np.ndarray:
    """Grid NPMLE of pi[m, g] from ell[b, m, g] by EM, all groups at once.

    Stops when no cell moves by more than `tol`; `init` warm-starts from an earlier solution.
    """
    B, M, G = ell.shape
    lik = np.exp(ell - ell.max(-1, keepdims=True)).transpose(1, 0, 2).copy()  # M, B, G
    pi = np.full((M, G), 1.0 / G) if init is None else init.copy()
    for _ in range(iters):
        denom = np.maximum(np.matmul(lik, pi[:, :, None])[:, :, 0], 1e-300)  # [M, B]
        new = np.maximum(pi * np.matmul((1.0 / denom)[:, None, :], lik)[:, 0, :] / B, 1e-300)
        delta = np.abs(new - pi).max()
        pi = new
        if delta < tol:
            break
    return pi


def fit_population_prior(ell: np.ndarray, group_modtype: list[str], lam: float = 5.0,
                         floor: float = 0.02, iters: int = 300, init: np.ndarray | None = None,
                         return_raw: bool = False):
    """Nonparametric (grid) prior on host phi per group, by EM over bins (Efron-style g-modelling).

    The per-group NPMLE is shrunk towards the mean over groups of the same mod type with `lam`
    pseudo-bins, then mixed with a `floor` of uniform-over-grid mass so no phi is impossible.
    Returns log pi[m, g] (and the raw NPMLE, for warm starts, if return_raw).
    """
    B, M, G = ell.shape
    pis = npmle(ell, iters, init=init)
    types = np.array(group_modtype)
    out = np.empty_like(pis)
    for t in np.unique(types):
        sel = types == t
        shrunk = (B * pis[sel] + lam * pis[sel].mean(0)) / (B + lam)
        out[sel] = (1 - floor) * shrunk + floor / G
    return (np.log(out), pis) if return_raw else np.log(out)


def host_profiles(ell: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """log posterior over phi_h per (bin, group)."""
    lp = ell + log_pi[None, :, :]
    return lp - logsumexp(lp, axis=-1, keepdims=True)


def group_scores(Lp: np.ndarray, log_prof: np.ndarray) -> np.ndarray:
    """A[u, h, m] = log sum_g exp(Lp[u, m, g] + log_prof[h, m, g])."""
    U, M, G = Lp.shape
    A = np.empty((U, log_prof.shape[0], M))
    for m in range(M):
        A[:, :, m] = _group_term(Lp[:, m, :], log_prof[:, m, :])
    return A


def _group_term(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    am = a.max(1, keepdims=True)
    bm = b.max(1, keepdims=True)
    q = np.exp(a - am).astype(np.float64) @ np.exp(b - bm).T
    return np.log(np.maximum(q, 1e-300)) + am + bm.T


def _n_threads():
    return min(16, os.cpu_count() or 1)


def summed_scores(Lp: np.ndarray, log_prof: np.ndarray, rel: tuple | None = None):
    """S[u, h] = sum_m log P(group m of u | h).

    Equal to group_scores(...).sum(-1) without the [U, H, M] tensor: groups are accumulated one at
    a time, split over threads (numpy releases the GIL).
    rel = (eta, Au[U, M]) also returns R[u, h] = sum_m log((1 - eta) P(group m | h) + eta exp(Au_um)):
    the score of an unbinned relative of bin h, whose level in each group is h's with prob 1 - eta
    and the community's otherwise.
    """
    U, M, _ = Lp.shape
    H = log_prof.shape[0]
    if rel is not None:
        le1, le0 = np.log1p(-rel[0]), np.log(rel[0]) + rel[1]

    def work(ms):
        acc = np.zeros((U, H))
        accR = np.zeros((U, H)) if rel is not None else None
        for m in ms:
            A = _group_term(Lp[:, m, :], log_prof[:, m, :])
            acc += A
            if rel is not None:
                accR += np.logaddexp(A + le1, le0[:, m, None])
        return acc, accR

    blocks = [b for b in np.array_split(np.arange(M), min(M, _n_threads())) if len(b)]
    if len(blocks) <= 1 or U * H < 10_000:
        S, R = work(range(M))
    else:
        with ThreadPoolExecutor(len(blocks)) as ex:
            parts = list(ex.map(work, blocks))
        S = sum(p[0] for p in parts)
        R = sum(p[1] for p in parts) if rel is not None else None
    return (S, R) if rel is not None else S


def unknown_score(R: np.ndarray, Su_comm: np.ndarray, eta: float, lam: float,
                  exclude: np.ndarray | None = None) -> np.ndarray:
    """log P(unit | unbinned genome) = log(lam * community + (1 - lam) * mean over bins of 'relative of h').

    lam = 1 (or eta = None) is the community-only model. exclude[u] = a bin left out of the relatives
    of unit u (its own bin, when that bin is removed as in a host-absent test), -1 for none.
    """
    if eta is None or lam >= 1:
        return Su_comm.copy()
    R = np.array(R, dtype=float, copy=True)
    H = R.shape[1]
    n = np.full(len(R), H, dtype=float)
    if exclude is not None:
        ex = exclude >= 0
        R[np.flatnonzero(ex), exclude[ex]] = -np.inf
        n[ex] = H - 1
    rel = logsumexp(R, axis=1) - np.log(np.maximum(n, 1))
    return np.logaddexp(np.log(lam) + Su_comm, np.log1p(-lam) + rel)


def score_units(Lp: np.ndarray, log_prof: np.ndarray) -> np.ndarray:
    """S[u, h] = log P(unit | host h) without own-MTase terms."""
    return summed_scores(Lp, log_prof)


def posterior(S_hosts: np.ndarray, S_unknown: np.ndarray, p_unknown: float, temperature: float = 1.0,
              host_mask: np.ndarray | None = None) -> np.ndarray:
    """Posterior over [bins..., unknown]. host_mask[u, h] = False removes bin h as a candidate for unit u."""
    U, H = S_hosts.shape
    prior = np.r_[np.full(H, np.log((1 - p_unknown) / H)), np.log(p_unknown)]
    logits = np.concatenate([S_hosts, S_unknown[:, None]], 1) / temperature + prior[None, :]
    if host_mask is not None:
        logits[:, :H] = np.where(host_mask, logits[:, :H], -np.inf)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def credible_set(post_row: np.ndarray, labels: list[str], level: float = 0.95) -> list[str]:
    order = np.argsort(-post_row)
    cum = np.cumsum(post_row[order])
    k = int(np.searchsorted(cum, level - 1e-12) + 1)
    return [labels[i] for i in order[:k]]


@dataclass
class Scores:
    S: np.ndarray        # [U, H]   log P(unit | bin)
    Su: np.ndarray       # [U]      log P(unit | unbinned genome)
    Au: np.ndarray       # [U, M]   per-group terms under a community draw
    Lp: np.ndarray       # [U, M, G]
    Su_comm: np.ndarray  # [U]      community part of the unbinned-genome score
    R: np.ndarray        # [U, H]   score as an unbinned relative of each bin


class FittedModel:
    """Host profiles and population prior for a given tau; scores arbitrary units."""

    def __init__(self, L_chunks, chunk_bin, n_bins, group_modtype, grid: Grid, tau, prior_lam=5.0,
                 prior_floor=0.02, contam_eps=0.05, robust_iters=4, init_w=None, init_pi=None):
        self.grid, self.tau, self.n_bins = grid, tau, n_bins
        self.T = transition_spec(grid, tau)
        self.Lp_chunks = marginalize_unit(L_chunks, self.T)
        self.chunk_bin = chunk_bin
        self._s_out = None
        self.w = np.ones(len(chunk_bin)) if init_w is None else init_w.copy()
        raw = init_pi
        for it in range(robust_iters + 1):
            self.ell = bin_loglik(self.Lp_chunks, chunk_bin, n_bins, self.w)
            self.log_pi, raw = fit_population_prior(self.ell, group_modtype, prior_lam, prior_floor,
                                                    init=raw, return_raw=True)
            self.log_prof = host_profiles(self.ell, self.log_pi)
            if it == robust_iters or contam_eps <= 0:
                break
            w_new = 1.0 - self._outlier_responsibility(contam_eps)
            converged = np.abs(w_new - self.w).max() < 1e-3
            self.w = w_new
            if converged:  # warm-started fits (e.g. the next tau) often need one or two rounds
                self.ell = bin_loglik(self.Lp_chunks, chunk_bin, n_bins, self.w)
                self.log_pi, raw = fit_population_prior(self.ell, group_modtype, prior_lam, prior_floor,
                                                        init=raw, return_raw=True)
                self.log_prof = host_profiles(self.ell, self.log_pi)
                break
        self.raw_pi = raw

    def _outlier_responsibility(self, eps: float, batch: int = 200) -> np.ndarray:
        """P(chunk is foreign to its bin): leave-chunk-out bin profile vs a random community host.

        Chunk batches run on threads (numpy releases the GIL); the community term does not depend
        on the weights and is computed once per model.
        """
        if getattr(self, "_s_out", None) is None:
            self._s_out = np.empty(len(self.chunk_bin))

            def out_term(sl):
                self._s_out[sl] = logsumexp(self.Lp_chunks[sl].astype(np.float64) + self.log_pi[None], -1).sum(-1)
            self._batched(out_term, batch)
        r = np.empty(len(self.chunk_bin))

        def work(sl):
            Lp = self.Lp_chunks[sl].astype(np.float64)
            ell_loo = self.ell[self.chunk_bin[sl]] - self.w[sl, None, None] * Lp
            s_in = logsumexp(Lp + host_profiles(ell_loo, self.log_pi), -1).sum(-1)
            r[sl] = expit(np.log(eps) - np.log1p(-eps) + self._s_out[sl] - s_in)
        self._batched(work, batch)
        return r

    def _batched(self, fn, batch):
        slices = [slice(s, s + batch) for s in range(0, len(self.chunk_bin), batch)]
        with ThreadPoolExecutor(_n_threads()) as ex:
            list(ex.map(fn, slices))

    def marginal(self, L_units):
        return marginalize_unit(L_units, self.T)

    eta: float | None = None   # unbinned genome = relative of a bin: per-group switch probability
    lam: float = 1.0           # weight of the community-only part of the unbinned-genome model

    def set_unknown_params(self, eta: float | None, lam: float):
        self.eta, self.lam = eta, lam

    def score(self, L_units) -> Scores:
        Lp = self.marginal(L_units)
        Au = group_scores(Lp, self.log_pi[None])[:, 0, :]
        S, R = summed_scores(Lp, self.log_prof, rel=(self.eta if self.eta else 0.05, Au))
        Su_comm = Au.sum(1)
        Su = unknown_score(R, Su_comm, self.eta, self.lam)
        return Scores(S, Su, Au, Lp, Su_comm, R)

    def host_terms(self, sc: Scores, rows: np.ndarray, hosts: np.ndarray | None = None,
                   log_prof: np.ndarray | None = None, batch: int = 1000) -> np.ndarray:
        """Per-group terms A[n, M] of units `rows`, each under one bin: hosts[i], or log_prof[i] if given."""
        A = np.empty((len(rows), sc.Lp.shape[1]))
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            prof = self.log_prof[hosts[s:s + batch]] if log_prof is None else log_prof[s:s + batch]
            A[s:s + batch] = logsumexp(sc.Lp[r].astype(np.float64) + prof, -1)
        return A

    def loo_profile(self, bin_idx: int, chunk_ids: np.ndarray) -> np.ndarray:
        ell = self.ell[bin_idx] - (self.w[chunk_ids, None, None] * self.Lp_chunks[chunk_ids]).sum(0)
        return host_profiles(ell[None], self.log_pi)

    def loo_rescore(self, sc: Scores, i: int, bin_idx: int, chunk_ids: np.ndarray):
        """Replace unit i's score for its own bin with a leave-chunks-out version (in place), and its
        score as a relative of that bin, so neither is built from the unit's own data.

        Returns the per-group terms A[m] of that leave-out score.
        """
        A = logsumexp(sc.Lp[i].astype(np.float64) + self.loo_profile(bin_idx, chunk_ids)[0], -1)
        sc.S[i, bin_idx] = A.sum()
        eta = self.eta if self.eta else 0.05
        sc.R[i, bin_idx] = np.logaddexp(np.log1p(-eta) + A, np.log(eta) + sc.Au[i]).sum()
        sc.Su[i] = unknown_score(sc.R[i:i + 1], sc.Su_comm[i:i + 1], self.eta, self.lam)[0]
        return A
