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
from scipy.special import expit

from .detection import DetectionParams, bb_logpmf
from .grid import Grid


def logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log sum exp(a) along one axis: the arithmetic of scipy.special.logsumexp (1.15 and later) without
    its array-API layer, which cost 86 us per call and was 60% of calibration and scoring, called once per
    fragment and per contig.

    As scipy does: the largest elements (m of them when tied) are taken out of the sum, s = the sum of
    exp(a - max) over the rest divided by m, result = log1p(s) + log(m) + max; rows whose result is not
    finite (all -inf, an inf, a nan) fall back to log(sum(exp(a))). Same operations on the same operands,
    so the same bits (test against scipy).
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a_max = a.max(axis=axis, keepdims=True)
        top = a == a_max
        m = top.sum(axis=axis, keepdims=True, dtype=a.dtype)
        e = np.exp(a - a_max)
        e[top] = 0.0
        s = e.sum(axis=axis, keepdims=True)
        s = np.where(s == 0, s, s / m)
        out = np.log1p(s) + np.log(m) + a_max
        finite = np.isfinite(out)
        if not finite.all():
            out = np.where(finite, out, np.log(np.exp(a).sum(axis=axis, keepdims=True)))
    return out if keepdims else out.squeeze(axis)


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
    np.log(np.maximum(q, 1e-300, out=q), out=q)  # in place: q is [U, H]
    q += am
    q += bm.T
    return q


def _n_threads():
    return min(16, os.cpu_count() or 1)


SCORE_BYTES = 4 << 30  # memory summed_scores may use for the [U, H] work arrays of its threads


def summed_scores(Lp: np.ndarray, log_prof: np.ndarray, rel: tuple | None = None, max_bytes: int | None = None):
    """S[u, h] = sum_m log P(group m of u | h).

    Equal to group_scores(...).sum(-1) without the [U, H, M] tensor: groups are accumulated one at
    a time, in blocks of groups that run on threads (numpy releases the GIL).
    rel = (eta, Au[U, M]) also returns R[u, h] = sum_m log((1 - eta) P(group m | h) + eta exp(Au_um)):
    the score of an unbinned relative of bin h, whose level in each group is h's with prob 1 - eta
    and the community's otherwise.

    Memory: a running block holds three [U, H] float64 arrays (its two sums and the current group's
    term, updated in place), and only as many blocks run at once as fit in max_bytes (at least one);
    a finished block is added to the result and freed. With every block on its own thread, each
    keeping its sums to the end next to four temporaries, 581,835 contigs x 495 bins needed 74 GB
    kept and up to 220 GB at peak; it now needs five arrays (11.5 GB), S and R included.
    The additions are the same in the same order whatever max_bytes is (groups within a block, then
    the blocks), and each group's matrix product is still one call over all units: scoring units in
    batches instead would cap memory further, but the BLAS product gives a row different last bits
    depending on the height of the batch (measured, up to 5e-16 relative), so results would change.
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
                A += le1
                accR += np.logaddexp(A, le0[:, m, None], out=A)
        return acc, accR

    blocks = [b for b in np.array_split(np.arange(M), min(M, _n_threads())) if len(b)]
    if len(blocks) <= 1 or U * H < 10_000:
        S, R = work(range(M))
        return (S, R) if rel is not None else S
    per_block = (3 if rel is not None else 2) * 8 * U * H
    at_once = int(max(1, min(len(blocks), (SCORE_BYTES if max_bytes is None else max_bytes) // per_block)))
    S = R = None
    with ThreadPoolExecutor(at_once) as ex:
        for i in range(0, len(blocks), at_once):
            for acc, accR in ex.map(work, blocks[i:i + at_once]):
                if S is None:  # 0 + first block, as sum() starts
                    S = np.add(acc, 0, out=acc)
                    R = np.add(accR, 0, out=accR) if rel is not None else None
                else:
                    S += acc
                    if rel is not None:
                        R += accR
                del acc, accR
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


def unknown_logit(R: np.ndarray | None, Su_comm: np.ndarray, eta: float | None, lam: float, temperature,
                  exclude: np.ndarray | None = None, temper_then_mix: bool = True) -> np.ndarray:
    """Tempered log-score of the unbinned-genome hypothesis, to be added to its log prior.

    "Unbinned genome" is a mixture of H + 1 elementary explanations: a community draw (prior weight lam) and
    a relative of bin h (weight (1 - lam) / H each). The temperature T is a power on the likelihood of the
    data under one explanation (the data carry 1 / T of their nominal information); the mixture weights are
    prior probabilities and are not data. So every explanation is tempered on its own and the tempered terms
    are mixed (temper_then_mix, the coherent order):

        Zu = log( lam exp(Su_comm / T) + (1 - lam) / H * sum_h exp(R_h / T) )

    exactly as the bins are (exp(S_h / T), then prior (1 - p_u) / H). temper_then_mix=False is the earlier
    order, unknown_score(...) / T: the mixture is summed first and then tempered, which also raises the
    weights 1 / H to the power 1 / T. A contig that fits one bin (and so that bin's relative) and nothing
    else then gains log(H) (1 - 1 / T) of log odds for "unknown" over its bin from the order of operations
    alone: 5 nats at H = 490, T = 5.

    temperature: scalar or one value per unit. lam = 1 (or eta = None) is the community-only model, where
    both orders agree.
    """
    T = np.asarray(temperature, float)
    if eta is None or lam >= 1 or R is None:
        return Su_comm / T
    if not temper_then_mix:
        return unknown_score(R, Su_comm, eta, lam, exclude) / T
    R = np.array(R, dtype=float, copy=True)
    H = R.shape[1]
    n = np.full(len(R), H, dtype=float)
    if exclude is not None:
        ex = exclude >= 0
        R[np.flatnonzero(ex), exclude[ex]] = -np.inf
        n[ex] = H - 1
    rel = logsumexp(R / (T[:, None] if T.ndim else T), axis=1) - np.log(np.maximum(n, 1))
    return np.logaddexp(np.log(lam) + Su_comm / T, np.log1p(-lam) + rel)


def score_units(Lp: np.ndarray, log_prof: np.ndarray) -> np.ndarray:
    """S[u, h] = log P(unit | host h) without own-MTase terms."""
    return summed_scores(Lp, log_prof)


def posterior(S_hosts: np.ndarray, S_unknown: np.ndarray | None, p_unknown: float, temperature=1.0,
              host_mask: np.ndarray | None = None, Zu: np.ndarray | None = None) -> np.ndarray:
    """Posterior over [bins..., unknown]. host_mask[u, h] = False removes bin h as a candidate for unit u.

    temperature: scalar or one value per unit. Zu: the unknown hypothesis' tempered log-score
    (unknown_logit); when given it is used as it is and S_unknown is ignored, otherwise the unknown's
    logit is S_unknown / temperature.
    """
    U, H = S_hosts.shape
    T = np.asarray(temperature, float)
    prior = np.r_[np.full(H, np.log((1 - p_unknown) / H)), np.log(p_unknown)]
    Tb = T[:, None] if T.ndim else T
    # one [U, H + 1] array, updated in place
    if Zu is None:
        logits = np.concatenate([S_hosts, S_unknown[:, None]], 1)
        logits /= Tb
    else:
        logits = np.concatenate([S_hosts, Zu[:, None]], 1)
        logits[:, :H] /= Tb
    logits += prior[None, :]
    if host_mask is not None:
        logits[:, :H] = np.where(host_mask, logits[:, :H], -np.inf)
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits, out=logits)


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
    Lp: np.ndarray       # [U, M, G], or [len(lp_rows), M, G] when only some units' rows were kept
    Su_comm: np.ndarray  # [U]      community part of the unbinned-genome score
    R: np.ndarray        # [U, H]   score as an unbinned relative of each bin
    lp_rows: np.ndarray | None = None  # ascending unit indices whose Lp rows are held (None: all units)

    def lp(self, rows):
        """Lp of unit(s) `rows` (unit indices), wherever their rows sit in the kept table."""
        if self.lp_rows is None:
            return self.Lp[rows]
        j = np.searchsorted(self.lp_rows, rows)
        if np.any(j >= len(self.lp_rows)) or np.any(self.lp_rows[np.minimum(j, len(self.lp_rows) - 1)] != rows):
            raise KeyError("Lp was not kept for these units (see FittedModel.score, lp_rows)")
        return self.Lp[j]


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

    def _outlier_responsibility(self, eps: float, batch: int | None = None) -> np.ndarray:
        """P(chunk is foreign to its bin): leave-chunk-out bin profile vs a random community host.

        Chunk batches run on threads (numpy releases the GIL); the community term does not depend
        on the weights and is computed once per model. A batch is at most 200 chunks and at most
        30,000 (chunk, group) cells: every thread holds about nine float64 [batch, M, G] arrays, 13.5 GB
        on 16 threads for 913 groups at 200 chunks. Each chunk's value is computed from its own rows
        only, so the batch size does not change it.
        """
        if batch is None:
            batch = min(200, max(1, 30_000 // self.Lp_chunks.shape[1]))
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

    def score(self, L_units, lp_rows=None) -> Scores:
        """Scores of every unit. lp_rows: ascending unit indices whose Lp rows the result keeps (for
        loo_rescore and host_terms); None keeps all. The pipeline keeps the binned contigs only: Lp of
        all targets is a second copy of their likelihood tables (18.8 GB for 80,526 contigs x 913
        groups), read afterwards for the leave-one-out scores of binned contigs and nothing else.
        """
        Lp = self.marginal(L_units)
        Au = group_scores(Lp, self.log_pi[None])[:, 0, :]
        S, R = summed_scores(Lp, self.log_prof, rel=(self.eta if self.eta else 0.05, Au))
        Su_comm = Au.sum(1)
        Su = unknown_score(R, Su_comm, self.eta, self.lam)
        if lp_rows is not None:
            lp_rows = np.asarray(lp_rows, dtype=np.int64)
            Lp = Lp[lp_rows]
        return Scores(S, Su, Au, Lp, Su_comm, R, lp_rows)

    def host_terms(self, sc: Scores, rows: np.ndarray, hosts: np.ndarray | None = None,
                   log_prof: np.ndarray | None = None, batch: int = 1000) -> np.ndarray:
        """Per-group terms A[n, M] of units `rows`, each under one bin: hosts[i], or log_prof[i] if given."""
        A = np.empty((len(rows), sc.Lp.shape[1]))
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            prof = self.log_prof[hosts[s:s + batch]] if log_prof is None else log_prof[s:s + batch]
            A[s:s + batch] = logsumexp(sc.lp(r).astype(np.float64) + prof, -1)
        return A

    def loo_profile(self, bin_idx: int, chunk_ids: np.ndarray) -> np.ndarray:
        ell = self.ell[bin_idx] - (self.w[chunk_ids, None, None] * self.Lp_chunks[chunk_ids]).sum(0)
        return host_profiles(ell[None], self.log_pi)

    def loo_rescore(self, sc: Scores, i: int, bin_idx: int, chunk_ids: np.ndarray):
        """Replace unit i's score for its own bin with a leave-chunks-out version (in place), and its
        score as a relative of that bin, so neither is built from the unit's own data.

        Returns the per-group terms A[m] of that leave-out score.
        """
        A = logsumexp(sc.lp(i).astype(np.float64) + self.loo_profile(bin_idx, chunk_ids)[0], -1)
        sc.S[i, bin_idx] = A.sum()
        eta = self.eta if self.eta else 0.05
        sc.R[i, bin_idx] = np.logaddexp(np.log1p(-eta) + A, np.log(eta) + sc.Au[i]).sum()
        sc.Su[i] = unknown_score(sc.R[i:i + 1], sc.Su_comm[i:i + 1], self.eta, self.lam)[0]
        return A

    def relatives(self, sc: Scores, eta: float, loo: dict | None = None) -> np.ndarray:
        """R[u, h] for another eta than the one `sc` was scored with (the unknown model can be refitted once
        p_unknown is known). loo: {unit: (bin, A[m])} from loo_rescore, so the leave-out terms are kept."""
        if sc.lp_rows is not None:
            raise ValueError("relatives() needs the Lp rows of every unit: score with lp_rows=None")
        _, R = summed_scores(sc.Lp, self.log_prof, rel=(eta, sc.Au))
        for i, (b, A) in (loo or {}).items():
            R[i, b] = np.logaddexp(np.log1p(-eta) + A, np.log(eta) + sc.Au[i]).sum()
        return R
