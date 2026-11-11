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
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.special import expit

from .detection import DetectionParams, bb_logpmf
from .grid import Grid


def logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False, work: np.ndarray | None = None) -> np.ndarray:
    """log sum exp(a) along one axis: the arithmetic of scipy.special.logsumexp (1.15 and later) without
    its array-API layer, which cost 86 us per call and was 60% of calibration and scoring, called once per
    fragment and per contig.

    As scipy does: the largest elements (m of them when tied) are taken out of the sum, s = the sum of
    exp(a - max) over the rest divided by m, result = log1p(s) + log(m) + max; rows whose result is not
    finite (all -inf, an inf, a nan) fall back to log(sum(exp(a))). Same operations on the same operands,
    so the same bits (test against scipy).

    work: an array of the shape and dtype of `a` that may be overwritten (the exponentials are written
    there instead of into two new arrays as large as `a`; same values).
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a_max = a.max(axis=axis, keepdims=True)
        top = a == a_max
        m = top.sum(axis=axis, keepdims=True, dtype=a.dtype)
        if work is None:
            e = np.exp(a - a_max)
        else:
            e = np.subtract(a, a_max, out=work)
            np.exp(e, out=e)
        e[top] = 0.0
        s = e.sum(axis=axis, keepdims=True)
        s = np.where(s == 0, s, s / m)
        out = np.log1p(s) + np.log(m) + a_max
        finite = np.isfinite(out)
        if not finite.all():
            out = np.where(finite, out, np.log(np.exp(a).sum(axis=axis, keepdims=True)))
    return out if keepdims else out.squeeze(axis)


def _rs(feature: str = "model_kernels"):
    """The optional compiled model kernels (package epirefine-rs), or None when they are not installed, are
    switched off (EPIREFINE_RS=0) or do not provide the interface this code was written for. The numpy code
    below is the reference; the kernels do the same float64 arithmetic in one pass per unit, and their results
    differ from it in the last bits of exp, log and sums (relative 1e-15), except cell_sums, which adds in
    numpy's order and returns numpy's bits."""
    if os.environ.get("EPIREFINE_RS", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        import epirefine_rs
    except ImportError:
        return None
    return epirefine_rs if epirefine_rs.supports("api>=1") and epirefine_rs.supports(feature) else None


def _cores() -> int:
    """The cores this process may run on: its CPU affinity where the platform has one (a SLURM job, taskset), not
    the machine's count (16 threads on a 4-core allocation only competed for the cores and held more memory)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _n_threads():
    """Threads for the passes of this module: the cores, at most 16, and no more than OMP_NUM_THREADS or
    OPENBLAS_NUM_THREADS allow when one is set (the worker processes of the unit-table pass run with both at
    1: each of them is one thread of that pass)."""
    n = min(16, _cores())
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        v = os.environ.get(var, "").strip()
        if v.isdigit() and int(v) > 0:
            n = min(n, int(v))
    return n


def _on_threads(fn, n: int, min_step: int = 1) -> None:
    """fn(slice) over [0, n) in contiguous pieces, one per thread (numpy releases the GIL in its loops).
    For work in which every row is computed from its own values only: the pieces do not change a result."""
    step = max(min_step, -(-n // _n_threads()))
    slices = [slice(s, min(s + step, n)) for s in range(0, n, step)]
    if len(slices) <= 1:
        for sl in slices:
            fn(sl)
        return
    with ThreadPoolExecutor(len(slices)) as ex:
        list(ex.map(fn, slices))


def site_terms(keys: pl.DataFrame, det: dict[int, DetectionParams], grid: Grid, block: int = 4096) -> np.ndarray:
    """log P(k | n, phi = x_g) for each row (group, n, k) of `keys`: one row per distinct count pattern.

    The [P, G] table is filled in row blocks: in one expression the two broadcast sums are held next
    to the result, three times the table (10.9 GB instead of 3.6 GB for 7.1 million patterns). Blocks of 4,096
    rows keep the work in the CPU cache (one worker thread: 10 instead of 16 ns a cell at 65,536 rows).
    """
    grp, n, k = (keys[c].to_numpy() for c in ("group", "n", "k"))
    n, k = n.astype(float), k.astype(float)
    p = [det[g] for g in range(max(det) + 1)]
    mu0, k0, mu1, k1 = (np.array([getattr(q, f) for q in p])[grp] for f in ("mu0", "kappa0", "mu1", "kappa1"))
    la, lb = bb_logpmf(k, n, mu1, k1), bb_logpmf(k, n, mu0, k0)
    return mix_log(la, lb, grid, block)


def mix_log(la: np.ndarray, lb: np.ndarray, grid: Grid, block: int = 65_536) -> np.ndarray:
    """[P, G] float64: logaddexp(la[p] + log x_g, lb[p] + log(1 - x_g)) for 1-d la, lb.

    With one logarithm per cell: with d = la - lb and the larger of the two as base, base + log(x + (1 - x) e^-d)
    (d > 0) or base + log((1 - x) + x e^d) (d <= 0); e^-|d| is one exponential per row. numpy's logaddexp is
    scalar code (20 ns a cell: a quarter of a unit-table worker's time on ZymoFecal); this agrees with it to
    rounding (float64, 5e-14 relative on a ZymoFecal part), and the float32 tables built from it were the same
    bit for bit (one ZymoFecal part, the training sets, ZymoOral)."""
    out = np.empty((len(la), grid.size))
    d = la - lb
    up = d > 0
    base = np.where(up, la, lb)[:, None]
    e = np.exp(-np.abs(d))[:, None]
    x, y = grid.x[None, :], 1.0 - grid.x[None, :]
    for s in range(0, len(la), block):
        sl = slice(s, s + block)
        o, u = out[sl], up[sl, None]
        np.multiply(np.where(u, y, x), e[sl], out=o)   # the smaller term's weight times e^-|d|
        o += np.where(u, x, y)                          # plus the larger term's weight
        np.log(o, out=o)
        o += base[sl]
    both = np.isneginf(la) & np.isneginf(lb)
    if both.any():
        out[both] = -np.inf
    return out


def _cell_sums(table: np.ndarray, r: np.ndarray, w: np.ndarray, starts: np.ndarray, chunk_rows: int = 200_000) -> np.ndarray:
    """[cells, G] float64: per cell (its rows start at starts[c]) the sum of w[i] * table[r[i]], as
    np.add.reduceat(table[r] * w[:, None], starts, axis=0) adds it (the first row plus numpy's pairwise sum of
    the others), in blocks of whole cells of about chunk_rows rows, so that the [rows, G] products are never
    held for more than a block. The kernel of epirefine-rs does the same additions in one pass (same bits)."""
    rs = _rs("cell_sums")
    if rs is not None:
        return rs.cell_sums(table, r, w, starts, threads=1 if len(r) < 100_000 else _n_threads())
    res = np.empty((len(starts), table.shape[1]))
    bounds = np.r_[starts, len(r)]
    # the blocks run on threads, each a share of chunk_rows, so that together they hold what one block would
    threads = _n_threads()
    blocks, i = [], 0
    while i < len(starts):
        j = max(i + 1, int(np.searchsorted(bounds, bounds[i] + max(1, chunk_rows // threads), "right")) - 1)
        blocks.append((i, j))
        i = j

    def reduce(block):
        i, j = block
        lo, hi = bounds[i], bounds[j]
        res[i:j] = np.add.reduceat(table[r[lo:hi]] * w[lo:hi, None], starts[i:j] - lo, axis=0)
    if threads <= 1 or len(blocks) <= 1:
        for block in blocks:
            reduce(block)
    else:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(reduce, blocks))
    return res


def unit_loglik(sites: pl.DataFrame, n_units: int, n_groups: int, det: dict[int, DetectionParams],
                grid: Grid, chunk_rows: int = 200_000, dtype=np.float64, batch_rows: int = 2_000_000) -> np.ndarray:
    """L[u, m, g] from a site table with integer columns unit, group, n, k (and optional count w); see
    unit_loglik_arrays."""
    w = sites["w"].to_numpy() if "w" in sites.columns else None
    return unit_loglik_arrays(*(sites[c].to_numpy() for c in ("unit", "group", "n", "k")), n_units, n_groups, det, grid,
                              w=w, chunk_rows=chunk_rows, dtype=dtype, batch_rows=batch_rows)


def _cells_by_keys(unit, group, n, k, n_groups, det, grid, out, chunk_rows) -> bool:
    """The body of unit_loglik_arrays for one batch without weights, with numpy on one integer key per row,
    (unit, group, n, k) packed into the bits of an int64, instead of polars' group-by on four columns, join and
    sort (single-threaded in a worker: 19 of the 42 s of a ZymoFecal part). np.unique sorts the keys, which is the
    order of the (unit, group) cells with their rows in ascending (n, k): the same rows in the same order, the
    same sums. Returns False (nothing done) when the key does not fit in 62 bits."""
    if not len(unit):
        return True
    bn, bk = max(int(n.max()), 1).bit_length(), max(int(k.max()), 1).bit_length()
    flat = unit.astype(np.int64) * n_groups + group.astype(np.int64)
    if max(int(flat.max()), 1).bit_length() + bn + bk > 62 or int(n.min()) < 0 or int(k.min()) < 0:
        return False
    key = (flat << (bn + bk)) | (n.astype(np.int64) << bk) | k.astype(np.int64)
    del flat
    key, wt = np.unique(key, return_counts=True)
    flat = key >> (bn + bk)
    nk = key & ((1 << (bn + bk)) - 1)
    del key
    codes, r = np.unique(((flat % n_groups) << (bn + bk)) | nk, return_inverse=True)   # distinct (group, n, k)
    del nk
    keys = pl.DataFrame({"group": codes >> (bn + bk), "n": (codes >> bk) & ((1 << bn) - 1), "k": codes & ((1 << bk) - 1)})
    table = site_terms(keys, det, grid)
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    out[flat[starts]] = _cell_sums(table, r, wt.astype(float), starts, chunk_rows)
    return True


def unit_loglik_arrays(unit: np.ndarray, group: np.ndarray, n: np.ndarray, k: np.ndarray, n_units: int, n_groups: int,
                       det: dict[int, DetectionParams], grid: Grid, w: np.ndarray | None = None,
                       chunk_rows: int = 200_000, dtype=np.float64, batch_rows: int = 2_000_000) -> np.ndarray:
    """L[u, m, g] from one row per site (or per w sites): unit, group, n reads, k of them methylated.

    The per-site term depends only on (group, n, k), so it is computed once per distinct pattern
    and gathered, instead of evaluating transcendental functions for every site.

    Every (unit, group) cell is one float64 sum over its distinct (n, k) patterns in ascending (n, k) (added
    as numpy's reduceat adds them, _cell_sums), stored in `dtype`. The order is part of the definition: up to 0.11 the rows of a cell were summed in the order
    polars' group_by returned them, which differs from run to run, and a cell that crossed a 200,000-row block
    was rounded to float32 twice, so two runs of the same command gave tables that differed in the last
    float32 digit of several hundred values per part (ZymoFecal). With the order fixed, a cell does not depend
    on which other units are processed with it: units are grouped and reduced `batch_rows` site rows at a time
    (whole units), so the group-by never sees more than that, whatever the size of the part.
    """
    out = np.zeros((n_units * n_groups, grid.size), dtype)
    N = len(unit)
    if N == 0:
        return out.reshape(n_units, n_groups, grid.size)
    if N <= batch_rows:
        batches = [slice(None)]
    else:  # rows of whole units, about batch_rows rows per batch: slices when the rows come unit by unit
        in_order = bool((unit[1:] >= unit[:-1]).all())
        order = None if in_order else np.argsort(unit, kind="stable")
        ends = np.cumsum(np.bincount(unit, minlength=n_units))  # rows of the units up to and including u
        batches, a = [], 0
        while a < N:
            j = int(np.searchsorted(ends, a + batch_rows, "right")) - 1      # last unit that still fits
            b = int(ends[j]) if j >= 0 and ends[j] > a else int(ends[np.searchsorted(ends, a, "right")])
            batches.append(slice(a, b) if in_order else order[a:b])
            a = b
    for idx in batches:
        if w is None and _cells_by_keys(unit[idx], group[idx], n[idx], k[idx], n_groups, det, grid, out, chunk_rows):
            continue
        cols = {"unit": unit, "group": group, "n": n, "k": k}
        if w is not None:
            cols["w"] = w
        df = pl.DataFrame({c: v[idx] for c, v in cols.items()})
        agg = df.group_by("unit", "group", "n", "k").agg((pl.col("w").sum() if w is not None else pl.len()).alias("w"))
        del df
        keys = agg.select("group", "n", "k").unique().with_row_index("r")
        table = site_terms(keys, det, grid)
        agg = (agg.with_columns((pl.col("unit").cast(pl.Int64) * n_groups + pl.col("group")).alias("flat"))
               .join(keys, on=["group", "n", "k"]).sort("flat", "n", "k"))
        r, wt, flat = agg["r"].to_numpy(), agg["w"].to_numpy().astype(float), agg["flat"].to_numpy()
        del agg, keys
        starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
        out[flat[starts]] = _cell_sums(table, r, wt, starts, chunk_rows)
    return out.reshape(n_units, n_groups, grid.size)


def _stack_product(e: np.ndarray, Tt: np.ndarray) -> np.ndarray:
    """e @ Tt for e [units, groups, G], with the bits of numpy's stacked product (one [groups, G] x [G, G]
    product per unit, one after another: 27 of the 30 s marginalize_unit took on 30,479 chunks x 710 groups).

    One product over all rows at once runs on the BLAS threads, but a BLAS library may sum a product of small
    matrices in another order than a large one (OpenBLAS does, below about a million multiplications): then
    the rows of the two forms differ in their last bits. So the forms are compared on the first units (as many
    as make the single product a large one); when they agree the single product is used, otherwise the
    per-unit products, in unit slices on threads.
    """
    n, Mm, G = e.shape
    k = min(n, max(3, -(-4_000_000 // max(1, Mm * G * Tt.shape[1]))))
    head = e[:k] @ Tt
    flat = (e[:k].reshape(-1, G) @ Tt).reshape(head.shape)
    if k == n:
        return head
    if np.array_equal(head, flat):
        return (e.reshape(-1, G) @ Tt).reshape(n, Mm, Tt.shape[1])
    q = np.empty((n, Mm, Tt.shape[1]), np.result_type(e, Tt))
    q[:k] = head

    def rows(sl):
        np.matmul(e[k:][sl], Tt, out=q[k:][sl])
    _on_threads(rows, n - k, 8)
    return q


def marginalize_unit(L: np.ndarray, T, batch: int = 2048, out: np.ndarray | None = None) -> np.ndarray:
    """Lp[..., i] = log sum_j T[i, j] exp(L[..., j]); keeps the dtype of L, works in unit batches.

    T is one [G, G] matrix for all groups, or a list of (group indices, [G, G] matrix) pairs when
    groups have different between-unit variation (per-group tau).

    Per batch, the element-wise passes (max, exp before the product; floor, log, shift after it) run on
    threads over row slices, and so does the matrix product (_stack_product). All of it was one thread's work
    (1.4 billion cells for 30,479 chunks x 710 groups: 30 s of the 210 s each calibration model took), with
    the same bits. With an identity matrix (no between-unit variation) the product returns its input and is
    skipped.
    """
    # out: a buffer of L's shape and dtype to write into (the calibration reuses one across its fits: a fresh 5.5 GB
    # table per fit is 5.5 GB of page faults)
    out = np.empty_like(L) if out is None or out.shape != L.shape or out.dtype != L.dtype else out
    parts = [(slice(None), T)] if isinstance(T, np.ndarray) else T
    tiny = np.finfo(L.dtype).tiny
    if not isinstance(T, np.ndarray):
        return _marginalize_parts(L, parts, batch, out, tiny)
    for ms, Tm in parts:
        Tt = Tm.T.astype(L.dtype)
        identity = Tm.shape[0] == Tm.shape[1] and np.array_equal(Tm, np.eye(Tm.shape[0]))
        for s in range(0, L.shape[0], batch):
            x = L[s:s + batch, ms]
            mx = np.empty(x.shape[:-1] + (1,), L.dtype)
            e = np.empty(x.shape, L.dtype)

            def before(sl):
                np.max(x[sl], axis=-1, keepdims=True, out=mx[sl])
                np.subtract(x[sl], mx[sl], out=e[sl])
                np.exp(e[sl], out=e[sl])
            _on_threads(before, len(x), 16)
            q = e if identity else _stack_product(e, Tt)
            dst = out[s:s + batch] if isinstance(ms, slice) else None

            def after(sl):
                r = q[sl]
                np.maximum(r, tiny, out=r)
                np.log(r, out=r)
                if dst is not None:
                    np.add(r, mx[sl], out=dst[sl])
                else:
                    r += mx[sl]
            _on_threads(after, len(x), 16)
            if dst is None:
                out[s:s + batch, ms] = q
    return out


def _marginalize_parts(L, parts, batch, out, tiny):
    """marginalize_unit for a per-group tau: each batch passes through max/exp and floor/log/shift once for all
    groups, and each tau's groups (a copy of their columns, as before) through that tau's product, the identity
    (tau = inf, most groups on ZymoFecal) through none. Before, every tau value made its own passes over a copy of
    its columns (21 s against 6 s for one tau on the 34,878 contigs of ZymoFecal). The same operations on the same
    operands: the same bits."""
    prods = []
    for ms, Tm in parts:
        if not (Tm.shape[0] == Tm.shape[1] and np.array_equal(Tm, np.eye(Tm.shape[0]))):
            prods.append((np.asarray(ms), Tm.T.astype(L.dtype)))
    for s in range(0, L.shape[0], batch):
        x = L[s:s + batch]
        mx = np.empty(x.shape[:-1] + (1,), L.dtype)
        e = np.empty(x.shape, L.dtype)

        def before(sl):
            np.max(x[sl], axis=-1, keepdims=True, out=mx[sl])
            np.subtract(x[sl], mx[sl], out=e[sl])
            np.exp(e[sl], out=e[sl])
        _on_threads(before, len(x), 16)
        for ms, Tt in prods:
            e[:, ms] = _stack_product(e[:, ms], Tt)
        dst = out[s:s + batch]

        def after(sl):
            r = e[sl]
            np.maximum(r, tiny, out=r)
            np.log(r, out=r)
            np.add(r, mx[sl], out=dst[sl])
        _on_threads(after, len(x), 16)
    return out


def transition_spec(grid: Grid, tau, delta: float = 1.0):
    """One transition matrix for a scalar tau, or (groups, matrix) pairs for a per-group tau array."""
    if np.ndim(tau) == 0:
        return grid.transition(tau, delta)
    tau = np.asarray(tau, dtype=float)
    return [(np.flatnonzero(tau == t), grid.transition(float(t), delta)) for t in np.unique(tau)]


BIN_BYTES = 2 << 30  # memory bin_loglik may use for the float64 copies of the bins its threads are summing


def bin_loglik(Lp_chunks: np.ndarray, chunk_bin: np.ndarray, n_bins: int, w: np.ndarray | None = None) -> np.ndarray:
    """ell[h] = sum of w * Lp over the chunks of bin h (float64).

    Bins are summed on threads (each bin is one product over its own chunks, as before: same bits), as many
    at once as keep the float64 copies of their chunks within BIN_BYTES (the largest bin sets the number)."""
    w = np.ones(len(chunk_bin)) if w is None else w
    rs = _rs()
    if rs is not None and Lp_chunks.dtype == np.float32 and Lp_chunks.ndim == 3:
        return rs.bin_loglik(Lp_chunks, chunk_bin, n_bins, w, threads=_n_threads())
    ell = np.zeros((n_bins,) + Lp_chunks.shape[1:])
    order = np.argsort(chunk_bin, kind="stable")
    bounds = np.searchsorted(chunk_bin[order], np.arange(n_bins + 1))

    def one(h):
        idx = order[bounds[h]:bounds[h + 1]]
        if len(idx):
            ell[h] = np.tensordot(w[idx], Lp_chunks[idx].astype(np.float64), 1)
    largest = int(np.diff(bounds).max()) if n_bins else 0
    per_bin = largest * int(np.prod(Lp_chunks.shape[1:])) * (8 + Lp_chunks.dtype.itemsize)
    threads = int(max(1, min(_n_threads(), n_bins, BIN_BYTES // max(per_bin, 1))))
    if threads <= 1:
        for h in range(n_bins):
            one(h)
    else:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, range(n_bins)))
    return ell


def npmle(ell: np.ndarray, iters: int = 300, tol: float = 1e-6, init: np.ndarray | None = None,
          blocks: int | None = None, round_iters: int = 10) -> np.ndarray:
    """Grid NPMLE of pi[m, g] from ell[b, m, g] by EM, all groups at once.

    Stops when no cell moves by more than `tol`; `init` warm-starts from an earlier solution.

    Groups do not interact, except through the stopping rule (the largest move over all groups). All groups
    in one array were one thread's work (17 s per fit for 490 bins x 710 groups, 86 s of the 165 s a
    calibration model took). The groups are therefore iterated in `blocks` blocks (default: 8),
    on threads, `round_iters` iterations at a time; the largest move per iteration is collected over the
    blocks after each round and the state of the first iteration that meets `tol` is returned: per group the
    same products as before, the same stop, the same bits.
    """
    B, M, G = ell.shape
    lik = np.exp(ell - ell.max(-1, keepdims=True)).transpose(1, 0, 2).copy()  # M, B, G
    pi = np.full((M, G), 1.0 / G) if init is None else init.copy()
    rs = _rs() if blocks is None else None
    if rs is not None:
        return rs.npmle(lik, pi, iters, tol, threads=_n_threads())
    n_blocks = min(8, _n_threads()) if blocks is None else blocks  # more threads than 8 wait for each other
    if M * B * G < 200_000:  # small tables: the threads cost more than they save
        n_blocks = 1
    per_block = -(-M // max(1, min(n_blocks, M)))
    blocks = [slice(s, s + per_block) for s in range(0, M, per_block)]
    threads = min(_n_threads(), len(blocks))
    if len(blocks) <= 1:
        for _ in range(iters):
            denom = np.maximum(np.matmul(lik, pi[:, :, None])[:, :, 0], 1e-300)  # [M, B]
            new = np.maximum(pi * np.matmul((1.0 / denom)[:, None, :], lik)[:, 0, :] / B, 1e-300)
            delta = np.abs(new - pi).max()
            pi = new
            if delta < tol:
                break
        return pi
    snaps = np.empty((round_iters, M, G))
    deltas = np.empty((len(blocks), round_iters))

    def advance(j, n):
        sl = blocks[j]
        lk, p = lik[sl], pi[sl]
        for i in range(n):
            denom = np.maximum(np.matmul(lk, p[:, :, None])[:, :, 0], 1e-300)
            new = np.maximum(p * np.matmul((1.0 / denom)[:, None, :], lk)[:, 0, :] / B, 1e-300)
            deltas[j, i] = np.abs(new - p).max()
            snaps[i, sl] = new
            p = new

    with ThreadPoolExecutor(max(1, threads)) as ex:
        done = 0
        while done < iters:
            n = min(round_iters, iters - done)
            list(ex.map(lambda j: advance(j, n), range(len(blocks))))
            moved = deltas[:, :n].max(0)
            hit = np.flatnonzero(moved < tol)
            last = int(hit[0]) if len(hit) else n - 1
            pi = snaps[last].copy()
            if len(hit):
                break
            done += n
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
    """log posterior over phi_h per (bin, group). Large tables in row slices on threads (each (bin, group) row
    from its own values: the same bits; 0.6 s a call on one thread for 490 bins x 710 groups, 5 calls a fit)."""
    lp = ell + log_pi[None, :, :]
    if lp.size < 1_000_000:
        return lp - logsumexp(lp, axis=-1, keepdims=True)
    flat = lp.reshape(-1, lp.shape[-1])

    def rows(sl):
        flat[sl] -= logsumexp(flat[sl], axis=-1, keepdims=True)
    _on_threads(rows, len(flat), 64)
    return lp


def group_scores(Lp: np.ndarray, log_prof: np.ndarray) -> np.ndarray:
    """A[u, h, m] = log sum_g exp(Lp[u, m, g] + log_prof[h, m, g])."""
    U, M, G = Lp.shape
    A = np.empty((U, log_prof.shape[0], M))
    if log_prof.shape[0] == 1 and U * M * G >= 1_000_000:
        # one profile (the community prior): _group_term's steps for a few groups at a time over all units, the
        # products as one stacked matmul whose items are the [units, G] x [G, 1] products of the group-by-group loop
        # (the same BLAS call on the same operands: the same bits; the loop's 710 small calls per scoring and their
        # element-wise steps took 33 s over a ZymoFecal run)
        b = log_prof[0]                                       # [M, G]
        bm = b.max(1, keepdims=True)                          # [M, 1]
        eb = _profile_exp(b)[:, :, None]                      # [M, G, 1]
        k = max(1, (32 << 20) // max(1, U * G * 8))           # groups per step: about 32 MB of exponentials

        def groups(sl):
            for m0 in range(sl.start, sl.stop, k):
                ms = slice(m0, min(m0 + k, sl.stop))
                a = Lp[:, ms, :]                              # [U, k, G]
                am = a.max(2, keepdims=True)
                e = np.empty((ms.stop - m0, U, G))
                e[...] = np.exp(a - am).transpose(1, 0, 2)    # (float32 exp, as _group_term)
                q = np.matmul(e, eb[ms])[:, :, 0].T           # [U, k]
                np.log(np.maximum(q, 1e-300, out=q), out=q)
                q += am[:, :, 0]
                q += bm[ms, 0]
                A[:, 0, ms] = q
        _on_threads(groups, M)
        return A
    for m in range(M):
        A[:, :, m] = _group_term(Lp[:, m, :], log_prof[:, m, :])
    return A


_TINY = 1e-250  # exp(log_prof - max) below this enters the products as 0 (see _profile_exp)


def _profile_exp(b: np.ndarray) -> np.ndarray:
    """exp(b - max b) along the last axis, entries below _TINY set to 0.

    About a third of the cells of a fitted profile lie more than 575 nats below their group's maximum: their
    exponentials are subnormal or close to it, and products with subnormal operands or results take a slow path
    in the CPU (the score products ran at 14 instead of 3 ns per unit, bin and group on ZymoFecal). A dropped
    term is below 1e-250, while the profile's maximum cell (1) meets an exponential of the unit table of at
    least 1.2e-38 (float32): a sum moves only where it is itself below about 1e-232. The scores were bit for bit
    the same on ZymoFecal tables (tests/test_speed_kernels.py checks it on synthetic ones)."""
    y = np.exp(b - b.max(-1, keepdims=True))
    y[y < _TINY] = 0.0
    return y


def _group_term(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    am = a.max(1, keepdims=True)
    bm = b.max(1, keepdims=True)
    q = np.exp(a - am).astype(np.float64) @ _profile_exp(b).T
    np.log(np.maximum(q, 1e-300, out=q), out=q)  # in place: q is [U, H]
    q += am
    q += bm.T
    return q


LOO_THREADS_MIN = 16_384  # groups x grid cells from which the leave-one-out loops run on threads (ZymoOral, 34 x 64:
                          # one thread was faster; ZymoFecal, 710 x 64: 1.6 ms of numpy work per contig)
LOO_BATCH = 16  # contigs per batch of the leave-one-out loops (loo_profiles / loo_rescore_many)
ROWS_MIN = 20_000_000   # unit x bin x group cells from which summed_scores takes the row form
ROWS_STEP = 128         # units per row slice of the row form (a divisor of FittedModel.SCORE_BATCH)
ROWS_GROUPS = 4         # groups per step of a row slice


def _score_form(U: int, H: int, M: int) -> str:
    return "rows" if U * H * M >= ROWS_MIN else "groups"


_blas_lock = threading.Lock()
_blas_api: list = []


def _blas_thread_api():
    """(set, get) of the number of threads of the OpenBLAS that numpy loaded, through ctypes; None when it
    cannot be found (another BLAS, another platform): the callers then run with BLAS's own threads."""
    if not _blas_api:
        api = None
        try:
            import ctypes
            with open("/proc/self/maps") as fh:
                libs = sorted({ln.split()[-1] for ln in fh if "openblas" in ln.lower() and ".so" in ln})
            names = [("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                     ("scipy_openblas_set_num_threads", "scipy_openblas_get_num_threads"),
                     ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
                     ("openblas_set_num_threads", "openblas_get_num_threads")]
            for path in libs:
                lib = ctypes.CDLL(path)
                for sn, gn in names:
                    if hasattr(lib, sn) and hasattr(lib, gn):
                        api = (getattr(lib, sn), getattr(lib, gn))
                        break
                if api:
                    break
        except (OSError, ValueError):
            api = None
        _blas_api.append(api)
    return _blas_api[0]


class _single_threaded_blas:
    """BLAS on one thread inside the block (restored after), for code that issues many small products from
    threads of its own: with BLAS's threads waiting next to them they ran 2-3x slower. Products then have
    the arithmetic of a one-thread BLAS whatever the number of cores."""

    def __enter__(self):
        _blas_lock.acquire()
        api = _blas_thread_api()
        self.prev = api[1]() if api else None
        if api:
            api[0](1)
        return self

    def __exit__(self, *exc):
        api = _blas_thread_api()
        if api and self.prev:
            api[0](self.prev)
        _blas_lock.release()


def _summed_scores_rows(Lp: np.ndarray, log_prof: np.ndarray, rel: tuple | None = None):
    """summed_scores for large tables: units in slices of ROWS_STEP on threads, each slice through all groups,
    ROWS_GROUPS at a time (stacked [step, G] x [G, H] products on a one-thread BLAS, work buffers reused).
    A unit's result depends only on its slice (ROWS_STEP-aligned), not on the thread count or on how many
    units are scored with it in steps of ROWS_STEP. The relatives' term is taken with one logarithm,
        log((1 - eta) P + eta e^Au) = log(q + c d) + log(1 - eta) + am + bm,
        q = sum_g e^(a - am) e^(b - bm), c = eta e^(Au - am) / (1 - eta), d = e^-bm,
    instead of numpy's logaddexp (scalar code, 20 ns a cell); c <= eta / (1 - eta) since Au <= am, and
    d <= G for normalised profiles. Agrees with the group form to about 1e-14 relative (other order of the
    sums over groups, another product kernel)."""
    U, M, G = Lp.shape
    H = log_prof.shape[0]
    bm = log_prof.max(-1)                                                          # [H, M]
    ebT = np.ascontiguousarray(_profile_exp(log_prof).transpose(1, 2, 0))          # [M, G, H]
    bmT = np.ascontiguousarray(bm.T)[:, None, :]                                   # [M, 1, H]
    S = np.empty((U, H))
    R = np.empty((U, H)) if rel is not None else None
    if rel is not None:
        eta = float(rel[0])
        dT = np.exp(-bmT)
        bmR = bmT + np.log1p(-eta)
        if not np.isfinite(dT).all():
            raise ValueError("summed_scores: profiles are not normalised (exp(-max) overflows)")
    K = ROWS_GROUPS

    def rows(sl):
        a = Lp[sl]
        n = len(a)
        am = a.max(-1)                                                             # [n, M]
        amT = np.ascontiguousarray(am.T)[:, :, None]                               # [M, n, 1]
        accS = np.zeros((n, H))
        xb, qb = np.empty((K, n, G)), np.empty((K, n, H))
        if rel is not None:
            cT = (np.exp(np.log(eta) + np.asarray(rel[1][sl], np.float64).T - am.T) / (1.0 - eta))[:, :, None]
            accR, tb = np.zeros((n, H)), np.empty((K, n, H))
        for m0 in range(0, M, K):
            ms = slice(m0, min(m0 + K, M))
            k = ms.stop - m0
            x, q = xb[:k], qb[:k]
            x[...] = np.exp(a[:, ms, :].transpose(1, 0, 2) - amT[ms])          # float32 exp, as the group form
            np.matmul(x, ebT[ms], out=q)
            np.maximum(q, 1e-300, out=q)
            if rel is not None:
                t = tb[:k]
                np.multiply(cT[ms], dT[ms], out=t)
                t += q
                np.log(t, out=t)
                t += amT[ms]
                t += bmR[ms]
                for j in range(k):
                    accR += t[j]
            np.log(q, out=q)
            q += amT[ms]
            q += bmT[ms]
            for j in range(k):
                accS += q[j]
        S[sl] = accS
        if rel is not None:
            R[sl] = accR

    slices = [slice(s, min(s + ROWS_STEP, U)) for s in range(0, U, ROWS_STEP)]
    with _single_threaded_blas():
        with ThreadPoolExecutor(max(1, min(_n_threads(), len(slices)))) as ex:
            list(ex.map(rows, slices))
    return (S, R) if rel is not None else S


SCORE_BYTES = 4 << 30  # memory summed_scores may use for the [U, H] work arrays of its threads


def summed_scores(Lp: np.ndarray, log_prof: np.ndarray, rel: tuple | None = None, max_bytes: int | None = None,
                  form: str | None = None):
    """S[u, h] = sum_m log P(group m of u | h).

    form: "rows" (_summed_scores_rows, the form for large tables: from ROWS_MIN unit x bin x group cells on) or
    "groups" (below); None chooses by the size of this call (_score_form). Callers that score the units of one
    table in batches pass the form of the whole table, so that every unit gets the same arithmetic.

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
    if (form or _score_form(U, H, M)) == "rows":
        return _summed_scores_rows(Lp, log_prof, rel)
    rs = _rs("score_sums") if U * H * M >= 100_000 else None
    if rs is not None:
        # The kernel of epirefine-rs does the sums; the exponentials stay numpy's (the float32 exp of the unit
        # tables decides the eighth digit of a score). Groups are added in ascending order there, in blocks of
        # groups here: scores agree to about 1e-14 relative.
        bm = log_prof.max(-1)                                                          # [H, M]
        eb = np.ascontiguousarray(_profile_exp(log_prof).transpose(1, 2, 0))           # [M, G, H]
        bm = np.ascontiguousarray(bm.T)
        S = np.empty((U, H))
        R = np.empty((U, H)) if rel is not None else None
        step = int(max(32, min(4096, (256 << 20) // max(1, M * Lp.shape[2] * 8))))  # e of a batch: about 256 MB
        for s in range(0, U, step):
            x = Lp[s:s + step]
            am = np.empty(x.shape[:2], Lp.dtype)
            e = np.empty(x.shape, np.float64)

            def fill(sl):
                np.max(x[sl], axis=-1, out=am[sl])
                e[sl] = np.exp(x[sl] - am[sl][..., None])
            _on_threads(fill, len(x))
            S[s:s + step], r = rs.score_sums(e, am.astype(np.float64), eb, bm, le1 if rel is not None else 0.0,
                                             le0[s:s + step] if rel is not None else None, threads=_n_threads())
            if rel is not None:
                R[s:s + step] = r
        return (S, R) if rel is not None else S

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


def unbinned_state_frequencies(off: np.ndarray, on: np.ndarray, q: np.ndarray, strength: float = 5.0,
                               iters: int = 500, tol: float = 1e-7) -> np.ndarray:
    """c[m]: how often motif group m is methylated among the given units (the unbinned contigs).

    off, on [n, M]: log-likelihood of each unit's sites in each group if the group is unmethylated / methylated
    (FittedModel.state_terms). Per group, c maximises sum_i log((1 - c) exp(off) + c exp(on)) over the units
    that have sites in it (EM, groups independent), with `strength` pseudo-units at the bins' frequency q, so
    that with few unbinned contigs the estimate stays at q (the community draw).

    Why: an unbinned genome was scored as a community draw, each group's state drawn at the frequency seen
    among the BINS. The motif list comes from the bins, so a genome without a bin lacks most listed motifs by
    construction, and a draw at the bins' frequencies makes that improbable (the product over groups of
    1 - q): the bin that is itself unmethylated, or methylated only at a motif many genomes share, then looks
    like the one good explanation of every such contig. The unbinned contigs are the population an unbinned
    contig comes from, so their own state frequencies are used instead: absence of a motif that is rare among
    them says little, presence of one that is rare among them says more than before, and a motif that many of
    them carry (Dam) says as little as it should.
    """
    off, on = off.astype(np.float64), on.astype(np.float64)
    has = np.abs(on - off) > 1e-9
    n = has.sum(0)
    d = np.where(has, on - off, 0.0)
    c = q.copy()
    for _ in range(iters):
        rho = expit(np.log(c) - np.log1p(-c) + d)
        new = np.clip(((rho * has).sum(0) + strength * q) / (n + strength), 1e-9, 1 - 1e-9)
        done = np.abs(new - c).max() < tol
        c = new
        if done:
            break
    return c


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
                 prior_floor=0.02, contam_eps=0.05, robust_iters=None, init_w=None, init_pi=None,
                 robust_fit="alternating", robust_damping=0.5, robust_tol=1e-3, lp_out=None):
        """robust_fit: how the chunk weights w = 1 - P(chunk foreign to its bin) and the bin profiles are
        brought into agreement.

        "alternating" (the scheme up to 0.10.x): w <- 1 - r(w) for a fixed 4 rounds. Where a bin holds two
        methylomes in comparable shares the pooled profile sits between them and fits no chunk, so every
        chunk is called foreign, the next profile is the prior, every chunk is a member again, and so on:
        a 2-cycle that the fixed round count ends in the pooled state (no chunk down-weighted).
        "damped": w <- w + damping * ((1 - r(w)) - w), until no weight moves by more than robust_tol
        (at most robust_iters rounds). The same fixed points, but the step is too short to jump over
        them; from the pooled start all weights shrink together, the profile widens, the majority's
        chunks fit it first and the fit settles with the minority down-weighted.
        robust_delta[u] is the last change of each chunk's weight (bins whose chunks still move did not
        converge: see robust_unconverged()).
        """
        self.grid, self.tau, self.n_bins = grid, tau, n_bins
        self.T = transition_spec(grid, tau)
        self.Lp_chunks = marginalize_unit(L_chunks, self.T, out=lp_out)
        self.chunk_bin = chunk_bin
        self._s_out = None
        self.w = np.ones(len(chunk_bin)) if init_w is None else init_w.copy()
        if robust_fit not in ("damped", "alternating"):
            raise ValueError(f"unknown robust_fit {robust_fit!r}")
        alpha = robust_damping if robust_fit == "damped" else 1.0
        if robust_iters is None:
            robust_iters = 50 if robust_fit == "damped" else 4
        raw = init_pi
        self.robust_delta = np.zeros(len(chunk_bin))
        self.robust_rounds = 0

        def fit_profiles(raw):
            self.ell = bin_loglik(self.Lp_chunks, chunk_bin, n_bins, self.w)
            self.log_pi, raw = fit_population_prior(self.ell, group_modtype, prior_lam, prior_floor,
                                                    init=raw, return_raw=True)
            self.log_prof = host_profiles(self.ell, self.log_pi)
            return raw

        for it in range(robust_iters + 1):
            raw = fit_profiles(raw)
            if it == robust_iters or contam_eps <= 0:
                break
            if robust_fit == "damped":
                self._s_out = None  # the community term follows the refitted population prior
            target = 1.0 - self._outlier_responsibility(contam_eps)
            # convergence is judged on the full step, so a damped fit does not stop early
            self.robust_delta = np.abs(target - self.w)
            # (alternating: the target itself, as in 0.10; w + 1.0 * (target - w) can differ from it in the last bit)
            self.w = target if alpha == 1.0 else self.w + alpha * (target - self.w)
            self.robust_rounds = it + 1
            if self.robust_delta.max() < robust_tol:  # warm-started fits (e.g. the next tau) often need one or two rounds
                raw = fit_profiles(raw)
                break
        self.raw_pi = raw

    def two_state_share(self, min_chunks: int = 5, screen: float = 0.05, max_groups: int = 20, batch: int = 500):
        """Per bin: the largest share, over motif groups, of the smaller of two methylation states among its chunks.

        One genome has one level per motif group: its chunks differ only by which of their sites are methylated,
        which the site model already describes (a group at level 0.5 gives chunks around 0.5). A bin that holds
        two methylomes (a merge of genomes, or of strains that differ in a methyltransferase) has chunks at two
        levels, one below and one above 0.5. For each bin and group the distribution of chunk levels is
        estimated (grid NPMLE over the bin's chunks) and compared with the best single level; when the
        distribution wins by more than the BIC cost of a second level and weight (log of the number of chunks),
        the share is the smaller of its masses below and above 0.5, otherwise 0.

        Why it is reported: when the two methylomes differ in few sites per chunk, the pooled profile between
        them explains every chunk better than a community draw, so the robust fit down-weights nothing and no
        contig is flagged (single-motif merges from about 20% on the monoculture stress sets). This share is
        then the only sign that the bin is mixed.

        Groups are screened first (chunks whose 95% interval lies wholly on one side of 0.5, at least
        min_chunks of them and two on the smaller side, smaller side >= screen of their evidence; at most
        max_groups per bin), so the NPMLE runs on few groups.
        Returns (share [H], group [H], -1 where no group shows two states).
        """
        H, (U, M, G) = self.n_bins, self.Lp_chunks.shape
        on_w, off_w = np.zeros((H, M)), np.zeros((H, M))
        on_n, off_n = np.zeros((H, M)), np.zeros((H, M))
        # per chunk and group: evidence and side of 0.5, computed row by row on threads (each row from its own
        # values: the same bits for any slicing), then added up per bin in chunk order as before
        on_all, off_all, info_all = np.empty((U, M), bool), np.empty((U, M), bool), np.empty((U, M))

        def side(sl):
            for s in range(sl.start, sl.stop, 64):
                e = min(s + 64, sl.stop)
                Lp = self.Lp_chunks[s:e].astype(np.float64)
                _, lo, hi, _ = self.grid.summarize(Lp)
                info = Lp.max(-1) - Lp.min(-1)
                on_all[s:e], off_all[s:e] = (lo >= 0.5) & (info > 0), (hi < 0.5) & (info > 0)
                info_all[s:e] = info
        _on_threads(side, U)
        for s in range(0, U, batch):  # np.add.at adds the rows one after another: the order of the old loop
            b, on, off, info = self.chunk_bin[s:s + batch], on_all[s:s + batch], off_all[s:s + batch], info_all[s:s + batch]
            np.add.at(on_w, b, info * on); np.add.at(off_w, b, info * off)
            np.add.at(on_n, b, on); np.add.at(off_n, b, off)
        del on_all, off_all, info_all
        tot = on_w + off_w
        minor = np.where((on_n + off_n >= min_chunks) & (np.minimum(on_n, off_n) >= 2) & (tot > 0),
                         np.minimum(on_w, off_w) / np.maximum(tot, 1e-300), 0.0)
        share, group = np.zeros(H), np.full(H, -1)
        upper = self.grid.x >= 0.5
        order = np.argsort(self.chunk_bin, kind="stable")
        bounds = np.searchsorted(self.chunk_bin[order], np.arange(H + 1))
        def one_bin(h):
            ms = np.flatnonzero(minor[h] >= screen)
            if not len(ms):
                return
            ms = ms[np.argsort(-minor[h, ms], kind="stable")[:max_groups]]
            idx = order[bounds[h]:bounds[h + 1]]
            ell = self.Lp_chunks[idx][:, ms, :].astype(np.float64)        # [chunks, groups, G]
            ell = ell[:, :, :] - ell.max(-1, keepdims=True)
            rho = npmle(ell, iters=200, tol=1e-5, blocks=1)               # [groups, G] (one thread per bin: the bins run on threads)
            two = np.log(np.maximum(np.einsum("umg,mg->um", np.exp(ell), rho), 1e-300)).sum(0)
            one = ell.sum(0).max(-1)
            ok = two - one > np.log(len(idx))
            sh = np.where(ok, np.minimum(rho[:, upper].sum(1), rho[:, ~upper].sum(1)), 0.0)
            k = int(sh.argmax())
            if sh[k] > 0:
                share[h], group[h] = sh[k], ms[k]
        # bins are independent (each writes its own entries); on threads
        with ThreadPoolExecutor(max(1, min(_n_threads(), H))) as ex:
            list(ex.map(one_bin, range(H)))
        return share, group

    def robust_unconverged(self, tol: float = 0.05) -> np.ndarray:
        """Per bin: True when some chunk weight was still moving by more than tol in the last round."""
        out = np.zeros(self.n_bins, bool)
        np.maximum.at(out, self.chunk_bin, self.robust_delta > tol)
        return out

    def _outlier_responsibility(self, eps: float, batch: int | None = None) -> np.ndarray:
        """P(chunk is foreign to its bin): leave-chunk-out bin profile vs a random community host.

        Chunk batches run on threads (numpy releases the GIL); the community term does not depend
        on the weights and is computed once per model. A batch is at most 200 chunks and at most
        30,000 (chunk, group) cells: every thread holds about nine float64 [batch, M, G] arrays, 13.5 GB
        on 16 threads for 913 groups at 200 chunks. Each chunk's value is computed from its own rows
        only, so the batch size does not change it.
        """
        rs = _rs() if batch is None and self.Lp_chunks.dtype == np.float32 else None
        if rs is not None:
            if getattr(self, "_s_out", None) is None:
                self._s_out = rs.community_terms(self.Lp_chunks, self.log_pi, threads=_n_threads())
            s_in = rs.leave_chunk_out_terms(self.Lp_chunks, self.chunk_bin, self.w, self.ell, self.log_pi,
                                            threads=_n_threads())
            return expit(np.log(eps) - np.log1p(-eps) + self._s_out - s_in)
        if batch is None:
            batch = min(200, max(1, 30_000 // self.Lp_chunks.shape[1]))
        if getattr(self, "_s_out", None) is None:
            self._s_out = np.empty(len(self.chunk_bin))

            def out_term(sl):
                a = self.Lp_chunks[sl].astype(np.float64)
                np.add(a, self.log_pi[None], out=a)
                self._s_out[sl] = logsumexp(a, -1, work=np.empty_like(a)).sum(-1)
            self._batched(out_term, batch)
        r = np.empty(len(self.chunk_bin))

        def work(sl):
            # the same operations on the same operands as
            #     ell_loo = ell[bin] - w * Lp;  s_in = logsumexp(Lp + host_profiles(ell_loo, log_pi)).sum()
            # written into three [batch, M, G] arrays instead of eleven (the time went into filling them)
            Lp = self.Lp_chunks[sl].astype(np.float64)
            a = self.ell[self.chunk_bin[sl]]
            t = np.multiply(self.w[sl, None, None], Lp)
            np.subtract(a, t, out=a)
            np.add(a, self.log_pi[None, :, :], out=a)
            np.subtract(a, logsumexp(a, axis=-1, keepdims=True, work=t), out=a)
            np.add(Lp, a, out=a)
            s_in = logsumexp(a, -1, work=t).sum(-1)
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
        form = _score_form(len(L_units), self.n_bins, L_units.shape[1])
        if lp_rows is not None and len(L_units) > self.SCORE_BATCH and (form == "rows" or _rs("score_sums") is not None):
            return self._score_in_batches(L_units, np.asarray(lp_rows, dtype=np.int64), form)
        Lp = self.marginal(L_units)
        Au = group_scores(Lp, self.log_pi[None])[:, 0, :]
        S, R = summed_scores(Lp, self.log_prof, rel=(self.eta if self.eta else 0.05, Au), form=form)
        Su_comm = Au.sum(1)
        Su = unknown_score(R, Su_comm, self.eta, self.lam)
        if lp_rows is not None:
            lp_rows = np.asarray(lp_rows, dtype=np.int64)
            Lp = Lp[lp_rows]
        return Scores(S, Su, Au, Lp, Su_comm, R, lp_rows)

    SCORE_BATCH = 8192  # units per batch of _score_in_batches (a multiple of marginalize_unit's 2048)

    def _score_in_batches(self, L_units, lp_rows: np.ndarray, form: str = "groups") -> Scores:
        """score() without the marginal tables of all units at once: they are a second copy of L_units (6.3 GB
        for the 34,878 contigs of ZymoFecal, and the peak of the run), of which only the rows lp_rows are kept.
        Used with the row form of summed_scores and with the scoring kernel of epirefine-rs, whose sums for a unit do
        not depend on the units scored with it (the BLAS products of the group form do, in the last bits)."""
        U, M, G = L_units.shape
        eta = self.eta if self.eta else 0.05
        S, R, Au = np.empty((U, self.n_bins)), np.empty((U, self.n_bins)), np.empty((U, M))
        kept = np.empty((len(lp_rows), M, G), L_units.dtype)
        for s in range(0, U, self.SCORE_BATCH):
            Lp = self.marginal(L_units[s:s + self.SCORE_BATCH])
            Au[s:s + len(Lp)] = group_scores(Lp, self.log_pi[None])[:, 0, :]
            S[s:s + len(Lp)], R[s:s + len(Lp)] = summed_scores(Lp, self.log_prof, rel=(eta, Au[s:s + len(Lp)]), form=form)
            a, b = np.searchsorted(lp_rows, [s, s + len(Lp)])
            kept[a:b] = Lp[lp_rows[a:b] - s]
        Su_comm = Au.sum(1)
        return Scores(S, unknown_score(R, Su_comm, self.eta, self.lam), Au, kept, Su_comm, R, lp_rows)

    def state_terms(self, sc: Scores, batch: int = 2000):
        """Per unit and motif group: log P(sites | the group is unmethylated) and log P(sites | methylated), the
        level drawn from the part of the population prior below / at or above 0.5, and that prior's methylated
        mass q[m]. A community draw is log((1 - q) exp(off) + q exp(on)) summed over groups (= Su_comm).
        """
        if sc.lp_rows is not None:
            raise ValueError("needs the Lp rows of every unit: score with lp_rows=None")
        up = self.grid.x[None, :] >= 0.5
        q = np.exp(logsumexp(np.where(up, self.log_pi, -np.inf), axis=-1))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        l_on = np.where(up, self.log_pi, -np.inf) - np.log(q)[:, None]
        l_off = np.where(~up, self.log_pi, -np.inf) - np.log1p(-q)[:, None]
        U, M, _ = sc.Lp.shape
        off, on = np.empty((U, M), np.float32), np.empty((U, M), np.float32)
        for s in range(0, U, batch):
            Lp = sc.Lp[s:s + batch].astype(np.float64)
            off[s:s + batch] = logsumexp(Lp + l_off[None], -1)
            on[s:s + batch] = logsumexp(Lp + l_on[None], -1)
        return off, on, q

    def unlisted_score(self, sc: Scores, batch: int = 2000) -> np.ndarray:
        """log P(unit | a genome with none of the listed motifs): every group's level drawn from the part of
        the population prior below 0.5 (the levels of bins that do not methylate the group)."""
        if sc.lp_rows is not None:
            raise ValueError("needs the Lp rows of every unit: score with lp_rows=None")
        off = np.where(self.grid.x[None, :] < 0.5, self.log_pi, -np.inf)
        off = off - logsumexp(off, axis=-1, keepdims=True)
        out = np.empty(len(sc.Lp))
        for s in range(0, len(out), batch):
            out[s:s + batch] = logsumexp(sc.Lp[s:s + batch].astype(np.float64) + off[None], -1).sum(-1)
        return out

    def host_terms(self, sc: Scores, rows: np.ndarray, hosts: np.ndarray | None = None,
                   log_prof: np.ndarray | None = None, batch: int = 1000) -> np.ndarray:
        """Per-group terms A[n, M] of units `rows`, each under one bin: hosts[i], or log_prof[i] if given."""
        A = np.empty((len(rows), sc.Lp.shape[1]))
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            prof = self.log_prof[hosts[s:s + batch]] if log_prof is None else log_prof[s:s + batch]
            A[s:s + batch] = logsumexp(sc.lp(r).astype(np.float64) + prof, -1)
        return A

    def chunk_llr(self, chunk_ids: np.ndarray, prof: np.ndarray) -> np.ndarray:
        """Per chunk: log P(chunk | community draw) - log P(chunk | bin profile `prof` [1, M, G])."""
        Lp = self.Lp_chunks[chunk_ids].astype(np.float64)
        return logsumexp(Lp + self.log_pi[None], -1).sum(-1) - logsumexp(Lp + prof, -1).sum(-1)

    def loo_profile(self, bin_idx: int, chunk_ids: np.ndarray) -> np.ndarray:
        ell = self.ell[bin_idx] - (self.w[chunk_ids, None, None] * self.Lp_chunks[chunk_ids]).sum(0)
        return host_profiles(ell[None], self.log_pi)

    def loo_profiles(self, bins, chunk_lists) -> np.ndarray:
        """loo_profile of many (bin, chunks) pairs at once: [K, M, G], row k the values loo_profile(bins[k],
        chunk_lists[k])[0] gives (the chunk sums contig by contig, the profiles for all rows in one step). One profile per call held the GIL for most of its time: on ZymoFecal the leave-one-out loops' 16
        threads spent 820 thread-seconds in 50,000 such calls."""
        ell = self.ell[np.asarray(bins, np.int64)]
        for k, ch in enumerate(chunk_lists):  # (per contig, as loo_profile: no more chunk terms in memory than it held)
            if len(ch):
                ell[k] -= (self.w[ch, None, None] * self.Lp_chunks[ch]).sum(0)
        lp = ell + self.log_pi[None, :, :]  # host_profiles' expression (its threaded form is for one large call)
        return lp - logsumexp(lp, axis=-1, keepdims=True)

    def loo_rescore_many(self, sc: Scores, rows, bins, profs: np.ndarray, which=None) -> np.ndarray:
        """loo_rescore for units rows[f] (distinct) of bins[f] against profiles profs[which[f]] ([K, M, G]; which
        None: profs[f]), in place on sc; each unit's values as loo_rescore gives them. Returns A [F, M]."""
        rows, bins = np.asarray(rows, np.int64), np.asarray(bins, np.int64)
        which = np.arange(len(rows)) if which is None else np.asarray(which, np.int64)
        # 8 units at a time: their [8, M, G] float64 temporaries stay small on every thread
        A = np.concatenate([logsumexp(sc.lp(rows[s:s + 8]).astype(np.float64) + profs[which[s:s + 8]], -1)
                            for s in range(0, len(rows), 8)]) if len(rows) else np.zeros((0, profs.shape[1]))
        sc.S[rows, bins] = A.sum(1)
        eta = self.eta if self.eta else 0.05
        sc.R[rows, bins] = np.logaddexp(np.log1p(-eta) + A, np.log(eta) + sc.Au[rows]).sum(1)
        sc.Su[rows] = unknown_score(sc.R[rows], sc.Su_comm[rows], self.eta, self.lam)
        return A

    def loo_rescore(self, sc: Scores, i: int, bin_idx: int, chunk_ids: np.ndarray, prof: np.ndarray | None = None):
        """Replace unit i's score for its own bin with a leave-chunks-out version (in place), and its
        score as a relative of that bin, so neither is built from the unit's own data.

        Returns the per-group terms A[m] of that leave-out score.
        """
        prof = self.loo_profile(bin_idx, chunk_ids) if prof is None else prof
        A = logsumexp(sc.lp(i).astype(np.float64) + prof[0], -1)
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
