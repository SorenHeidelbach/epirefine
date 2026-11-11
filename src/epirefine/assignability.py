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

from concurrent.futures import ThreadPoolExecutor

from .model import FittedModel, _cell_sums, _n_threads, _on_threads, _profile_exp, mix_log, score_units


_ARRAYS = ("arrays",)  # key of expected_tables' table under which it keeps its rows as arrays


def expected_tables(sites: pl.DataFrame, n_units: int, n_groups: int, det: dict[int, DetectionParams],
                    grid: Grid, n_cap: int = 500, dtype=np.float64, table: dict | None = None,
                    chunk_rows: int = 200_000, unit_batch: int | None = None, out=None):
    """sites: integer columns unit, group, n (optional count w). Returns H1, H0 with shape [U, M, G], or with
    unit_batch an iterator of (first unit, H1, H0) over consecutive batches of that many units (the same values;
    only one batch's tables are held). out: (H1, H0) buffers of unit_batch rows that the batches are written into
    (each batch overwrites the previous one: use it before taking the next).

    table: a dict reused across calls (the per-(group, n) expectations depend only on the caller
    model and the grid), so streaming many batches doesn't recompute them in Python each time.
    """
    # one integer key per (unit, group, n): its order is that of the (unit, group) cells with their rows in
    # ascending n (the order of the sum is then the same in every run, see model.unit_loglik_arrays); one key
    # column grouped and sorted instead of three
    wcol = pl.col("w").sum() if "w" in sites.columns else pl.len()
    s = (sites.select(((pl.col("unit").cast(pl.Int64) * n_groups + pl.col("group").cast(pl.Int64)) * (n_cap + 1)
                       + pl.col("n").clip(upper_bound=n_cap).cast(pl.Int64)).alias("key"),
                      *([pl.col("w")] if "w" in sites.columns else []))
         .group_by("key").agg(wcol.alias("w")).sort("key"))
    key = s["key"].to_numpy()
    w = s["w"].to_numpy().astype(float)
    del s
    flat = key // (n_cap + 1)
    # the distinct (group, n) of the rows in ascending order, and each row's index among them (integer keys:
    # as tuples in a Python dictionary this took seconds per part and about 1 GB of objects)
    codes, r = np.unique((flat % n_groups) * (n_cap + 1) + key % (n_cap + 1), return_inverse=True)
    del key
    keys = list(zip((codes // (n_cap + 1)).tolist(), (codes % (n_cap + 1)).tolist()))
    table = {} if table is None else table
    missing: dict[int, list[int]] = {}
    for g, n in sorted(keys, key=lambda k: (k[1], k[0])):
        if (g, n) not in table:
            missing.setdefault(n, []).append(g)

    def fill(task):
        # the expectations of a block of groups at one coverage n: for each group the operations of
        #     k = 0..n; l1, l0 = bb_logpmf(k, n, ...); p = exp(logc + l); term = logaddexp(l1 + logx, l0 + log1mx) (mix_log)
        #     table[(g, n)] = (p1 / p1.sum()) @ term, (p0 / p0.sum()) @ term
        # with the element-wise steps done for the whole block at once
        n, gs = task
        k = np.arange(n + 1, dtype=float)
        logc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        mu1, k1, mu0, k0 = (np.array([getattr(det[g], f) for g in gs])[:, None] for f in ("mu1", "kappa1", "mu0", "kappa0"))
        l1, l0 = bb_logpmf(k[None, :], n, mu1, k1), bb_logpmf(k[None, :], n, mu0, k0)
        p1, p0 = np.exp(logc[None, :] + l1), np.exp(logc[None, :] + l0)
        term = mix_log(l1.ravel(), l0.ravel(), grid).reshape(l1.shape + (grid.size,))
        # the weights of every group of the block at once; the products as before, one vector-matrix product
        # per group (the same BLAS call, the same bits)
        w1 = p1 / p1.sum(1, keepdims=True)
        w0 = p0 / p0.sum(1, keepdims=True)
        e1 = np.matmul(w1[:, None, :], term)[:, 0, :]
        e0 = np.matmul(w0[:, None, :], term)[:, 0, :]
        for i, g in enumerate(gs):
            table[(g, n)] = e1[i], e0[i]
    # A combination is (n + 1) x G logaddexp evaluations, about 0.5 ms: 150-200 s on one thread, one combination
    # at a time, for the 300,000 combinations of a sample with 700 groups (the largest single item of the
    # evidence step). Blocks of groups at the same n are independent and run on threads (numpy releases the GIL
    # in loops this long; one combination per call did not: the threads waited for each other).
    tasks = [(n, gs[i:i + 64]) for n, gs in sorted(missing.items(), reverse=True) for i in range(0, len(gs), 64)]
    threads = min(_n_threads(), len(tasks))
    if threads <= 1:
        for task in tasks:
            fill(task)
    else:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(fill, tasks))
    if keys and not tasks:
        # nothing was missing: the rows come from the whole table as arrays, stacked once per table (stacking each
        # call's rows from the dictionary took about a second per evidence part on ZymoFecal)
        arr = table.get(_ARRAYS)
        if arr is None or arr[3] != len(table) - 1 or arr[4] != n_cap:
            ks = sorted(k for k in table if k != _ARRAYS)
            arr = (np.array([g * (n_cap + 1) + n for g, n in ks], np.int64), np.stack([table[k][0] for k in ks]),
                   np.stack([table[k][1] for k in ks]), len(ks), n_cap)
            table[_ARRAYS] = arr
        at = np.searchsorted(arr[0], codes)
        T1, T0 = arr[1][at], arr[2][at]
    else:
        T1 = np.stack([table[k][0] for k in keys]) if keys else None
        T0 = np.stack([table[k][1] for k in keys]) if keys else None
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]]) if len(flat) else np.zeros(0, np.int64)
    cell = flat[starts]
    bounds = np.r_[starts, len(flat)]

    def tables(u0, u1):
        if out is not None and len(out[0]) >= u1 - u0:  # the caller's buffers (one batch at a time), cleared
            H1, H0 = out[0][:u1 - u0], out[1][:u1 - u0]
            H1[...] = 0
            H0[...] = 0
        else:
            H1 = np.zeros((u1 - u0, n_groups, grid.size), dtype)
            H0 = np.zeros_like(H1)
        c0, c1 = np.searchsorted(cell, [u0 * n_groups, u1 * n_groups])
        if c1 > c0:
            lo, hi = bounds[c0], bounds[c1]
            # reduced in blocks of whole (unit, group) cells of about chunk_rows rows: T[r] * w in one piece is
            # 8 G bytes per distinct (unit, group, n) row, 3.4 GB for the 6.6 million rows of one part of contigs
            for T, Ht in ((T1, H1), (T0, H0)):
                Ht.reshape(-1, grid.size)[cell[c0:c1] - u0 * n_groups] = _cell_sums(T, r[lo:hi], w[lo:hi],
                                                                                    starts[c0:c1] - lo, chunk_rows)
        return H1, H0

    if unit_batch is None:
        return tables(0, n_units)
    return ((u0, *tables(u0, min(u0 + unit_batch, n_units))) for u0 in range(0, n_units, unit_batch))


def asimov(H1, H0, phi, out=None):
    """phi: [U, M] true methylation level per unit and group. out: a buffer of at least len(H1) rows to write into
    (a fresh 0.7 GB array per batch of the evidence step otherwise)."""
    if H1.ndim != 3 or H1.size < 1_000_000:
        return phi[..., None] * H1 + (1 - phi[..., None]) * H0
    dt = np.result_type(H1, phi)
    out = np.empty(H1.shape, dt) if out is None or out.dtype != dt or len(out) < len(H1) else out[:len(H1)]

    def rows(sl):  # the same expression on unit slices (element-wise), on threads
        out[sl] = phi[sl, :, None] * H1[sl] + (1 - phi[sl, :, None]) * H0[sl]
    _on_threads(rows, len(H1))
    return out


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


def profile_exps(log_prof: np.ndarray):
    """(max over the grid [H, M], _profile_exp [H, M, G]) of a model's profiles, for own_scores: the same for every
    call on one model (22 million exponentials on ZymoFecal, about 0.2 s, once per evidence batch before)."""
    return log_prof.max(-1), _profile_exp(log_prof)


def own_scores(Lp: np.ndarray, log_prof: np.ndarray, t: np.ndarray, batch: int = 64, prof_exp=None) -> np.ndarray:
    """score_units(Lp, log_prof)[u, t[u]] without scoring the other bins: the same terms per group,
    log(exp(a - max a) . exp(b - max b)) + max a + max b, with the dot product taken for the unit's own bin only
    (H times less work; the sums run in another order than in the matrix product, so the score agrees with
    score_units to rounding, about 1e-15 relative)."""
    U = len(Lp)
    bm, eb = profile_exps(log_prof) if prof_exp is None else prof_exp   # [H, M], [H, M, G] (cells < 1e-250 as 0)
    out = np.empty(U)

    def rows(sl):
        for s in range(sl.start, sl.stop, batch):
            e_ = min(s + batch, sl.stop)
            x, tt = Lp[s:e_], t[s:e_]
            am = x.max(-1)
            q = np.einsum("umg,umg->um", np.exp(x - am[..., None]).astype(np.float64), eb[tt])
            np.log(np.maximum(q, 1e-300, out=q), out=q)
            q += am
            q += bm[tt]
            out[s:e_] = q.sum(1)
    _on_threads(rows, U)
    return out


def margin_vs_unknown(model: FittedModel, H1, H0, truth: np.ndarray, temperature: float, batch: int = 2000,
                      prof_exp=None, phi_h=None, buf=None):
    """(b) of margin_vs_competitors alone: the expected log BF of the true bin over unknown, with only the true
    bin scored (own_scores) instead of every bin."""
    U = len(truth)
    vs_unk = np.zeros(U)
    phi_h = phi_means(model) if phi_h is None else phi_h
    prof_exp = profile_exps(model.log_prof) if prof_exp is None else prof_exp
    for s in range(0, U, batch):
        sl = slice(s, s + batch)
        t = truth[sl]
        Lp = model.marginal(asimov(H1[sl], H0[sl], phi_h[t], out=buf))
        Su = score_units(Lp, model.log_pi[None])[:, 0]
        vs_unk[sl] = (own_scores(Lp, model.log_prof, t, prof_exp=prof_exp) - Su) / temperature
    return vs_unk


EVIDENCE_BATCH = 2000  # contigs per batch of the evidence tables and margins


def contig_evidence(parts: list[str], unit_of: dict[int, int], truth: np.ndarray, model: FittedModel,
                    det: dict[int, DetectionParams], grid: Grid, temperature: float, min_cov: int = 1,
                    workers: int = 1, group_weights: np.ndarray | None = None, drop_groups=(),
                    nearest: bool = False, log=None):
    """Expected methylation evidence and coverage per target contig.

    Evidence: the expected log BF (tempered) that the contig's sites would give for bin truth[u] over a
    random genome of the community, had the contig come from that bin (assignability, D12). Low
    values mean its sites can't show where it belongs, whatever their reads say: too few sites, too
    few reads, or a bin without a distinctive methylome. Coverage: median reads per motif site (all
    sites, before min_cov).

    parts: site-cache parquet files (contig, group, pos, n, k); unit_of: cache contig id -> target
    index. nearest=False leaves vs_nearest_bin NaN and scores each contig against its true bin only (the
    pipeline reads vs_unknown alone; scoring every bin was most of the step's time). Parts are processed one after another (BLAS uses the cores inside each): running parts
    on threads put 12 x 16 BLAS threads over OpenBLAS's limit and corrupted the heap (ZymoFecal,
    2026-10-27). Contigs split across parts are evaluated together at the end. Returns (vs_unknown [U], vs_nearest_bin [U], coverage [U]); NaN where truth < 0 or no sites.
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
    # the (group, n) expectation table, filled once for every combination present (Python loop),
    # then shared read-only by the threads
    shared: dict = {}
    keep = ~pl.col("group").is_in(list(drop_groups))
    combos = (pl.scan_parquet(parts).filter((pl.col("n") >= min_cov) & keep).select("group", pl.col("n").clip(upper_bound=500))
              .unique().collect())
    expected_tables(combos.with_columns(pl.lit(0).alias("unit")), 1, model.log_prof.shape[1], det, grid,
                    dtype=np.float32, table=shared)
    if log is not None:
        log(f"  evidence: expectation table of {len([k for k in shared if k != _ARRAYS])} (group, coverage) combinations")

    prof_exp, phi_h = profile_exps(model.log_prof), phi_means(model)   # once for all batches
    bufs: list = []

    def evaluate(sites: pl.DataFrame):
        sites = sites.filter(pl.col("contig").is_in(want))
        if not sites.height:
            return
        med = sites.group_by("contig").agg(pl.col("n").median())
        for c, m in med.iter_rows():
            cov[unit_of[c]] = m
        sites = sites.filter((pl.col("n") >= min_cov) & keep)
        contigs = [c for c in sites["contig"].unique().sort().to_list() if truth[unit_of[c]] >= 0]
        if not contigs:
            return
        units = np.array([unit_of[c] for c in contigs])
        inv = {c: i for i, c in enumerate(contigs)}
        s = (sites.filter(pl.col("contig").is_in(contigs))
             .select(pl.col("contig").replace_strict(inv, return_dtype=pl.Int64).alias("unit"), "group", "n"))
        # the tables of 2,000 contigs at a time (the batch of the margins), not of the whole part (1.5 GB for the
        # 4,125 contigs of a ZymoFecal part, held while the main process is at its peak)
        if not bufs:  # tables, Asimov table and their weights: reused by every batch (fresh 0.4-0.7 GB arrays before)
            shape = (EVIDENCE_BATCH, model.log_prof.shape[1], grid.size)
            bufs.extend([np.empty(shape, np.float32), np.empty(shape, np.float32), np.empty(shape, np.float64)])
        for u0, H1, H0 in expected_tables(s, len(units), model.log_prof.shape[1], det, grid, dtype=np.float32,
                                          table=shared, unit_batch=EVIDENCE_BATCH, out=(bufs[0], bufs[1])):
            if group_weights is not None:
                gw = np.asarray(group_weights, np.float32)[None, :, None]
                H1 *= gw   # (in place: the same float32 products)
                H0 *= gw
            ub = units[u0:u0 + len(H1)]
            if nearest:
                b, _, u = margin_vs_competitors(model, H1, H0, truth[ub], temperature, batch=EVIDENCE_BATCH)
                vs_bin[ub], vs_unk[ub] = b, u   # each contig is evaluated in exactly one call
            else:
                vs_unk[ub] = margin_vs_unknown(model, H1, H0, truth[ub], temperature, batch=EVIDENCE_BATCH,
                                               prof_exp=prof_exp, phi_h=phi_h, buf=bufs[2])
            del H1, H0

    done = [0]

    def one(f):
        d = pl.read_parquet(f, columns=["contig", "group", "n"])
        if split.len():
            held = d.filter(pl.col("contig").is_in(split))
            d = d.filter(~pl.col("contig").is_in(split))
        else:
            held = None
        evaluate(d)
        done[0] += 1
        if log is not None and done[0] % max(1, len(parts) // 4) == 0:
            log(f"  evidence: {done[0]}/{len(parts)} parts")
        return held

    with ThreadPoolExecutor(max(1, workers)) as ex:
        held = [h for h in ex.map(one, parts) if h is not None and h.height]
    if held:
        evaluate(pl.concat(held))
    return vs_unk, vs_bin, cov
