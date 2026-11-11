"""Self-calibration with pseudo-contigs.

Binned contigs are cut into fragments of typical contig lengths. Each fragment is scored as if it
were unbinned, with every chunk of its source contig removed from its own bin's profile (leave the
contig out, as contig decisions do), so the true answer is known without external ground truth.
Removing only the chunks a fragment overlaps left the rest of its own contig in the profile (a
median 33% of it on ZymoFecal, 52% on human_gut): calibration then measured agreement within a
contig, saw no between-contig variation (tau = inf) and under-estimated the temperature.

Two tests:
  A  host present:  candidates are all bins; the right answer is the source bin.
  B  host absent:   the source bin is removed from the candidates; the right answer is "unknown".

Uses:
  * tau (between-unit variation) is chosen by maximising the LOO marginal likelihood of the
    fragments under their own host.
  * a temperature on the log-likelihoods is fitted to minimise log loss on A and B, to absorb
    leftover overconfidence (e.g. correlated sites).
  * accuracy / calibration / credible-set coverage are reported per fragment length, giving a
    sample-specific detection limit.

"""
from __future__ import annotations

import numpy as np
import polars as pl

from dataclasses import dataclass, field

from scipy.special import expit, logsumexp

from .assignability import margin_vs_competitors
from .model import LOO_THREADS_MIN, FittedModel, _n_threads, posterior, summed_scores, unknown_logit, unknown_score

from concurrent.futures import ThreadPoolExecutor

DEFAULT_LENGTHS = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
DEFAULT_TAUS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4, float("inf"))
DEFAULT_SIGMAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)  # logit-normal kernel
DEFAULT_ETAS = (0.02, 0.05)  # test: cap the per-group swap of a relative (eta 0.35 collapsed inclusion)
DEFAULT_LAMS = (0.1, 0.5, 0.9, 1.0)  # 1.0 = community-only unknown (the earlier model)
DEFAULT_TEMPS = tuple(np.round(np.exp(np.linspace(np.log(0.2), np.log(20), 31)), 3))
PRIOR_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)  # p_unknown values the prior-averaged objective is averaged over

# Calibration design, as a set of features (dev log proposal 2026-11-10, calibration coherence). The empty
# set is the calibration of 0.10 ("legacy"); every feature can be switched on alone, so each can be
# benchmarked against the others.
FEATURES = {
    "unit-temperature": "the temperature grid holds exactly 1, so 'never sharpen' means T >= 1 and not T >= 1.082",
    "temper-then-mix": "every explanation of 'unknown' (community draw, relative of each bin) is tempered, then "
                       "mixed with its prior weight, as the bins are",
    "prior-averaged": "(T, eta, lam) minimise the log loss averaged over priors p_unknown = 0.1 ... 0.9, each with "
                      "tests A and B weighted (1 - p, p): the likelihood is calibrated, whatever the prior",
    "prior-matched": "(T, eta, lam) are refitted at the estimated p_unknown, tests A and B weighted (1 - p, p), "
                     "until p_unknown and the fit agree",
    "sites-temperature": "the temperature may rise with a contig's number of sites (monotone, fitted per fragment "
                         "length, kept only when it beats one temperature by 2 SE)",
    "conservative-temperature": "among temperatures >= 1, the largest one that is not worse than the best by more "
                                "than 2 SE (bootstrap over source contigs): where the fragments cannot tell two "
                                "temperatures apart, the flatter one is used",
    "full-test-b": "the host-absent test counts every fragment, also those that cannot tell their bin from its "
                   "nearest competitor: unbinned contigs of a genome that resembles a bin are part of the "
                   "population the posteriors are reported for (needs a proper objective to make sense)",
    "common-tau": "tau values are compared on the fragments every tau fit keeps",
    "em-priors": "p_unknown has a Beta(2, 2) prior; the contamination rate counts only contigs that pass the "
                 "methylome-difference gate",
}
# The proposed set (rounds of the proposal in the dev log, 2026-11-10). Not in it, on the evidence of the
# development sets: unit-temperature (contig posteriors at exactly T = 1 are over-confident: 83% right at
# P 0.9-0.95 on the training communities), sites-temperature (worse than one temperature where its guard lets it
# act), prior-matched (same fits as prior-averaged, with a feedback loop between p_unknown and the fit),
# conservative-temperature (useful without full-test-b; with it the rule only costs inclusions).
COHERENT = ("temper-then-mix", "prior-averaged", "full-test-b", "common-tau", "em-priors")


def parse_features(spec) -> frozenset:
    """'legacy', 'coherent', or a comma-separated list of FEATURES (optionally added to 'coherent'/'legacy')."""
    if spec is None:
        return frozenset()
    if not isinstance(spec, str):
        spec = ",".join(spec)
    out: set[str] = set()
    for tok in (t.strip() for t in spec.split(",")):
        if tok in ("", "legacy"):
            continue
        if tok == "coherent":
            out |= set(COHERENT)
        elif tok.startswith("-") and tok[1:] in FEATURES:
            out.discard(tok[1:])
        elif tok in FEATURES:
            out.add(tok)
        else:
            raise ValueError(f"unknown calibration feature {tok!r}; choose from legacy, coherent, {', '.join(FEATURES)}")
    if {"prior-averaged", "prior-matched"} <= out:
        raise ValueError("prior-averaged and prior-matched exclude each other")
    return frozenset(out)


def fragments_per_bin(budget: int, n_lengths: int, n_bins: int, cap: int) -> int:
    """Fragments per (bin, length) for a total budget: calibration error depends on the number of
    fragments, not on the number of bins, so the total stays fixed as bins grow."""
    return int(max(1, min(cap, round(budget / max(1, n_lengths * n_bins)))))


def select_fragments(host_contigs: pl.DataFrame, lengths, n_per_bin: int, rng: np.random.Generator) -> pl.DataFrame:
    """Pseudo-contigs: up to n_per_bin random tiles of each length per bin.

    host_contigs: contig (any id type), bin_idx, length. Returns unit, contig, bin_idx, tile, length, start, end.
    """
    frag_rows, fid = [], 0
    hc = host_contigs.select("contig", "bin_idx", "length")
    for Lf in lengths:
        t = (hc.filter(pl.col("length") >= Lf).sort("bin_idx", "contig")
             .with_columns((pl.col("length") // Lf).alias("n_tiles")))
        if not t.height:
            continue
        # sample tiles uniformly within each bin without enumerating them: draw contigs in proportion to
        # their tile count, then a tile within the contig; duplicates are removed
        t = t.with_columns(pl.col("n_tiles").cum_sum().over("bin_idx").alias("cum"),
                           pl.col("n_tiles").sum().over("bin_idx").alias("tot"))
        draws = []
        for (b,), g in t.group_by("bin_idx", maintain_order=True):
            tot = int(g["tot"][0])
            k = min(n_per_bin, tot)
            pick = rng.choice(tot, size=k, replace=False) if tot <= 1_000_000 else np.unique(rng.integers(0, tot, k))
            ci = np.searchsorted(g["cum"].to_numpy(), pick, side="right")
            prev = np.r_[0, g["cum"].to_numpy()][ci]
            draws.append(pl.DataFrame({"contig": g["contig"].gather(ci), "bin_idx": np.full(len(pick), b),
                                       "tile": pick - prev}))
        tdf = pl.concat(draws).sort("bin_idx", "contig", "tile")
        tdf = tdf.with_columns((pl.int_range(pl.len()) + fid).alias("unit"), pl.lit(Lf).alias("length"))
        fid += tdf.height
        frag_rows.append(tdf.select("unit", "contig", "bin_idx", "tile", "length"))
    return pl.concat(frag_rows).with_columns(
        (pl.col("tile") * pl.col("length")).alias("start"), ((pl.col("tile") + 1) * pl.col("length")).alias("end"))


def contig_chunks(frag_contigs, chunk_contig: np.ndarray) -> list[np.ndarray]:
    """All host chunks of each fragment's source contig (chunk_contig[u] = contig id of chunk u)."""
    order = np.argsort(chunk_contig, kind="stable")
    ids, starts = np.unique(chunk_contig[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    of = {int(c): order[s:e] for c, s, e in zip(ids, starts, ends)}
    empty = np.empty(0, dtype=np.int64)
    return [of.get(int(c), empty).astype(np.int64) for c in frag_contigs]


def fragment_chunks(frags: pl.DataFrame, chunk_size: int, chunk_index) -> list[np.ndarray]:
    """Host chunks each fragment overlaps; chunk_index(contig, chunk number) -> unit or -1."""
    out = []
    for contig, s, e in frags.select("contig", "start", "end").iter_rows():
        ids = [chunk_index(contig, c) for c in range(s // chunk_size, (e - 1) // chunk_size + 1)]
        out.append(np.array([i for i in ids if i >= 0], dtype=np.int64))
    return out


def _metrics(P: np.ndarray, truth: np.ndarray, level: float = 0.95) -> dict:
    p_true = P[np.arange(len(truth)), truth]
    order = np.argsort(-P, axis=1)
    cum = np.cumsum(np.take_along_axis(P, order, 1), 1)
    set_size = (cum < level - 1e-12).sum(1) + 1
    in_set = np.array([truth[i] in order[i, :set_size[i]] for i in range(len(truth))])
    conf = P.max(1)
    correct = P.argmax(1) == truth
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {"n": len(truth), "accuracy": correct.mean(), "mean_confidence": conf.mean(),
            "ece": ece, "log_loss": -np.log(np.maximum(p_true, 1e-12)).mean(),
            "set95_coverage": in_set.mean(), "set95_mean_size": set_size.mean()}


def _reliability(P: np.ndarray, truth: np.ndarray, test: str) -> list[dict]:
    conf, correct = P.max(1), P.argmax(1) == truth
    edges = np.linspace(0, 1, 11)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if sel.any():
            rows.append({"test": test, "conf_lo": lo, "conf_hi": hi, "n": int(sel.sum()),
                         "mean_confidence": conf[sel].mean(), "accuracy": correct[sel].mean()})
    return rows


def loglik_on_common(own_ll: list[np.ndarray], keeps: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """Total own-bin log-likelihood per candidate, over the fragments every candidate kept: totals over
    each candidate's own kept set would reward the candidate that drops the hardest fragments."""
    common = np.logical_and.reduce(keeps)
    return common, [float(v[common].sum()) for v in own_ll]


def isotonic_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(map(float, y), map(float, w)):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v
            vals.pop(); wts.pop(); cnt.pop()
    return np.repeat(vals, cnt)


@dataclass
class TemperatureCurve:
    """Temperature as a non-decreasing function of a unit's number of sites: log T is interpolated linearly
    in log(sites) between the calibration lengths and held constant outside them (no fragment is longer
    than the longest calibration length, so nothing is known beyond it)."""
    log_sites: np.ndarray
    log_T: np.ndarray

    def __call__(self, n_sites) -> np.ndarray:
        return np.exp(np.interp(np.log(np.maximum(np.asarray(n_sites, float), 1.0)), self.log_sites, self.log_T))

    def table(self) -> list[dict]:
        return [{"sites": float(np.exp(a)), "temperature": float(np.exp(b))} for a, b in zip(self.log_sites, self.log_T)]


@dataclass
class FragmentFit:
    """Scores of the calibration fragments, enough to (re)fit the temperature and the unknown model for any
    prior p_unknown and to report how the deployed posteriors behave on them.

    All arrays are over the fragments with an independent profile to be scored against (`rest`); `kept`
    marks those the fit uses (not from chunks the robust model calls foreign to their bin).
    """
    S: np.ndarray            # [n, H] log P(fragment | bin), own bin leave-contig-out
    fbin: np.ndarray         # [n] source bin
    Su_comm: np.ndarray      # [n] community-draw score
    R: dict                  # eta -> [n, H] score as a relative of each bin (own bin leave-contig-out)
    kept: np.ndarray         # [n] bool: used by the fit
    distinct: np.ndarray     # [n] bool: test B is meaningful (bin decidable from its nearest competitor)
    length: np.ndarray       # [n] fragment length
    n_sites: np.ndarray      # [n] motif sites in the fragment
    difference: np.ndarray   # [n] methylome difference to the own bin (the gate statistic of contig decisions)
    contig: np.ndarray | None = None  # [n] source contig (fragments of one contig are not independent)
    features: frozenset = frozenset()
    temps: tuple = DEFAULT_TEMPS
    etas: tuple = DEFAULT_ETAS
    lams: tuple = DEFAULT_LAMS
    allow_sharpening: bool = False
    _cache: dict = field(default_factory=dict)

    @property
    def H(self) -> int:
        return self.S.shape[1]

    @property
    def coherent(self) -> bool:
        return "temper-then-mix" in self.features

    def _bins(self, T: float):
        """Per kept fragment at temperature T: own-bin logit, log sum over all bins, log sum over the others."""
        if T not in self._cache:
            k = np.flatnonzero(self.kept)
            Z = self.S[k] / T
            own = Z[np.arange(len(k)), self.fbin[k]]
            all_ = logsumexp(Z, axis=1)
            Z[np.arange(len(k)), self.fbin[k]] = -np.inf
            self._cache[T] = (own, all_, logsumexp(Z, axis=1) if self.H > 1 else np.full(len(k), -np.inf))
        return self._cache[T]

    def _unknown(self, T, eta, lam, rows=None, exclude_own=False):
        rows = np.arange(len(self.fbin)) if rows is None else rows
        R = self.R.get(eta)
        return unknown_logit(None if R is None else R[rows], self.Su_comm[rows], eta, lam, T,
                             exclude=self.fbin[rows] if exclude_own else None, temper_then_mix=self.coherent)

    def losses(self, T: float, eta, lam, p: float):
        """Log loss per kept fragment in test A (host present: P of the source bin) and, for the decidable
        ones, in test B (host absent: P of 'unknown'). With a proper objective the H - 1 remaining bins
        share 1 - p in test B; with the objective of 0.10 each keeps (1 - p) / H, as there."""
        k = np.flatnonzero(self.kept)
        key = ("u", T, eta, lam)
        if key not in self._cache:
            self._cache[key] = (self._unknown(T, eta, lam, k), self._unknown(T, eta, lam, k, exclude_own=True))
        zu_a, zu_b = self._cache[key]
        own, all_, others = self._bins(T)
        lpb, lpu = np.log1p(-p) - np.log(self.H), np.log(p)
        la = -(lpb + own - np.logaddexp(lpb + all_, lpu + zu_a))
        d = self.distinct[k]
        lpb1 = np.log1p(-p) - np.log(max(self.H - 1, 1) if self.proper else self.H)
        lb = -(lpu + zu_b[d] - np.logaddexp(lpb1 + others[d], lpu + zu_b[d]))
        cap = -np.log(1e-12)
        return np.minimum(la, cap), np.minimum(lb, cap)

    def objective_terms(self, T, eta, lam, priors):
        """Per-fragment contributions (A, B) to the objective: sum over (prior, weight A, weight B) of
        weight A * loss_A and weight B * loss_B at that prior."""
        la = lb = 0.0
        for p, wa, wb in priors:
            a, b = self.losses(T, eta, lam, p)
            la, lb = la + wa * a, lb + wb * b
        return la, lb

    def objective(self, T, eta, lam, priors) -> float:
        la, lb = self.objective_terms(T, eta, lam, priors)
        return float(la.mean() + (lb.mean() if len(lb) else 0.0))

    @property
    def proper(self) -> bool:
        """The objective is a proper score (prior = share of test B), not the objective of 0.10."""
        return bool(self.features & {"prior-averaged", "prior-matched"})

    def priors_for(self, p_unknown: float) -> tuple:
        """The objective as (prior p_unknown, weight of test A, weight of test B) triples."""
        if "prior-averaged" in self.features:
            return tuple((p, (1 - p) / len(PRIOR_GRID), p / len(PRIOR_GRID)) for p in PRIOR_GRID)
        p = float(np.clip(p_unknown, 1e-3, 1 - 1e-3))
        if "prior-matched" in self.features:
            return ((p, 1 - p, p),)
        return ((p, 1.0, 1.0),)   # the objective of 0.10: the prior as given, the tests weighted 1 : 1

    def fit(self, p_unknown: float, log=print) -> dict:
        """(temperature, eta, lam) minimising the proper log score of the fragments' posteriors.

        A share p of unbinned contigs comes from genomes without a bin (test B), 1 - p from binned genomes
        (test A), and the posterior is computed with that same prior p: the expected log score is then
        minimised by the true posterior, so the temperature measures how far the likelihood is off and
        does not also absorb a wrong prior. prior-matched uses p = p_unknown; prior-averaged averages the
        objective over PRIOR_GRID (a likelihood that is calibrated is calibrated under every prior).
        Without either, the objective is the one of 0.10 (prior p_unknown as passed, tests weighted 1 : 1).
        """
        priors = self.priors_for(p_unknown)
        temps = sorted(set(self.temps) | ({1.0} if "unit-temperature" in self.features else set()))
        ok_T = [T for T in temps if self.allow_sharpening or T >= 1.0]
        models = [(eta, lam) for eta in self.etas for lam in self.lams if lam >= 1 or (self.H > 1 and eta in self.R)]
        rows = [{"eta": eta, "lam": lam, "temperature": T, "log_loss": self.objective(T, eta, lam, priors)}
                for eta, lam in models for T in temps]
        fitted = min(rows, key=lambda r: r["log_loss"])
        fit = min((r for r in rows if r["temperature"] in ok_T), key=lambda r: r["log_loss"])
        simple = [r for r in rows if r["lam"] >= 1 and r["temperature"] in ok_T]
        if fit["lam"] < 1 and simple:  # parsimony (D28): relatives only when they beat community-only by 2 SE
            alt = min(simple, key=lambda r: r["log_loss"])
            gain, se = self._paired_gain((fit["temperature"], fit["eta"], fit["lam"]),
                                         (alt["temperature"], alt["eta"], alt["lam"]), priors)
            keep_flexible = gain > 2 * se
            log(f"  unknown model: relatives (eta={fit['eta']:g}, lam={fit['lam']:g}) vs community only: log-loss gain "
                f"{gain:.4f} +- {se:.4f} -> {'relatives' if keep_flexible else 'community only (not significant)'}")
            if not keep_flexible:
                fit = alt
        if "conservative-temperature" in self.features:
            # The fragments are cut from binned contigs and are easier to place than real unbinned contigs
            # (D7): they can show overconfidence but cannot justify confidence. So where they cannot tell
            # two temperatures apart, the flatter one is taken: the largest T whose objective is within 2 SE
            # of the best (paired differences, bootstrap over source contigs; 2 SE as in the parsimony rule).
            best = (fit["temperature"], fit["eta"], fit["lam"])
            chosen = fit
            for T in (t for t in ok_T if t > fit["temperature"]):
                gain, se = self._paired_gain((T, fit["eta"], fit["lam"]), best, priors, by_contig=True)
                if -gain > 2 * se:   # T is worse than the best by more than 2 SE
                    break
                chosen = next(r for r in rows if r["temperature"] == T and r["eta"] == fit["eta"] and r["lam"] == fit["lam"])
            if chosen is not fit:
                log(f"  temperature: best {fit['temperature']:g}; {chosen['temperature']:g} is within 2 SE of it "
                    f"(log loss {chosen['log_loss']:.4f} vs {fit['log_loss']:.4f}) -> {chosen['temperature']:g}")
            fit = chosen
        out = {"temperature": float(fit["temperature"]), "eta": float(fit["eta"]), "lam": float(fit["lam"]),
               "fitted_temperature": float(fitted["temperature"]), "temperature_fit": rows, "curve": None,
               "priors": priors}
        out["by_length"] = self.by_length(out, priors, temps)
        if "sites-temperature" in self.features:
            out["curve"] = self._curve(out, priors, ok_T, log)
        return out

    def _paired_gain(self, a, b, priors, n_boot: int = 200, by_contig: bool = False):
        """Objective(b) - objective(a) with a paired-bootstrap standard error over fragments, or over
        source contigs (by_contig: fragments of one contig share its departures from the bin, so resampling
        them one by one understates the error)."""
        la1, lb1 = self.objective_terms(*a, priors)
        la0, lb0 = self.objective_terms(*b, priors)
        dA, dB = la0 - la1, lb0 - lb1
        gain = dA.mean() + (dB.mean() if len(dB) else 0.0)
        rng = np.random.default_rng(0)
        if by_contig and self.contig is not None:
            k = np.flatnonzero(self.kept)
            _, inv = np.unique(self.contig[k], return_inverse=True)
            n = inv.max() + 1
            sA = np.bincount(inv, dA, n)
            sB = np.bincount(inv[self.distinct[k]], dB, n) if len(dB) else np.zeros(n)
            draws = rng.integers(0, n, (n_boot, n))
            boots = sA[draws].sum(1) / len(dA) + (sB[draws].sum(1) / len(dB) if len(dB) else 0.0)
            return float(gain), float(np.std(boots))
        boots = [dA[rng.integers(0, len(dA), len(dA))].mean()
                 + (dB[rng.integers(0, len(dB), len(dB))].mean() if len(dB) else 0.0) for _ in range(n_boot)]
        return float(gain), float(np.std(boots))

    def _class_objective(self, sel_k, T, eta, lam, priors):
        la, lb = self.objective_terms(T, eta, lam, priors)
        d = self.distinct[self.kept]
        sb = sel_k[d]
        return la[sel_k].mean() + (lb[sb].mean() if sb.any() else 0.0)

    def by_length(self, fit, priors, temps) -> list[dict]:
        """Diagnostic: the temperature each fragment length would want (any value of the grid)."""
        k = np.flatnonzero(self.kept)
        flen = self.length[k]
        rows = []
        for Lf in sorted(set(flen.tolist())):
            sel = flen == Lf
            best = min(temps, key=lambda T: self._class_objective(sel, T, fit["eta"], fit["lam"], priors))
            rows.append({"length": int(Lf), "fragments": int(sel.sum()), "temperature": float(best),
                         "median_sites": float(np.median(self.n_sites[k][sel]))})
        return rows

    def _curve(self, fit, priors, ok_T, log):
        """Temperature by number of sites: the best T >= 1 per fragment length, made non-decreasing in the
        length's median site count, used only when it beats the single temperature by more than 2 SE of the
        paired difference (as for the unknown model, D28: no extra freedom without evidence for it)."""
        k = np.flatnonzero(self.kept)
        flen, eta, lam = self.length[k], fit["eta"], fit["lam"]
        classes = sorted(set(flen.tolist()))
        if len(classes) < 2:
            return None
        med = np.array([np.median(self.n_sites[k][flen == Lf]) for Lf in classes])
        cnt = np.array([(flen == Lf).sum() for Lf in classes], float)
        raw = np.array([min(ok_T, key=lambda T: self._class_objective(flen == Lf, T, eta, lam, priors)) for Lf in classes])
        order = np.argsort(med, kind="stable")
        iso = np.exp(isotonic_increasing(np.log(raw[order]), cnt[order]))
        curve = TemperatureCurve(np.log(np.maximum(med[order], 1.0)), np.log(iso))
        if np.allclose(iso, fit["temperature"]):
            return None
        # paired comparison on the fragments: each at its class temperature (snapped to the grid) vs the scalar
        grid = np.array(ok_T)
        la1 = np.empty(len(k)); la0, lb0 = self.objective_terms(fit["temperature"], eta, lam, priors)
        d = self.distinct[k]
        lb1 = np.empty(int(d.sum()))
        pos_b = np.cumsum(d) - 1
        for j, Lf in enumerate(np.array(classes)[order]):
            Tj = float(grid[np.abs(np.log(grid) - np.log(iso[j])).argmin()])
            a, b = self.objective_terms(Tj, eta, lam, priors)
            sel = flen == Lf
            la1[sel] = a[sel]
            lb1[pos_b[sel & d]] = b[pos_b[sel & d]]
        dA, dB = la0 - la1, lb0 - lb1
        gain = dA.mean() + (dB.mean() if len(dB) else 0.0)
        rng = np.random.default_rng(0)
        boots = [dA[rng.integers(0, len(dA), len(dA))].mean()
                 + (dB[rng.integers(0, len(dB), len(dB))].mean() if len(dB) else 0.0) for _ in range(200)]
        se = float(np.std(boots))
        use = gain > 2 * se
        log("  temperature by sites: " + ", ".join(f"{int(np.exp(a))} sites {np.exp(b):.3g}" for a, b in zip(curve.log_sites, curve.log_T))
            + f"; log-loss gain over one temperature {gain:.4f} +- {se:.4f} -> {'used' if use else 'not used (not significant)'}")
        return curve if use else None

    # ---- posteriors of the fragments under given settings (reports)
    def posteriors(self, T, eta, lam, p: float, rows=None):
        """(Pa, Pb): [n, H + 1] posteriors with the host present and with it removed. T scalar or per row."""
        rows = np.arange(len(self.fbin)) if rows is None else rows
        mask = np.ones((len(rows), self.H), bool)
        mask[np.arange(len(rows)), self.fbin[rows]] = False
        Pa = posterior(self.S[rows], None, p, T, Zu=self._unknown(T, eta, lam, rows))
        Pb = posterior(self.S[rows], None, p, T, host_mask=mask, Zu=self._unknown(T, eta, lam, rows, exclude_own=True))
        return Pa, Pb

    def temperature_of(self, fit, rows=None):
        rows = np.arange(len(self.fbin)) if rows is None else rows
        return fit["curve"](self.n_sites[rows]) if fit.get("curve") is not None else fit["temperature"]

    def false_flags(self, fit, p_unknown: float, eps: float, threshold: float = 0.5, min_difference: float = 0.4) -> list[dict]:
        """How often would a true member be flagged? Every fragment is a member of its bin, scored without
        its source contig, so the share with P(foreign) >= threshold that also clears the methylome-
        difference gate estimates the false-flag rate of contig decisions at that length, under the
        deployed priors (p_unknown, contamination rate eps) and temperature. Two rows per length bracket the
        rate: `kept` leaves out the fragments from chunks the robust profiles call foreign to their bin (the
        model chooses its own exam questions: a lower bound), `all` counts them (an upper bound: where a
        bin really holds foreign contigs, their fragments are correctly flagged and counted here as false).
        """
        n = len(self.fbin)
        T = self.temperature_of(fit)
        Tc = T[:, None] if np.ndim(T) else T
        Z = self.S / Tc
        own = Z[np.arange(n), self.fbin].copy()
        Z[np.arange(n), self.fbin] = -np.inf
        zu = self._unknown(T, fit["eta"], fit["lam"])   # relative of the own bin: leave-contig-out, as in decisions
        alt = np.logaddexp(np.log1p(-p_unknown) - np.log(max(self.H - 1, 1)) + (logsumexp(Z, axis=1) if self.H > 1 else -np.inf),
                           np.log(p_unknown) + zu)
        pf = expit(np.log(eps) - np.log1p(-eps) + alt - own)
        gate = self.difference >= min_difference
        rows = []
        for Lf in sorted(set(self.length.tolist())) + ["all"]:
            sel = np.ones(n, bool) if Lf == "all" else self.length == Lf
            for name, s in (("all", sel), ("kept", sel & self.kept)):
                if not s.any():
                    continue
                flag = (pf[s] >= threshold)
                rows.append({"length": str(Lf), "fragments": name, "n": int(s.sum()),
                             "median_sites": float(np.median(self.n_sites[s])),
                             "mean_p_foreign": float(pf[s].mean()),
                             "rate_p_foreign_above_threshold": float(flag.mean()),
                             "rate_gate_passed": float(gate[s].mean()),
                             "false_flag_rate": float((flag & gate[s]).mean()),
                             "false_flags": int((flag & gate[s]).sum())})
        return rows

    def deployed_reliability(self, fit, p_unknown: float, eps: float, n_bins: int = 10) -> list[dict]:
        """Reliability of the deployed posteriors on the kept fragments, per fragment length: P(source bin)
        with the host present (test A), P(unknown) with it removed (test B, decidable fragments), and
        P(member) = 1 - P(foreign) of the fragment as a binned contig of its own bin."""
        k = np.flatnonzero(self.kept)
        T = self.temperature_of(fit, k)
        Pa, Pb = self.posteriors(T, fit["eta"], fit["lam"], p_unknown, k)
        ff = None
        rows = []
        flen = self.length[k]
        edges = np.linspace(0, 1, n_bins + 1)
        def add(test, Lf, conf, hit):
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (conf >= lo) & ((conf < hi) | (hi == 1.0))
                if s.any():
                    rows.append({"test": test, "length": str(Lf), "p_lo": lo, "p_hi": hi, "n": int(s.sum()),
                                 "mean_p": float(conf[s].mean()), "observed": float(hit[s].mean())})
        for Lf in sorted(set(flen.tolist())) + ["all"]:
            sel = np.ones(len(k), bool) if Lf == "all" else flen == Lf
            # test A: every candidate (bin or unknown) with its probability; observed = it is the truth
            Pa_s = Pa[sel]
            truth = np.zeros_like(Pa_s, bool); truth[np.arange(sel.sum()), self.fbin[k][sel]] = True
            add("A_any_candidate", Lf, Pa_s.ravel(), truth.ravel())
            add("A_map", Lf, Pa_s.max(1), Pa_s.argmax(1) == self.fbin[k][sel])
            sd = sel & self.distinct[k]
            if sd.any():
                add("B_unknown", Lf, Pb[sd, self.H], np.ones(int(sd.sum()), bool))
        return rows


def _by_bin(A_own: np.ndarray, keep: np.ndarray, fbin: np.ndarray, n_bins: int) -> np.ndarray:
    """[H, M]: the kept fragments' per-group leave-out terms, summed per source bin."""
    by_bin = np.zeros((n_bins, A_own.shape[1]))
    np.add.at(by_bin, fbin[keep], A_own[keep])
    return by_bin


def _pick_tau(group_ll: np.ndarray, group_ll_bin, base: int, tau_support: str) -> np.ndarray:
    """Index of the tau chosen for each motif group. group_ll[k, m]: leave-one-out log-likelihood of group m
    under tau k; group_ll_bin[k][h, m]: the same per source bin (needed for leave_bin_out only).

    "all": the tau with the highest total. "leave_bin_out": between-unit variation is a property of a motif
    group, shared by all bins. A finite tau that wins only through the fragments of one bin describes that bin
    (two methylomes in it: a contaminant's fragments scored against the host's profile), not the group, and it
    switches the group off as evidence in every bin. So each candidate's gain over "no variation" is counted
    without the bin that gains most."""
    if tau_support == "all":
        return group_ll.argmax(0)
    if tau_support != "leave_bin_out":
        raise ValueError(f"unknown tau_support {tau_support!r}")
    gain = np.array(group_ll_bin) - group_ll_bin[base][None]       # [n_taus, H, M]
    return (gain.sum(1) - gain.max(1)).argmax(0)


def calibrate(L_chunks, chunk_bin, n_bins, group_modtype, grid, frags: pl.DataFrame, L_frag,
              chunks_of, p_unknown, H_frag=None, taus=DEFAULT_TAUS, temps=DEFAULT_TEMPS,
              lookalike_dist=0.15, decidable_bf=3.0, etas=DEFAULT_ETAS, lams=DEFAULT_LAMS,
              allow_sharpening=False, n_sites_frag=None, leave_out=None, log=print, features=frozenset(),
              tau_support="all",              **model_kw):
    """chunks_of[i]: host chunks fragment i overlaps (to check its label); leave_out[i]: chunks removed
    from its bin's profile when scoring it (default chunks_of; the pipeline passes the whole source
    contig). features: see FEATURES (empty = the calibration of 0.10). The result's "fragment_fit" refits
    the temperature and the unknown model for another prior and reports on the deployed posteriors."""
    features = frozenset(features)
    fbin = frags["bin_idx"].to_numpy()
    H = n_bins
    leave_out = chunks_of if leave_out is None else leave_out
    bin_chunks = np.bincount(chunk_bin, minlength=H)
    # a fragment whose contig is its whole bin has no independent profile to be scored against
    has_rest = np.array([len(lo) < bin_chunks[b] for lo, b in zip(leave_out, fbin)])

    loo_keys = {int(i): (int(fbin[i]), np.asarray(leave_out[i]).tobytes()) for i in np.flatnonzero(has_rest)}
    loo_order = sorted(loo_keys, key=lambda i: (loo_keys[i], i))

    def own_and_scores(model):
        sc = model.score(L_frag)
        A_own = np.zeros(L_frag.shape[:2])
        # fragments of one contig leave the same chunks out of the same bin: one profile for all of them
        # (it was rebuilt for each: 4,450 fragments from 1,359 contigs on ZymoFecal); the contigs are
        # independent (each fragment writes its own rows) and run on threads
        by_key: dict = {}
        for i in loo_order:
            by_key.setdefault(loo_keys[i], []).append(i)

        def one_contig(frs):
            prof = model.loo_profile(fbin[frs[0]], leave_out[frs[0]])
            for i in frs:
                A_own[i] = model.loo_rescore(sc, i, fbin[i], leave_out[i], prof=prof)
        if L_frag.shape[1] * L_frag.shape[2] < LOO_THREADS_MIN:  # small tables: Python-bound, threads only contend
            for frs in by_key.values():
                one_contig(frs)
        else:
            with ThreadPoolExecutor(max(1, min(_n_threads(), len(by_key)))) as ex:
                list(ex.map(one_contig, by_key.values()))
        # fragments from chunks the robust model calls foreign to their bin carry a wrong label
        keep = has_rest & np.array([bool((model.w[ch] >= 0.5).all()) for ch in chunks_of])
        # the chunks' marginal tables (as large as L_chunks) were last needed by the leave-out profiles
        # above; dropped here, the best model so far no longer holds a second copy during the next fit
        model.Lp_chunks = None
        return sc, A_own, keep

    base = next((k for k, t in enumerate(taus) if grid.is_point(t)), len(taus) - 1)
    if "common-tau" not in features:
        tau_rows, best, warm, group_ll, group_ll_bin = [], None, {}, [], []
        for tau in taus:
            model = FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, tau, **warm, **model_kw)
            warm = {"init_w": model.w, "init_pi": model.raw_pi}  # not the model: at most two are held at once
            sc, A_own, keep = own_and_scores(model)
            ll = float(sc.S[keep, fbin[keep]].sum())
            group_ll.append(A_own[keep].sum(0))
            group_ll_bin.append(_by_bin(A_own, keep, fbin, H))
            tau_rows.append({"tau": float(tau), "label": f"{tau:g}", "loo_loglik": ll, "n_fragments": int(keep.sum())})
            log(f"  tau={tau:>8g}  LOO log-lik={ll:.1f}")
            if best is None or ll > best[0]:
                best = (ll, tau, sc, A_own, model, keep)
            del model
        # per-group tau: each group's own leave-one-out term is maximised separately (groups with
        # intermediate, variable methylation need a finite tau; groups at 0 or 1 an infinite one)
        group_ll = np.array(group_ll)  # [n_taus, M]
        pick = _pick_tau(group_ll, group_ll_bin, base, tau_support)
        tau_g = np.array([taus[k] for k in pick], dtype=float)
        if len(np.unique(tau_g)) > 1:
            model = FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, tau_g, **warm, **model_kw)
            sc, A_own, keep_g = own_and_scores(model)
            ll = float(sc.S[keep_g, fbin[keep_g]].sum())
            tau_rows.append({"tau": float("nan"), "label": "per-group", "loo_loglik": ll, "n_fragments": int(keep_g.sum())})
            log(f"  tau=per-group ({', '.join(f'{int((tau_g == t).sum())} groups at {t:g}' for t in np.unique(tau_g))})"
                f"  LOO log-lik={ll:.1f}")
            if ll > best[0]:
                best = (ll, tau_g, sc, A_own, model, keep_g)
            del model
    else:
        # every tau is scored on the same fragments: those no fit's robust weights call foreign. Each fit
        # drops the fragments it finds hardest (a large tau calls more chunks foreign), so totals over each
        # fit's own kept set favour large tau (ZymoFecal: tau = inf kept 2324 fragments, tau = 3 kept 2510,
        # and the 186 extra fragments were worth about as much log-likelihood as the whole spread between
        # taus; mono_train_1: 5,000 log units appear between tau = 100 and 300 from two fragments).
        tau_rows, warm, own_ll, own_A, keeps = [], {}, [], [], []
        held = None
        for k, tau in enumerate(taus):
            model = FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, tau, **warm, **model_kw)
            warm = {"init_w": model.w, "init_pi": model.raw_pi}
            sc, A_own, keep = own_and_scores(model)
            own_ll.append(sc.S[np.arange(len(fbin)), fbin].copy())
            own_A.append(A_own.astype(np.float32))
            keeps.append(keep)
            held = (k, sc, A_own, model)
            del model, sc, A_own
        common, lls = loglik_on_common(own_ll, keeps)
        for tau, ll, keep in zip(taus, lls, keeps):
            tau_rows.append({"tau": float(tau), "label": f"{tau:g}", "loo_loglik": ll, "n_fragments": int(common.sum()),
                             "n_kept_by_this_fit": int(keep.sum())})
            log(f"  tau={tau:>8g}  LOO log-lik={ll:.1f} ({int(common.sum())} common fragments; this fit keeps {int(keep.sum())})")
        k_best = int(np.argmax(lls))
        if held[0] == k_best:
            _, sc, A_own, model = held
        else:
            model = FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, taus[k_best], **warm, **model_kw)
            sc, A_own, _ = own_and_scores(model)
        del held
        best = (lls[k_best], taus[k_best], sc, A_own, model, common)
        del sc, A_own, model
        group_ll = np.array([A[common].sum(0, dtype=np.float64) for A in own_A])  # [n_taus, M]
        group_ll_bin = [_by_bin(A, common, fbin, H) for A in own_A] if tau_support != "all" else None
        del own_A
        pick = _pick_tau(group_ll, group_ll_bin, base, tau_support)
        tau_g = np.array([taus[k] for k in pick], dtype=float)
        if len(np.unique(tau_g)) > 1:
            model = FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, tau_g, **warm, **model_kw)
            sc, A_own, keep_g = own_and_scores(model)
            both = common & keep_g
            ll = float(sc.S[both, fbin[both]].sum())
            ll_single = float(own_ll[k_best][both].sum())
            tau_rows.append({"tau": float("nan"), "label": "per-group", "loo_loglik": ll, "n_fragments": int(both.sum()),
                             "n_kept_by_this_fit": int(keep_g.sum())})
            log(f"  tau=per-group ({', '.join(f'{int((tau_g == t).sum())} groups at {t:g}' for t in np.unique(tau_g))})"
                f"  LOO log-lik={ll:.1f} vs {ll_single:.1f} for tau={taus[k_best]:g} on the same {int(both.sum())} fragments")
            if ll > ll_single:
                best = (ll, tau_g, sc, A_own, model, both)
            del model
    tau_groups = pl.DataFrame({"group": np.arange(len(tau_g)), "tau": tau_g,
                               "loo_loglik_gain_vs_no_variation": group_ll[pick, np.arange(len(pick))] - group_ll[base]})
    _, tau, sc_best, A_own_best, best_model, keep = best
    if (~keep).any():
        log(f"  {int((~keep).sum())} fragments excluded: from chunks flagged foreign to their bin")
    fbin_all = fbin
    kidx = np.flatnonzero(keep)
    S, fbin, frags_k = sc_best.S[kidx], fbin[kidx], frags[kidx]
    Su_comm, Au, A_own_k = sc_best.Su_comm[kidx], sc_best.Au[kidx], A_own_best[kidx]
    H_frag_all = H_frag
    if H_frag is not None:
        H_frag = (H_frag[0][kidx], H_frag[1][kidx])

    mask_b = np.ones((len(fbin), H), bool)
    mask_b[np.arange(len(fbin)), fbin] = False
    truth_a, truth_b = fbin, np.full(len(fbin), H)

    # bins with a lookalike: another bin whose phi differs by < lookalike_dist in every group
    phi = grid.summarize(best_model.log_prof)[0]
    dist = np.abs(phi[:, None, :] - phi[None, :, :]).max(-1)
    np.fill_diagonal(dist, np.inf)
    has_lookalike = dist.min(1) < lookalike_dist
    f_look = has_lookalike[fbin]

    # Temperature: log loss on test A plus test B restricted to fragments that could tell their bin
    # from every other bin. Where a fragment cannot (a lookalike bin, or no sites in the motifs that
    # separate it from a sibling strain), "unknown" is unreachable for any method once its bin is
    # removed, and demanding it would only flatten everything; without B at all, evidence against
    # "unknown" is never checked (overconfident host calls). Decidable = expected log BF of the true
    # bin over its nearest competitor (Asimov, from site positions and coverage only) >= decidable_bf.
    decidable = np.ones(len(fbin), bool)
    margins = None  # expected log Bayes factors at temperature 1, reused for the assignability report below
    if H_frag is not None and H > 1:
        margins = margin_vs_competitors(best_model, *H_frag, fbin, 1.0)
        decidable = margins[0] >= decidable_bf
    distinct = ~f_look & decidable
    log(f"  test B uses {int(distinct.sum())} of {len(fbin)} fragments (bin decidable from its nearest competitor)")

    # Unknown host = community draw (weight lam) or an unbinned relative of a bin, which takes each
    # group's level from that bin with prob 1 - eta (weight 1 - lam). (eta, lam, T) are fitted together
    # on test A + decidable test B; in test B the removed bin is not offered as a relative either.
    def relatives(eta, rows=kidx):
        _, R = summed_scores(sc_best.Lp[rows], best_model.log_prof, rel=(eta, sc_best.Au[rows]))
        R[np.arange(len(rows)), fbin_all[rows]] = np.logaddexp(np.log1p(-eta) + A_own_best[rows],
                                                              np.log(eta) + sc_best.Au[rows]).sum(1)  # leave-out
        return R

    # everything a refit or a report on the deployed posteriors needs, for all fragments with an independent
    # profile (rest); the methylome difference is the gate statistic of contig decisions, computed the same way
    rest = np.flatnonzero(has_rest)
    obs = [grid.summarize(L_frag[rest[s:s + 2000]].astype(np.float64)) for s in range(0, len(rest), 2000)]
    lo95 = np.concatenate([o[1] for o in obs]) if obs else np.zeros((0, L_frag.shape[1]))
    hi95 = np.concatenate([o[2] for o in obs]) if obs else np.zeros((0, L_frag.shape[1]))
    pb = phi[fbin_all[rest]]
    has_sites = np.abs(L_frag[rest]).max(-1) > 0 if len(rest) else np.zeros((0, L_frag.shape[1]), bool)
    difference = np.where(has_sites, np.maximum(np.maximum(lo95 - pb, pb - hi95), 0.0), 0.0).max(1) if len(rest) else np.zeros(0)
    distinct_rest = np.zeros(len(rest), bool)
    # full-test-b: D12 left the undecidable fragments out of test B because, under the objective of 0.10,
    # they forced the temperature up (3.7); with a proper objective they cost what they should
    distinct_rest[np.searchsorted(rest, kidx)] = True if "full-test-b" in features else distinct
    ff = FragmentFit(S=sc_best.S[rest], fbin=fbin_all[rest], Su_comm=sc_best.Su_comm[rest],
                     R={eta: relatives(eta, rest) for eta in etas} if H > 1 else {},
                     kept=keep[rest], distinct=distinct_rest, length=frags["length"].to_numpy()[rest],
                     n_sites=(n_sites_frag[rest] if n_sites_frag is not None else np.zeros(len(rest))),
                     difference=difference, contig=frags["contig"].to_numpy()[rest], features=features, temps=tuple(temps), etas=tuple(etas), lams=tuple(lams),
                     allow_sharpening=allow_sharpening)

    def fragment_losses(T, Su_a, Su_b):
        """Per-fragment log loss in test A, and in test B over the decidable fragments."""
        Pa = posterior(S, Su_a, p_unknown, T)
        la = -np.log(np.maximum(Pa[np.arange(len(fbin)), truth_a], 1e-12))
        if not distinct.any():
            return la, np.empty(0)
        Tb = T[distinct] if np.ndim(T) else T
        Pb = posterior(S[distinct], Su_b[distinct], p_unknown, Tb, host_mask=mask_b[distinct])
        return la, -np.log(np.maximum(Pb[:, H], 1e-12))

    def logloss(T, Su_a, Su_b):
        la, lb = fragment_losses(T, Su_a, Su_b)
        if not len(lb):
            return la.mean(), la.mean(), np.nan
        return la.mean() + lb.mean(), la.mean(), lb.mean()

    flen = frags_k["length"].to_numpy()
    if not features - {"common-tau", "em-priors"}:  # no feature that touches the temperature / unknown-model fit
        # the calibration of 0.10: prior p_unknown as passed (0.1), tests A and B weighted 1 : 1, the unknown's
        # mixture summed before tempering
        temp_rows, Su_by = [], {}
        for eta in etas:
            R = relatives(eta) if H > 1 else None
            for lam in lams:
                if R is None and lam < 1:
                    continue
                Su_a = unknown_score(R, Su_comm, eta, lam) if R is not None else Su_comm
                Su_b = unknown_score(R, Su_comm, eta, lam, exclude=fbin) if R is not None else Su_comm
                Su_by[(eta, lam)] = (Su_a, Su_b)
                for T in temps:
                    temp_rows.append(dict(zip(("eta", "lam", "temperature", "log_loss", "log_loss_A", "log_loss_B_distinct"),
                                              (eta, lam, T, *logloss(T, Su_a, Su_b)))))
        fit = min(temp_rows, key=lambda r: r["log_loss"])
        fitted_temperature = fit["temperature"]
        # Never sharpen: pseudo-contigs are cut from binned contigs, which are easier to place than real
        # unbinned contigs; they can reveal overconfidence but not justify more confidence.
        if not allow_sharpening:
            fit = min((r for r in temp_rows if r["temperature"] >= 1.0), key=lambda r: r["log_loss"])
        # Parsimony: the community-only unknown (lam = 1) is kept unless a model with relatives fits
        # better by more than chance. The choice between them rests on ~1% of the log loss (ZymoFecal:
        # 1.641 vs 1.661), trading worse test A for better test B, and flipped with unrelated changes;
        # the flexible model then absorbed nearly all unbinned contigs (p_unknown 0.98-0.995; inclusions
        # 1-16 of the 380 that Hi-C places, against 42 of 42 correct with lam = 1).
        simple = [r for r in temp_rows if r["lam"] >= 1 and (allow_sharpening or r["temperature"] >= 1.0)]
        if fit["lam"] < 1 and simple:
            alt = min(simple, key=lambda r: r["log_loss"])
            la1, lb1 = fragment_losses(fit["temperature"], *Su_by[(fit["eta"], fit["lam"])])
            la0, lb0 = fragment_losses(alt["temperature"], *Su_by[(alt["eta"], alt["lam"])])
            dA, dB = la0 - la1, lb0 - lb1   # > 0: the flexible model is better on that fragment
            gain = dA.mean() + (dB.mean() if len(dB) else 0.0)
            brng = np.random.default_rng(0)
            boots = [dA[brng.integers(0, len(dA), len(dA))].mean()
                     + (dB[brng.integers(0, len(dB), len(dB))].mean() if len(dB) else 0.0) for _ in range(200)]
            se = float(np.std(boots))
            keep_flexible = gain > 2 * se
            log(f"  unknown model: relatives (eta={fit['eta']:g}, lam={fit['lam']:g}) vs community only: log-loss gain "
                f"{gain:.4f} +- {se:.4f} -> {'relatives' if keep_flexible else 'community only (not significant)'}")
            if not keep_flexible:
                fit = alt
        temperature, eta, lam = fit["temperature"], fit["eta"], fit["lam"]
        Su, Su_b = Su_by[(eta, lam)]
        # diagnostic: the temperature each fragment length would want. A temperature that rises with
        # length points to correlated sites (evidence over-counted more, the more sites a unit has)
        by_length = []
        for Lf in sorted(set(flen.tolist())):
            sel = flen == Lf

            def ll_len(T):
                Pa = posterior(S[sel], Su[sel], p_unknown, T)
                la = -np.log(np.maximum(Pa[np.arange(sel.sum()), truth_a[sel]], 1e-12)).mean()
                sd = sel & distinct
                if not sd.any():
                    return la
                Pb = posterior(S[sd], Su_b[sd], p_unknown, T, host_mask=mask_b[sd])
                return la - np.log(np.maximum(Pb[:, H], 1e-12)).mean()
            best_T = min(temps, key=ll_len)
            by_length.append({"length": int(Lf), "fragments": int(sel.sum()), "temperature": best_T,
                              "median_sites": float(np.median(n_sites_frag[kidx][sel])) if n_sites_frag is not None else float("nan")})
        first_fit = {"temperature": float(temperature), "eta": float(eta), "lam": float(lam), "curve": None,
                     "fitted_temperature": float(fitted_temperature), "priors": (p_unknown,)}
    else:
        first_fit = ff.fit(p_unknown, log=log)
        temperature, eta, lam = first_fit["temperature"], first_fit["eta"], first_fit["lam"]
        fitted_temperature = first_fit["fitted_temperature"]
        temp_rows, by_length = first_fit["temperature_fit"], first_fit["by_length"]
    log(f"  unknown model: eta={eta:g}, lam={lam:g}; temperature fitted {fitted_temperature:g} -> used {temperature:g}")
    log("  temperature by fragment length: " + ", ".join(f"{r['length'] // 1000} kb {r['temperature']:g}" for r in by_length))

    def report(fit, p):
        """Summary per fragment length and reliability, for the fragments' posteriors at prior p."""
        rows, rel = [], []
        kk = np.searchsorted(rest, kidx)
        for label, T in (("uncalibrated", 1.0), ("fitted_temperature", fit["fitted_temperature"]),
                         ("calibrated", ff.temperature_of(fit, kk))):
            Pa, Pb = ff.posteriors(T, fit["eta"], fit["lam"], p, kk)
            for Lf in sorted(frags_k["length"].unique().to_list()) + ["all"]:
                sel = np.ones(len(fbin), bool) if Lf == "all" else (flen == Lf)
                if not sel.any():
                    continue
                ma = _metrics(Pa[sel], truth_a[sel])
                mb = _metrics(Pb[sel], truth_b[sel])
                sd = sel & distinct  # host absent, and the fragment could tell its bin from all others
                rows.append({"scaling": label, "length": str(Lf), "n": ma["n"],
                             **{f"A_{k}": v for k, v in ma.items() if k != "n"},
                             "A_mean_p_unknown": Pa[sel, H].mean(),
                             "B_unknown_rate": mb["accuracy"],
                             "B_false_confident_rate": (Pb[sel, :H].max(1) >= 0.95).mean(),
                             "B_log_loss": mb["log_loss"],
                             "B_distinct_n": int(sd.sum()),
                             "B_distinct_unknown_rate": (Pb[sd].argmax(1) == H).mean() if sd.any() else np.nan,
                             "B_distinct_false_confident_rate": (Pb[sd, :H].max(1) >= 0.95).mean() if sd.any() else np.nan})
            if label == "calibrated":
                rel = _reliability(Pa, truth_a, "A") + _reliability(Pb, truth_b, "B")
        return pl.DataFrame(rows), pl.DataFrame(rel)

    summary, reliability = report(first_fit, p_unknown)
    # does the assignability score (computed without looking at methylation) predict accuracy?
    assign_rows, frag_extra = [], {}
    if H_frag is not None:
        kk = np.searchsorted(rest, kidx)
        Pa = ff.posteriors(ff.temperature_of(first_fit, kk), eta, lam, p_unknown, kk)[0]
        # the same tables, model and fragments as for test B above: the margins there (differences of scores
        # divided by 1) divided by the temperature are the values a second pass would compute, bit for bit
        if margins is None:
            margins = margin_vs_competitors(best_model, *H_frag, fbin, 1.0)
        margin, nearest, vs_unk = margins[0] / temperature, margins[1], margins[2] / temperature
        frag_extra = {"expected_log_bf_vs_nearest": margin, "expected_log_bf_vs_unknown": vs_unk,
                      "posterior_true": Pa[np.arange(len(fbin)), fbin], "map_correct": Pa.argmax(1) == fbin}
        edges = [-np.inf, 0.5, 1, 2, 3, 5, 10, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (margin >= lo) & (margin < hi)
            if sel.any():
                m = _metrics(Pa[sel], fbin[sel])
                assign_rows.append({"expected_log_bf_lo": lo, "expected_log_bf_hi": hi, "n": int(sel.sum()),
                                    "accuracy": m["accuracy"], "mean_confidence": m["mean_confidence"],
                                    "set95_mean_size": m["set95_mean_size"],
                                    "mean_observed_log_bf_true_vs_nearest": float(np.mean(
                                        (S[sel][np.arange(sel.sum()), fbin[sel]]
                                         - np.where(np.arange(H)[None, :] == fbin[sel][:, None], -np.inf, S[sel]).max(1))
                                        / temperature))})

    lookalikes = pl.DataFrame({"bin_idx": np.arange(H), "nearest_bin_idx": dist.argmin(1),
                               "max_phi_difference": dist.min(1), "has_lookalike": has_lookalike})
    return {"tau": tau, "eta": float(eta), "lam": float(lam), "temperature": float(temperature), "fitted_temperature": float(fitted_temperature), "summary": summary, "lookalikes": lookalikes,
            "tau_fit": pl.DataFrame(tau_rows), "tau_groups": tau_groups, "temperature_fit": pl.DataFrame(temp_rows),
            "temperature_by_length": pl.DataFrame(by_length),
            "reliability": reliability, "assignability": pl.DataFrame(assign_rows),
            "fragments": frags.with_columns(pl.Series("foreign_excluded", ~keep), **{
                k: pl.Series(_expand(v, kidx, len(keep))) for k, v in frag_extra.items()}), "model": best_model,
            "fragment_fit": ff, "fit": first_fit, "report": report}


def _expand(values, idx, n):
    out = np.full(n, np.nan)
    out[idx] = values
    return out
