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

import dataclasses
from dataclasses import dataclass, field

from scipy.special import expit, logsumexp

from .assignability import margin_vs_competitors
from .model import LOO_BATCH, LOO_THREADS_MIN, FittedModel, _n_threads, log_bins, posterior, summed_scores, unknown_logit, unknown_score

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
    "bin-parsimony": "the unknown model with relatives (lam < 1) must beat the community-only model by 2 SE with the "
                     "standard error taken over bins (the fragments of a bin share its sibling bins), not over "
                     "single fragments",
    "tempered-bin-prior": "the uniform prior over bins is tempered with the evidence (each bin costs log(H) / T "
                          "instead of log(H)), in calibration and in the decisions, so the estimated p_unknown "
                          "no longer rises with the temperature",
    "deployed-unknown": "an unknown model with relatives (lam < 1) is kept only if it still beats the community-only "
                        "model by 2 SE when each is scored on the fragments at the p_unknown it would be deployed "
                        "with (EM on the unbinned contigs), not at the reference prior of the fit",
}
# The proposed set (rounds of the proposal in the dev log, 2026-11-10). Not in it, on the evidence of the
# development sets: unit-temperature (contig posteriors at exactly T = 1 are over-confident: 83% right at
# P 0.9-0.95 on the training communities), sites-temperature (worse than one temperature where its guard lets it
# act), prior-matched (same fits as prior-averaged, with a feedback loop between p_unknown and the fit),
# conservative-temperature (useful without full-test-b; with it the rule only costs inclusions).
COHERENT = ("temper-then-mix", "prior-averaged", "full-test-b", "common-tau", "em-priors")
# A named set ('consistent'; not the default): the features of the proposal that change almost no decision on
# the development sets and
# correct confirmed defects of the 0.10 calibration (the order of tempering and mixing, tau compared over
# different fragment sets, a flat prior on p_unknown, a contamination rate estimated before the gate).
CONSISTENT = ("temper-then-mix", "common-tau", "em-priors")
# The default since 0.11.1: the calibration of 0.10 with the parsimony test of the unknown model resampled over
# bins (dev log proposal 2026-11-12). 'legacy' is the calibration of 0.10 and 0.11.0 exactly.
DEFAULT = ("bin-parsimony",)


def _relatives_statistic(gain: float, se: float, relatives_row: dict, community_row: dict, best_is_relatives: bool,
                         over_bins: bool) -> dict:
    """The relatives test of 0.11 as a number (params.json relatives_statistic). gain: mean log-loss gain per
    calibration piece of the best unknown model with relatives (lam < 1) over the best community-only one (lam = 1),
    each at its own grid temperature; se: its standard error (resampled over bins with the feature bin-parsimony,
    the default). statistic = gain / se. The 2-SE switch of 0.11 chose relatives when the best row had relatives and
    statistic > 2; since option E (0.12) the switch is removed and the number is a report only. Large positive
    values say that some calibration pieces are explained better by an unbinned relative of a bin than by a random
    community genome: the sample may hold an unbinned genome with (nearly) the same methylation as a bin, whose
    contigs EpiRefine then places in that bin (development data: 4.4-5.4 on the simulated look-alikes, 0.4-1.9 on
    ZymoFecal)."""
    z = float(gain / se) if se > 0 else None   # None: no spread to divide by (a single bin or piece)
    return {"gain": float(gain), "se": float(se), "statistic": z, "se_over_bins": bool(over_bins),
            "relatives_eta": float(relatives_row["eta"]), "relatives_lam": float(relatives_row["lam"]),
            "relatives_temperature": float(relatives_row["temperature"]),
            "community_temperature": float(community_row["temperature"]),
            "switch_choice": "relatives" if best_is_relatives and gain > 2 * se else "community only"}


def calibration_guard(temperature, temps, n_pieces: int, n_test_b: int) -> dict:
    """The guard for calibrations that carry no usable temperature (0.12; dev log proposal 2026-11-13 calib_guard).

    It fires when the calibrated temperature is at the largest temperature of the grid (the cap, 20): no tempering
    the grid allows predicted the calibration pieces better, so the fit says nothing about how far the contig
    scores can be trusted. On development data the temperature is at most 5.9 on every normal run (monocultures,
    ZymoOral, simulated look-alikes, ZymoFecal and its bin resamples, fecal_inhouse) and either at most 1.7 or
    exactly 20 on the 653 stress-set-2 runs that finish; it never lies in between, and the runs at 20 are those where one
    calibration piece holds almost no sites or is confidently wrong, or where half of the methylation is knocked
    out. The numbers of pieces are reported but do not trigger the guard: from a single piece the calibration
    usually lands at the never-sharpen floor and the calls are right (stress set 2: 52 runs with calls from at most
    two pieces, 100% flag precision).
    The fallback is no calls (the pipeline sets the decision thresholds beyond 1)."""
    cap = float(max(temps))
    T = float(np.max(np.atleast_1d(temperature)))
    fired = T >= cap
    return {"fired": bool(fired), "reason": f"temperature at the grid cap ({cap:g})" if fired else None,
            "temperature": T, "cap": cap, "pieces": int(n_pieces), "test_b_pieces": int(n_test_b),
            "fallback": "no calls" if fired else None}


def clustered_se(dA, dB, cluster_a, cluster_b, n_boot: int = 400, seed: int = 0) -> float:
    """Standard error of mean(dA) + mean(dB) by a bootstrap that resamples whole clusters (the bins the fragments
    were cut from): the evidence for relatives is carried by sibling pairs of bins, a few of which hold most of
    the gain (ZymoFecal: 5 of 213 bins hold 44% of the test-B gain), so fragments are not independent draws.
    Each replicate takes the means over the fragments of the bins it drew."""
    cl = np.unique(np.concatenate([cluster_a, cluster_b]))
    ia, ib = np.searchsorted(cl, cluster_a), np.searchsorted(cl, cluster_b)
    n = len(cl)
    sA, nA = np.bincount(ia, dA, n), np.bincount(ia, minlength=n)
    sB, nB = np.bincount(ib, dB, n), np.bincount(ib, minlength=n)
    draws = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    a = sA[draws].sum(1) / np.maximum(nA[draws].sum(1), 1)
    b = sB[draws].sum(1) / np.maximum(nB[draws].sum(1), 1) if len(dB) else 0.0
    return float(np.std(a + b))


def parse_features(spec) -> frozenset:
    """'default' (DEFAULT, since 0.11.1), 'legacy' (the calibration of 0.10 and 0.11.0: no feature), 'consistent'
    (CONSISTENT), 'coherent' (COHERENT), or a comma-separated list of FEATURES, optionally after one of these names
    ('-name' removes a feature)."""
    if spec is None:
        return frozenset()
    if not isinstance(spec, str):
        spec = ",".join(spec)
    out: set[str] = set()
    for tok in (t.strip() for t in spec.split(",")):
        if tok in ("", "legacy"):
            continue
        if tok == "default":
            out |= set(DEFAULT)
            continue
        if tok == "coherent":
            out |= set(COHERENT)
        elif tok == "consistent":
            out |= set(CONSISTENT)
        elif tok.startswith("-") and tok[1:] in FEATURES:
            out.discard(tok[1:])
        elif tok in FEATURES:
            out.add(tok)
        else:
            raise ValueError(f"unknown calibration feature {tok!r}; choose from default, legacy, consistent, coherent, {', '.join(FEATURES)}")
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
        lpb, lpu = np.log1p(-p) - log_bins(self.H, T), np.log(p)
        la = -(lpb + own - np.logaddexp(lpb + all_, lpu + zu_a))
        d = self.distinct[k]
        lpb1 = np.log1p(-p) - log_bins(max(self.H - 1, 1) if self.proper else self.H, T)
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
        flexible = [r for r in rows if r["lam"] < 1 and r["temperature"] in ok_T]
        relatives_test = None
        if fit["lam"] < 1 and simple:  # parsimony (D28): relatives only when they beat community-only by 2 SE
            alt = min(simple, key=lambda r: r["log_loss"])
            gain, se = self._paired_gain((fit["temperature"], fit["eta"], fit["lam"]),
                                         (alt["temperature"], alt["eta"], alt["lam"]), priors,
                                         by_bin="bin-parsimony" in self.features)
            relatives_test = _relatives_statistic(gain, se, fit, alt, True, "bin-parsimony" in self.features)
            keep_flexible = gain > 2 * se
            log(f"  unknown model: relatives (eta={fit['eta']:g}, lam={fit['lam']:g}) vs community only: log-loss gain "
                f"{gain:.4f} +- {se:.4f} -> {'relatives' if keep_flexible else 'community only (not significant)'}")
            if not keep_flexible:
                fit = alt
        elif simple and flexible:  # community only fits best: the statistic is still reported (gain <= 0)
            rel = min(flexible, key=lambda r: r["log_loss"])
            gain, se = self._paired_gain((rel["temperature"], rel["eta"], rel["lam"]),
                                         (fit["temperature"], fit["eta"], fit["lam"]), priors,
                                         by_bin="bin-parsimony" in self.features)
            relatives_test = _relatives_statistic(gain, se, rel, fit, False, "bin-parsimony" in self.features)
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
               "priors": priors, "relatives_test": relatives_test}
        out["by_length"] = self.by_length(out, priors, temps)
        if "sites-temperature" in self.features:
            out["curve"] = self._curve(out, priors, ok_T, log)
        return out

    def _paired_gain(self, a, b, priors, n_boot: int = 200, by_contig: bool = False, priors_b=None, by_bin: bool = False):
        """Objective(b) - objective(a) with a paired-bootstrap standard error over fragments, or over
        source contigs (by_contig: fragments of one contig share its departures from the bin, so resampling
        them one by one understates the error). priors_b: b's objective at other priors than a's."""
        la1, lb1 = self.objective_terms(*a, priors)
        la0, lb0 = self.objective_terms(*b, priors if priors_b is None else priors_b)
        dA, dB = la0 - la1, lb0 - lb1
        gain = dA.mean() + (dB.mean() if len(dB) else 0.0)
        rng = np.random.default_rng(0)
        if by_bin:  # feature bin-parsimony: whole bins resampled
            k = np.flatnonzero(self.kept)
            return float(gain), clustered_se(dA, dB, self.fbin[k], self.fbin[k][self.distinct[k]] if len(dB) else self.fbin[:0])
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

    def deployed_gain(self, flexible, p_flexible: float, simple, p_simple: float) -> tuple[float, float]:
        """Log-loss gain of the unknown model with relatives over the community-only one, each scored on the
        fragments at the p_unknown it would be deployed with (feature deployed-unknown).

        flexible, simple: (temperature, eta, lam). p_flexible, p_simple: the EM estimates of p_unknown on the
        real unbinned contigs under each. The fit compares the two at a reference prior (0.1), but a model is
        deployed at the prior its own EM finds, and that prior depends on the model: a relative of bin h fits
        every contig that fits h, so wherever the relatives' likelihood ratio unknown : bins exceeds 1 - f (f the
        share of unbinned contigs that fit no bin) the EM runs to p_unknown = 1 and no contig can be placed
        (dev log, proposal 2026-11-10, R1.1; ZymoFecal with Nanomotif 1.3.0 motifs, 2026-11-12). The fragments are
        a fixed population, tests A (host present) and B (host absent) weighted 1 : 1 as in the fit of 0.10;
        only the prior of each posterior changes. Standard error by paired bootstrap over source contigs.
        Returns (gain, se); gain > 0: the relatives model is better at deployment."""
        clip = lambda p: float(np.clip(p, 1e-6, 1 - 1e-6))
        return self._paired_gain(flexible, simple, ((clip(p_flexible), 1.0, 1.0),), by_contig=True,
                                 priors_b=((clip(p_simple), 1.0, 1.0),))

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
              tau_support="all", exclude_foreign=True, **model_kw):
    """chunks_of[i]: host chunks fragment i overlaps (to check its label); leave_out[i]: chunks removed
    from its bin's profile when scoring it (default chunks_of; the pipeline passes the whole source
    contig). features: see FEATURES (empty = the calibration of 0.10). The result's "fragment_fit" refits
    the temperature and the unknown model for another prior and reports on the deployed posteriors.
    exclude_foreign=False (hidden diagnostic option, WP3 step 0): keep the fragments from chunks the robust model
    calls foreign to their bin, in every tau fit."""
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

        groups = list(by_key.values())
        batches = [groups[s:s + LOO_BATCH] for s in range(0, len(groups), LOO_BATCH)]

        def one_batch(batch):  # LOO_BATCH contigs: their profiles, then all their fragments, in whole-array steps
            profs = model.loo_profiles([fbin[frs[0]] for frs in batch], [leave_out[frs[0]] for frs in batch])
            rows = np.array([i for frs in batch for i in frs], np.int64)
            which = np.repeat(np.arange(len(batch)), [len(frs) for frs in batch])
            A_own[rows] = model.loo_rescore_many(sc, rows, fbin[rows], profs, which)
        if L_frag.shape[1] * L_frag.shape[2] < LOO_THREADS_MIN:  # small tables: Python-bound, threads only contend
            for batch in batches:
                one_batch(batch)
        else:
            with ThreadPoolExecutor(max(1, min(_n_threads(), len(batches)))) as ex:
                list(ex.map(one_batch, batches))
        # fragments from chunks the robust model calls foreign to their bin carry a wrong label
        keep = has_rest & np.array([bool((model.w[ch] >= 0.5).all()) for ch in chunks_of]) if exclude_foreign else has_rest.copy()
        # the chunks' marginal tables (as large as L_chunks) were last needed by the leave-out profiles
        # above; dropped here, the best model so far no longer holds a second copy during the next fit
        spare[:] = [model.Lp_chunks]  # (its memory is the next fit's marginal table: written over, not re-faulted)
        model.Lp_chunks = None
        return sc, A_own, keep

    spare: list = []

    def fit(tau_):
        return FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, tau_, **warm, **model_kw,
                           lp_out=spare.pop() if spare else None)

    base = next((k for k, t in enumerate(taus) if grid.is_point(t)), len(taus) - 1)
    if "common-tau" not in features:
        tau_rows, best, warm, group_ll, group_ll_bin = [], None, {}, [], []
        for tau in taus:
            model = fit(tau)
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
            model = fit(tau_g)
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
            model = fit(tau)
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
            model = fit(taus[k_best])
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
            model = fit(tau_g)
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
    spare.clear()  # no fit follows: the reused table is freed as before
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
    relatives_test = None   # the relatives statistic (_relatives_statistic); None: no model with relatives to test
    if not features - {"common-tau", "em-priors", "deployed-unknown", "bin-parsimony", "tempered-bin-prior"}:  # no feature that touches the temperature / unknown-model fit
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
        flexible = [r for r in temp_rows if r["lam"] < 1 and (allow_sharpening or r["temperature"] >= 1.0)]
        if simple and flexible:
            # the relatives test (the 2-SE switch of 0.11): best row with relatives against the best community-only
            # row. Computed on every run, also when community only fits best (gain <= 0 then), so that the statistic
            # can be reported as a number (relatives_statistic) whatever the unknown model used
            alt = min(simple, key=lambda r: r["log_loss"])
            rel = fit if fit["lam"] < 1 else min(flexible, key=lambda r: r["log_loss"])
            la1, lb1 = fragment_losses(rel["temperature"], *Su_by[(rel["eta"], rel["lam"])])
            la0, lb0 = fragment_losses(alt["temperature"], *Su_by[(alt["eta"], alt["lam"])])
            dA, dB = la0 - la1, lb0 - lb1   # > 0: the flexible model is better on that fragment
            gain = dA.mean() + (dB.mean() if len(dB) else 0.0)
            brng = np.random.default_rng(0)
            boots = [dA[brng.integers(0, len(dA), len(dA))].mean()
                     + (dB[brng.integers(0, len(dB), len(dB))].mean() if len(dB) else 0.0) for _ in range(200)]
            se = float(np.std(boots))
            if "bin-parsimony" in features:
                se = clustered_se(dA, dB, fbin, fbin[distinct] if len(dB) else fbin[:0])
            relatives_test = _relatives_statistic(gain, se, rel, alt, fit["lam"] < 1, "bin-parsimony" in features)
            if fit["lam"] < 1:
                keep_flexible = gain > 2 * se
                log(f"  unknown model: relatives (eta={fit['eta']:g}, lam={fit['lam']:g}) vs community only: log-loss gain "
                    f"{gain:.4f} +- {se:.4f}{' (over bins)' if 'bin-parsimony' in features else ''} -> "
                    f"{'relatives' if keep_flexible else 'community only (not significant)'}")
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
        relatives_test = first_fit.get("relatives_test")
    # the best community-only unknown model (lam = 1) at temperatures that may be used: the fallback when an unknown
    # model with relatives fails at its deployed prior (feature deployed-unknown, checked in pipeline.run)
    simple_rows = [r for r in temp_rows if r["lam"] >= 1 and (allow_sharpening or r["temperature"] >= 1.0)]
    community = min(simple_rows, key=lambda r: r["log_loss"]) if simple_rows else None
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
            "fragment_fit": ff, "fit": first_fit, "report": report, "relatives_test": relatives_test,
            "community": None if community is None else {"temperature": float(community["temperature"]),
                                                          "eta": float(community["eta"]), "lam": 1.0}}


def _expand(values, idx, n):
    out = np.full(n, np.nan)
    out[idx] = values
    return out


# ------------------------------------------------------------------------------------------------------------------
# Calibration core (mission WP3, dev log proposals/2026-11-12/wp3_designs): one function from the calibration pieces
# to the decisions before the redundancy check, called by pipeline.run and, with resampled pieces, by the R4 bootstrap.
# Every option's default is the behaviour of 0.11.1.

T_MIN = 1.082          # smallest temperature used: the grid's smallest value >= 1 (never sharpen, D7)
N_REF_SITES = 1000.0   # reference site count of the temperature law (log T = a + b log(n / N_REF_SITES))


@dataclass(frozen=True)
class CoreOptions:
    """The WP3 core options (CLI --tau-select, --npmle-start, --unknown-share, --temperature-law,
    --evidence-temperature). tau_select and npmle_start act upstream, in the model fits; they are kept here so a run
    records one set of options."""
    tau_select: str = "per-group"         # 'per-group' (0.11) | 'inf'
    npmle_start: str = "warm"             # 'warm' (0.11) | 'cold'
    unknown_share: str = "deployed"       # 'deployed' (EM at the deployed T, 0.11) | 't1' (EM at T = 1)
    temperature_law: str = "grid"         # 'grid' (one T from the 31-point grid, 0.11) | 'sites'
    evidence_temperature: str = "median"  # 'median' (0.11) | 'contig' (each contig's own T in the evidence cut-off)
    relatives: str = "switch"             # 'switch' (one model for all bins, 2-SE rule, 0.11) | 'per-bin' (weights)
    #                                       | 'slot' | 'community-only' (the switch removed: lam = 1 always)
    temper_then_mix: bool = False         # the unknown's explanations tempered, then mixed (feature temper-then-mix)
    slot_kappa: float | None = None       # relatives 'slot': the shrinkage strength (None: by cross-validation)
    slot_objective: str = "legacy"        # relatives 'slot': 'legacy' (prior 0.1, tests 1 : 1) | 'averaged' (PRIOR_GRID)
    slot_test_b: str = "decidable"        # relatives 'slot': test B on the decidable pieces (D12) | 'all'

    def __post_init__(self):
        for name, ok in (("tau_select", ("per-group", "inf")), ("npmle_start", ("warm", "cold")),
                         ("unknown_share", ("deployed", "t1")), ("temperature_law", ("grid", "sites")),
                         ("evidence_temperature", ("median", "contig")), ("relatives", ("switch", "per-bin", "slot", "community-only"))):
            if getattr(self, name) not in ok:
                raise ValueError(f"unknown {name} {getattr(self, name)!r}; choose from {', '.join(ok)}")

    @property
    def decisions_default(self) -> bool:
        """True when the decision stage is that of 0.11.1 (the upstream options may still differ)."""
        return ((self.unknown_share, self.temperature_law, self.evidence_temperature, self.relatives, self.temper_then_mix)
                == ("deployed", "grid", "median", "switch", False))


@dataclass
class SitesLaw:
    """Temperature as a smooth function of a unit's number of motif sites: T(n) = max(t_min, exp(a + b log(n / n_ref))),
    b in [0, 1]. On ZymoFecal (development) the best temperature rises from 3.1 at 2 kb to 13.8 at 500 kb and this
    form beats one temperature by 0.19 nats per piece, 5.6 SE over bins, cross-validated (dev log
    model_diagnostics.md section 4); on the look-alikes b is near 0."""
    a: float
    b: float
    n_ref: float = N_REF_SITES
    t_min: float = T_MIN

    def __call__(self, n_sites) -> np.ndarray:
        x = np.log(np.maximum(np.asarray(n_sites, float), 1.0) / self.n_ref)
        return np.maximum(self.t_min, np.exp(self.a + self.b * x))

    def table(self) -> list[dict]:
        return [{"law": "sites", "a": self.a, "b": self.b, "n_ref": self.n_ref, "t_min": self.t_min,
                 **{f"T_at_{int(n)}_sites": float(self(n)) for n in (300, 1000, 3000, 10000, 30000)}}]


def piece_losses(ff: FragmentFit, T_kept, eta, lam, p: float, weights: "RelativesWeights | None" = None):
    """FragmentFit.losses with one temperature per kept piece (T_kept [n_kept] or scalar): the same terms. weights:
    the unknown term is the per-bin relatives mixture (RelativesWeights.logit) instead of (eta, lam)."""
    k = np.flatnonzero(ff.kept)
    T = np.broadcast_to(np.asarray(T_kept, float), (len(k),)).astype(float)
    Z = ff.S[k] / T[:, None]
    rows = np.arange(len(k))
    own = Z[rows, ff.fbin[k]].copy()
    all_ = logsumexp(Z, axis=1)
    Z[rows, ff.fbin[k]] = -np.inf
    others = logsumexp(Z, axis=1) if ff.H > 1 else np.full(len(k), -np.inf)
    if weights is None:
        zu_a, zu_b = ff._unknown(T, eta, lam, k), ff._unknown(T, eta, lam, k, exclude_own=True)
    else:
        R = ff.R[weights.eta][k]
        zu_a = weights.logit(R, ff.Su_comm[k], T)
        zu_b = weights.logit(R, ff.Su_comm[k], T, exclude=ff.fbin[k])
    lpb, lpu = np.log1p(-p) - log_bins(ff.H, T), np.log(p)
    la = -(lpb + own - np.logaddexp(lpb + all_, lpu + zu_a))
    d = ff.distinct[k]
    lpb1 = np.log1p(-p) - log_bins(max(ff.H - 1, 1) if ff.proper else ff.H, T[d])
    lb = -(lpu + zu_b[d] - np.logaddexp(lpb1 + others[d], lpu + zu_b[d]))
    cap = -np.log(1e-12)
    return np.minimum(la, cap), np.minimum(lb, cap)


def fit_sites_law(ff: FragmentFit, eta, lam, priors, t_start: float, log=print, weights=None) -> SitesLaw:
    """(a, b) of the temperature law minimising the calibration objective (FragmentFit.objective with one T per
    piece) under the given unknown model. b is profiled on a grid in [0, 1] with a bounded 1-D search for a, then
    polished by L-BFGS-B; deterministic (no random start)."""
    from scipy.optimize import minimize, minimize_scalar
    k = np.flatnonzero(ff.kept)
    x = np.log(np.maximum(ff.n_sites[k], 1.0) / N_REF_SITES)

    def obj(a, b):
        T = np.maximum(T_MIN, np.exp(a + b * x))
        la = lb = 0.0
        for p, wa, wb in priors:
            pa, pb = piece_losses(ff, T, eta, lam, p, weights)
            la, lb = la + wa * pa, lb + wb * pb
        return float(np.mean(la) + (np.mean(lb) if np.size(lb) else 0.0))

    lo, hi = np.log(T_MIN) - 3.0, np.log(100.0)
    best = None
    for b in np.linspace(0.0, 1.0, 11):
        r = minimize_scalar(lambda a: obj(a, b), bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        if best is None or r.fun < best[0]:
            best = (float(r.fun), float(r.x), float(b))
    r = minimize(lambda v: obj(v[0], v[1]), x0=[best[1], best[2]], method="L-BFGS-B", bounds=[(lo, hi), (0.0, 1.0)],
                 options={"maxiter": 200})
    a, b, f = (float(r.x[0]), float(r.x[1]), float(r.fun)) if r.fun <= best[0] else (best[1], best[2], best[0])
    law = SitesLaw(a, b)
    log(f"  temperature law: log T = {a:.4f} + {b:.4f} log(sites / {N_REF_SITES:g}), T >= {T_MIN:g} "
        f"(T at 300 / 3,000 / 30,000 sites: {law(300):.3g} / {law(3000):.3g} / {law(30000):.3g}); objective {f:.5f} "
        f"against {obj(np.log(max(t_start, T_MIN)), 0.0):.5f} at the grid temperature {t_start:g}")
    return law


@dataclass
class ContigScores:
    """What the decisions read; fixed within a bootstrap. R: eta -> [N, H] relative-of-bin scores (only the etas
    available); Su: the unknown's score as the run mixed it (mix-then-temper at the model's eta, lam), used as it is
    on the 0.11 path."""
    S: np.ndarray
    Su_comm: np.ndarray
    R: dict
    given: np.ndarray
    testable: np.ndarray
    decidable: np.ndarray
    n_sites: np.ndarray
    Su: np.ndarray | None = None
    Su_eta_lam: tuple | None = None   # (eta, lam) Su was mixed with


@dataclass
class CalibrationResult:
    fit: dict                 # temperature (scalar, the grid value or the law's median over contigs), eta, lam, curve
    T_contig: object          # temperature used in the decisions: scalar or [N]
    p_unknown: float          # deployed unknown share
    p_unknown_t1: float | None
    refinement: object        # refine.Refinement (flags and inclusions before the redundancy check)
    law: SitesLaw | None = None

    @property
    def inclusions(self) -> int:
        r = self.refinement
        return int(((r.include_bin >= 0) & (self.contigs_given < 0)).sum())

    contigs_given: np.ndarray | None = None
    weights: "RelativesWeights | None" = None
    slot: "SlotModel | None" = None
    slot_extra: dict | None = None


def _su_for(contigs: ContigScores, eta, lam):
    """Mix-then-temper unknown score at T = 1 (0.11 order); None when R at eta is not available."""
    if lam >= 1 or eta is None:
        return contigs.Su_comm
    if contigs.Su is not None and contigs.Su_eta_lam == (eta, lam):
        return contigs.Su
    R = contigs.R.get(eta)
    return None if R is None else unknown_score(R, contigs.Su_comm, eta, lam)


def community_only(fit: dict, allow_sharpening: bool = False, log=print) -> dict:
    """--relatives community-only (WP3, option E of question U13): the unknown-model switch removed. The fit of
    FragmentFit.fit is replaced by its best community-only row (lam = 1) at a temperature that may be used, the row
    the 2-SE rule falls back to when relatives are not significant; a fit that already chose community only is
    returned unchanged. The relatives test itself is still run and logged by the fit (report only)."""
    if fit["lam"] >= 1:
        return fit
    simple = [r for r in fit["temperature_fit"] if r["lam"] >= 1 and (allow_sharpening or r["temperature"] >= 1.0)]
    alt = min(simple, key=lambda r: r["log_loss"])
    log(f"  unknown model forced to community only (--relatives community-only): the switch chose relatives "
        f"(eta={fit['eta']:g}, lam={fit['lam']:g}, T={fit['temperature']:g}); community only at T={alt['temperature']:g}")
    return dict(fit, temperature=float(alt["temperature"]), eta=float(alt["eta"]), lam=1.0, curve=None,
                switch_choice={"eta": fit["eta"], "lam": fit["lam"], "temperature": fit["temperature"]})


def calibrate_decisions(ff: FragmentFit | None, contigs: ContigScores, rules: dict, options: CoreOptions = CoreOptions(),
                        fit: dict | None = None, p_cal: float = 0.1, share_rows=None, log=print) -> CalibrationResult:
    """Calibration pieces in; model choice, temperature (law), unknown share and the decisions before the redundancy
    check out.

    ff: the calibration pieces (FragmentFit; the R4 bootstrap passes a copy resampled by source bins). fit: the
    unknown-model choice and grid temperature already made by calibrate() (the pipeline passes it); None refits them
    on ff exactly as FragmentFit.fit does at the reference prior p_cal (unknown-model switch with the 2-SE rule of
    the run's features, T on the grid). Then:
      temperature_law 'sites': the law is fitted on ff under the chosen unknown model (the T fit is redone because
        the unknown term moves T: ZymoFecal 5.024 community only, 1.715 with relatives);
      unknown_share 't1': p_unknown is the EM over the decided unbinned contigs at T = 1 (bins (1 - p) / H against
        the unknown at T = 1; at T = 1 the two orders of tempering and mixing agree), then the decisions use that p
        with the proper, untempered prior at each contig's temperature. 'deployed' (0.11): the EM runs inside refine
        at the deployed temperature, where the untempered 1/H prior makes p rise with T (0.41 at T 1, 0.69 at 5.02 on
        ZymoFecal, development).
    share_rows: rows of the unbinned contigs for the EM (the bootstrap resamples them); None = all decided ones.
    rules: refine() options (contamination_threshold, include_threshold, methylome_difference, ...; p_unknown when
    the user fixed it).
    """
    from .refine import em_weight, refine
    rules = dict(rules)
    p_user = rules.pop("p_unknown", None)
    thr_c = rules.pop("contamination_threshold", 0.5)
    thr_i = rules.pop("include_threshold", 0.9)
    if options.relatives == "slot":
        if options.unknown_share != "t1":
            raise ValueError("--relatives slot estimates the unknown share at T = 1: it needs --unknown-share t1")
        if ff is None or (fit is not None and fit.get("user_temperature") is not None):
            raise ValueError("--relatives slot needs the calibration and a fitted temperature")
        thr_c_, thr_i_ = rules.get("contamination_threshold", 0.5), rules.get("include_threshold", 0.9)
        rest = {k: v for k, v in rules.items() if k not in ("contamination_threshold", "include_threshold", "p_unknown")}
        model = fit_slot_model(ff, law_mode=options.temperature_law == "sites", p_cal=p_cal, kappa=options.slot_kappa,
                               objective=options.slot_objective, test_b=options.slot_test_b, log=log)
        ref, p_c, T, extra = slot_decide(contigs, model, rest, thr_c_, thr_i_, share_rows=share_rows, log=log)
        sfit = {"temperature": float(np.median(T)) if np.ndim(T) else float(T), "eta": model.eta,
                "lam": float(1.0 - np.mean(model.pi)), "curve": model.law, "fitted_temperature": float("nan")}
        res = CalibrationResult(fit=sfit, T_contig=T, p_unknown=ref.p_unknown, p_unknown_t1=p_c, refinement=ref,
                                law=model.law, contigs_given=contigs.given)
        res.slot, res.slot_extra = model, extra
        return res
    if ff is not None and options.temper_then_mix != ff.coherent:
        # the pieces are scored in the order of tempering and mixing the decisions use
        feats = ff.features | {"temper-then-mix"} if options.temper_then_mix else ff.features - {"temper-then-mix"}
        ff = dataclasses.replace(ff, features=frozenset(feats), _cache={})
    if fit is None:
        if ff is None:
            raise ValueError("calibrate_decisions: no calibration pieces and no fit")
        fit = ff.fit(p_cal, log=log)
        if options.relatives == "community-only":
            fit = community_only(fit, ff.allow_sharpening, log=log)
    elif options.relatives == "community-only" and fit["lam"] < 1:
        raise ValueError("--relatives community-only: the fit passed in has relatives (lam < 1)")
    fit = dict(fit)
    eta, lam = fit["eta"], fit["lam"]
    weights = None
    user_T = fit.get("user_temperature")
    if options.relatives == "per-bin":
        if ff is None:
            raise ValueError("--relatives per-bin needs the calibration")
        weights = fit_relatives_weights(ff, log=log)
        eta, lam = weights.eta, weights.lam
        fit.update(eta=eta, lam=lam, curve=None, relatives_weights=weights)
        if user_T is None and options.temperature_law == "grid":
            fit["temperature"] = grid_temperature(ff, ff.priors_for(p_cal), weights)
            log(f"  temperature under the per-bin relatives: {fit['temperature']:g}")
    law = None
    if options.temperature_law == "sites" and user_T is None:
        if ff is None:
            raise ValueError("--temperature-law sites needs the calibration")
        law = fit_sites_law(ff, eta, lam, ff.priors_for(p_cal), fit["temperature"], log=log, weights=weights)
        fit["curve"] = law
    T = law(contigs.n_sites) if law is not None else (fit["curve"](contigs.n_sites) if fit.get("curve") is not None
                                                      else fit["temperature"])
    if weights is not None:
        if weights.eta not in contigs.R:
            raise ValueError(f"calibrate_decisions: no contig scores for relatives at eta {weights.eta:g}")
        Rc = contigs.R[weights.eta]
        Su1 = weights.logit(Rc, contigs.Su_comm, 1.0)          # the unknown at T = 1 (both orders agree there)
        zu_of = lambda Tx, rows=slice(None): weights.logit(Rc[rows], contigs.Su_comm[rows], Tx)
    elif options.temper_then_mix and lam < 1:
        if eta not in contigs.R:
            raise ValueError(f"calibrate_decisions: no contig scores for relatives at eta {eta:g}")
        Rc = contigs.R[eta]
        Su1 = unknown_logit(Rc, contigs.Su_comm, eta, lam, 1.0, temper_then_mix=True)
        zu_of = lambda Tx, rows=slice(None): unknown_logit(Rc[rows], contigs.Su_comm[rows], eta, lam, Tx,
                                                           temper_then_mix=True)
    else:
        Su1 = _su_for(contigs, eta, lam)
        if Su1 is None:
            raise ValueError(f"calibrate_decisions: no contig scores for relatives at eta {eta:g}")
        zu_of = lambda Tx, rows=slice(None): Su1[rows] / Tx     # mix-then-temper, as 0.11
    p_t1 = None
    p_fixed = p_user
    prior = rules.get("p_unknown_prior", (1.0, 1.0))
    if options.unknown_share == "t1" and p_user is None:
        unb = np.flatnonzero((contigs.given < 0) & contigs.decidable) if share_rows is None else np.asarray(share_rows)
        l_bins = logsumexp(contigs.S[unb], axis=1) - np.log(contigs.S.shape[1])
        p_t1 = float(em_weight(l_bins, Su1[unb], *prior)[0])
        p_fixed = p_t1
        log(f"  unknown share by the EM at T = 1: {p_t1:.4f}")
    elif share_rows is not None and p_user is None:
        unb = np.asarray(share_rows)
        Tu = T[unb] if np.ndim(T) else T
        l_bins = logsumexp(contigs.S[unb] / (Tu[:, None] if np.ndim(Tu) else Tu), axis=1) - np.log(contigs.S.shape[1])
        p_fixed = float(em_weight(l_bins, zu_of(Tu, unb), *prior)[0])
    if weights is None and not (options.temper_then_mix and lam < 1):
        ref = refine(contigs.S, Su1, contigs.given, contigs.testable, T, thr_c, thr_i, p_unknown=p_fixed,
                     decidable=contigs.decidable, **rules)
    else:
        ref = refine(contigs.S, None, contigs.given, contigs.testable, T, thr_c, thr_i, p_unknown=p_fixed,
                     decidable=contigs.decidable, Zu=zu_of(T), **rules)
    if law is not None:
        fit["temperature"] = float(np.median(T))
    res = CalibrationResult(fit=fit, T_contig=T, p_unknown=ref.p_unknown, p_unknown_t1=p_t1, refinement=ref, law=law,
                            contigs_given=contigs.given)
    res.weights = weights
    return res


# ---- candidate 2 of WP3: a relatives weight per bin (design A, dev log wp3_designs/design_A.md section 2a)
ETA_REL = 0.05   # the eta every relatives fit chose on ZymoFecal and on the look-alikes (development)


@dataclass
class RelativesWeights:
    """The unknown genome's density as one mixture, no switch: a community draw (weight lam) or an unbinned relative
    of bin h (weight w[h]), lam + sum(w) = 1. alpha: the Dirichlet concentration that shrinks the spread over bins."""
    lam: float
    w: np.ndarray
    alpha: float
    eta: float = ETA_REL
    n_pieces: int = 0
    rounds: int = 0

    def logit(self, R, Su_comm, temperature, exclude=None) -> np.ndarray:
        """Tempered log-score of 'unbinned genome', each explanation tempered then mixed (temper-then-mix):
        log(lam exp(Su / T) + sum_h w_h exp(R_h / T)). exclude[u]: a bin left out (its own, in a host-absent test);
        the remaining weights are renormalised."""
        T = np.asarray(temperature, float)
        Tc = T[:, None] if T.ndim else T
        lw = np.log(np.maximum(self.w, 1e-300))
        Z = np.asarray(R, float) / Tc + lw[None, :]
        norm = np.zeros(len(Z))
        if exclude is not None:
            ex = np.asarray(exclude) >= 0
            rows = np.flatnonzero(ex)
            Z[rows, np.asarray(exclude)[ex]] = -np.inf
            norm[ex] = np.log(self.lam + self.w.sum() - self.w[np.asarray(exclude)[ex]])
        return np.logaddexp(np.log(self.lam) + np.asarray(Su_comm, float) / T, logsumexp(Z, axis=1)) - norm

    def summary(self) -> dict:
        H = len(self.w)
        rel = 1.0 - self.lam
        top = np.sort(self.w)[::-1]
        return {"lam": self.lam, "alpha": self.alpha, "eta": self.eta, "pieces": self.n_pieces, "rounds": self.rounds,
                "share_of_relatives_on_top10_bins": float(top[:10].sum() / rel) if rel > 0 else float("nan"),
                "bins_with_Hw_below_0.05": float(np.mean(H * self.w < 0.05)),
                "bins_with_Hw_at_least_0.2": float(np.mean(H * self.w >= 0.2))}


def _dm_alpha(m: np.ndarray) -> float:
    """Type-II ML of a symmetric Dirichlet-multinomial concentration from (soft) counts m [H]."""
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln
    H, M = len(m), float(m.sum())

    def nll(la):
        a = np.exp(la)
        return -(gammaln(H * a) - gammaln(M + H * a) + np.sum(gammaln(m + a) - gammaln(a)))
    r = minimize_scalar(nll, bounds=(-8.0, 4.0), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def fit_relatives_weights(ff: FragmentFit, eta: float = ETA_REL, tol: float = 1e-8, max_rounds: int = 500,
                          log=print) -> RelativesWeights:
    """Weights of the unknown mixture by EM on the host-absent calibration pieces (test B) at T = 1.

    Every kept piece of a binned genome, scored with its own bin removed, is a genome without a bin: its
    responsibilities are r_0 ~ lam exp(Su) (community draw) and r_h ~ w_h exp(R_h), h != its own bin (so a piece never
    informs the weight of its own bin: leave-one-bin-out by construction). lam = (m_0 + 1) / (n + 2) (Beta(1, 1)); the
    relatives' share is spread over bins as w_h = (1 - lam) (m_h + alpha) / (M + H alpha), alpha the type-II ML of the
    Dirichlet-multinomial on the soft counts: concentrated sibling evidence (ZymoFecal: 5 of 213 bins hold 44% of the
    test-B gain, D51) gives a small alpha and most bins near 0; evidence spread over the bins (look-alikes) gives a large
    alpha and weights near the uniform (1 - lam) / H of the global model. Deterministic."""
    k = np.flatnonzero(ff.kept)
    R = np.array(ff.R[eta][k], float)
    Su = ff.Su_comm[k].astype(float)
    b = ff.fbin[k]
    n, H = R.shape
    R[np.arange(n), b] = -np.inf
    mx = np.maximum(Su, R.max(1))
    E0 = np.exp(Su - mx)
    E = np.exp(R - mx[:, None])
    lam, w, alpha = 0.5, np.full(H, 0.5 / H), 1.0
    it = 0
    for it in range(1, max_rounds + 1):
        den = lam * E0 + E @ w
        m0 = float((lam * E0 / den).sum())
        mh = w * (E / den[:, None]).sum(0)
        new_lam = (m0 + 1.0) / (n + 2.0)
        alpha = _dm_alpha(mh)
        M = float(mh.sum())
        new_w = (1.0 - new_lam) * (mh + alpha) / (M + H * alpha)
        done = abs(new_lam - lam) < tol and np.abs(new_w - w).max() < tol
        lam, w = new_lam, new_w
        if done:
            break
    out = RelativesWeights(lam=float(lam), w=w, alpha=alpha, eta=eta, n_pieces=n, rounds=it)
    s = out.summary()
    log(f"  relatives per bin (eta {eta:g}): lam {lam:.4f}, alpha {alpha:.4g}, {s['share_of_relatives_on_top10_bins']:.1%} "
        f"of the relatives' weight on the top 10 bins, H w < 0.05 for {s['bins_with_Hw_below_0.05']:.1%} of bins "
        f"({n} pieces, {it} rounds)")
    return out


def grid_temperature(ff: FragmentFit, priors, weights: RelativesWeights) -> float:
    """Best temperature >= 1 of the grid under the per-bin unknown mixture (objective of FragmentFit.fit)."""
    best = None
    for T in ff.temps:
        if T < 1.0 and not ff.allow_sharpening:
            continue
        la = lb = 0.0
        for p, wa, wb in priors:
            a, b = piece_losses(ff, T, None, None, p, weights)
            la, lb = la + wa * a, lb + wb * b
        v = float(np.mean(la) + (np.mean(lb) if np.size(lb) else 0.0))
        if best is None or v < best[0]:
            best = (v, float(T))
    return best[1]


def hidden_relative(contigs: ContigScores, weights: RelativesWeights, p_unknown: float, T, best_bin) -> np.ndarray:
    """Per contig: the probability that an unbinned relative of `best_bin` explains it, among all explanations (bins
    with (1 - p) / H, the community draw p lam, the relatives p w_h; each likelihood tempered by the contig's T).
    Report only."""
    N, H = contigs.S.shape
    Tv = np.broadcast_to(np.asarray(T, float), (N,))
    R = contigs.R[weights.eta]
    rows = np.arange(N)
    lw = np.log(np.maximum(weights.w, 1e-300))
    l_bins = logsumexp(contigs.S / Tv[:, None], axis=1) + np.log1p(-p_unknown) - np.log(H)
    l_unk = np.log(p_unknown) + weights.logit(R, contigs.Su_comm, Tv)
    l_rel = np.log(p_unknown) + lw[best_bin] + R[rows, best_bin] / Tv
    return np.exp(l_rel - np.logaddexp(l_bins, l_unk))


# ---- candidate 3 of WP3: the slot model (design C, dev log wp3_designs/design_C.md)
SLOT_KAPPAS = (3.0, 10.0, 30.0, 100.0, float("inf"))
SLOT_U_BOUNDS = (-12.0, 12.0)   # logit pi_h


def _sigmoid_logs(u):
    """log pi, log(1 - pi) for pi = sigmoid(u)."""
    u = np.asarray(u, float)
    return -np.logaddexp(0.0, -u), -np.logaddexp(0.0, u)


def slot_piece_terms(S, R, Sc, fbin, distinct, T, u, p: float, need_grad: bool = True, proper: bool = False,
                     split: bool = False):
    """Losses and their gradient in logit pi for the slot model on calibration pieces (rows already selected).

    A piece of bin g is scored against: the community draw (prior p, exp(Sc / T)) and H slots ((1 - p) / H each), slot
    h holding h's own genome ((1 - pi_h) exp(S_h / T)) or an unbinned relative of h (pi_h exp(R_h / T)); every
    explanation tempered, then mixed. Test A: the right answer is g's own genome. Test B (g removed, the objective of
    0.10: the other bins keep (1 - p) / H): right = the community draw or a relative of any other bin. Returns
    (loss A [n], loss B [n_B], grad [H] of mean A + mean B)."""
    n, H = S.shape
    Tc = np.asarray(T, float).reshape(-1, 1) if np.ndim(T) else float(T)
    lpi, l1pi = _sigmoid_logs(u)
    pi = np.exp(lpi)
    c = np.log1p(-p) - np.log(H)
    d = np.log(p) + Sc / (Tc[:, 0] if np.ndim(T) else Tc)
    A = S / Tc + l1pi[None, :]
    B = R / Tc + lpi[None, :]
    slot = np.logaddexp(A, B)
    rows = np.arange(n)
    den = np.logaddexp(c + logsumexp(slot, axis=1), d)
    la = den - (c + A[rows, fbin])
    cap = -np.log(1e-12)
    okA = la < cap
    D = np.flatnonzero(distinct)
    A2, B2 = A[D].copy(), B[D].copy()
    A2[np.arange(len(D)), fbin[D]] = -np.inf
    B2[np.arange(len(D)), fbin[D]] = -np.inf
    cB = np.log1p(-p) - np.log(max(H - 1, 1)) if proper else c   # proper: the H - 1 remaining slots share 1 - p
    good = np.logaddexp(d[D], cB + logsumexp(B2, axis=1))
    bad = cB + logsumexp(A2, axis=1)
    tot = np.logaddexp(good, bad)
    lb = tot - good
    okB = lb < cap
    if not need_grad:
        return np.minimum(la, cap), np.minimum(lb, cap), None
    P_slot = np.exp(c + slot - den[:, None])
    rho = np.exp(B - slot)
    gA = P_slot * (rho - pi[None, :])
    gA[rows, fbin] += pi[fbin]
    gA[~okA] = 0.0
    eBt, eBg, eAt = np.exp(cB + B2 - tot[:, None]), np.exp(cB + B2 - good[:, None]), np.exp(cB + A2 - tot[:, None])
    gB = (1.0 - pi)[None, :] * (eBt - eBg) - pi[None, :] * eAt
    gB[~okB] = 0.0
    gA_mean = gA.sum(0) / max(n, 1)
    gB_mean = gB.sum(0) / len(D) if len(D) else np.zeros(H)
    if split:   # the gradients of mean A and of mean B separately
        return np.minimum(la, cap), np.minimum(lb, cap), (gA_mean, gB_mean)
    return np.minimum(la, cap), np.minimum(lb, cap), gA_mean + gB_mean


@dataclass
class SlotModel:
    """Fitted slot model: pi [H] (P(relative | slot h)), pi_0 and kappa of the shrinkage, the temperature (law or one
    value), eta of the relatives, and the CV table of kappa."""
    pi: np.ndarray
    pi0: float
    kappa: float
    law: SitesLaw | None
    T: float | None
    eta: float = ETA_REL
    objective: float = float("nan")
    rounds: int = 0
    cv: list = field(default_factory=list)

    def temperature(self, n_sites):
        return self.law(n_sites) if self.law is not None else self.T

    def unknown_logit(self, R, Sc, T, p_c: float) -> np.ndarray:
        """Zu with p_c exp(Zu) = p_c exp(Sc / T) + (1 - p_c) / H sum_h pi_h exp(R_h / T): the unknown term of refine()
        (community draw plus every bin's relative) at prior p_c."""
        Tv = np.asarray(T, float)
        Tc = Tv[:, None] if Tv.ndim else Tv
        H = R.shape[1]
        rel = logsumexp(R / Tc + np.log(np.maximum(self.pi, 1e-300))[None, :], axis=1)
        return np.logaddexp(Sc / Tv, np.log1p(-p_c) - np.log(p_c) - np.log(H) + rel)

    def summary(self) -> dict:
        return {"pi0": self.pi0, "kappa": self.kappa, "eta": self.eta, "rounds": self.rounds,
                "objective": self.objective, "pi_median": float(np.median(self.pi)),
                "bins_pi_at_least_0.15": float(np.mean(self.pi >= 0.15)), "bins_pi_at_least_0.3": int((self.pi >= 0.3).sum()),
                **({"law_a": self.law.a, "law_b": self.law.b} if self.law is not None else {"T": self.T})}


def _slot_objective(arr5, T, u, priors, need_grad=True, sums=False):
    """The calibration objective of the slot model: sum over (prior p, weight A, weight B) of wA mean(loss A) +
    wB mean(loss B); legacy = ((0.1, 1, 1),) with the 0.10 test B, averaged = PRIOR_GRID with ((1 - p) / 5, p / 5) and
    the proper test B. sums=True returns the summed (not mean) losses for cross-validation."""
    S, R, Sc, fb, dist = arr5
    proper = len(priors) > 1
    val, grad = 0.0, 0.0
    for p, wa, wb in priors:
        la, lb, g = slot_piece_terms(S, R, Sc, fb, dist, T, u, p, need_grad=need_grad, proper=proper, split=True)
        if sums:
            val += wa * la.sum() + wb * lb.sum()
        else:
            val += wa * la.mean() + (wb * lb.mean() if len(lb) else 0.0)
        if need_grad:
            grad = grad + wa * g[0] + wb * g[1]
    return float(val), grad


def _slot_arrays(ff: FragmentFit, eta: float, rows=None):
    k = np.flatnonzero(ff.kept) if rows is None else rows
    return (ff.S[k], ff.R[eta][k], ff.Su_comm[k], ff.fbin[k], ff.distinct[k],
            np.log(np.maximum(ff.n_sites[k], 1.0) / N_REF_SITES))


def _fit_u(arr, T, priors, u0, kappa, pi0, pooled: bool):
    """pi given the temperature: L-BFGS-B on logit pi (one shared value when pooled) with the pseudo-count penalty."""
    from scipy.optimize import minimize
    N, H = len(arr[0]), arr[0].shape[1]

    def f(v):
        u = np.full(H, v[0]) if pooled else v
        val, g = _slot_objective(arr[:5], T, u, priors)
        if not pooled and np.isfinite(kappa):
            pi = 1.0 / (1.0 + np.exp(-u))
            lpi, l1pi = _sigmoid_logs(u)
            val -= float(kappa * (pi0 * lpi + (1 - pi0) * l1pi).sum()) / N
            g = g - (kappa / N) * (pi0 - pi)
        return val, (np.array([g.sum()]) if pooled else g)
    x0 = np.array([float(np.mean(u0))]) if pooled else np.asarray(u0, float)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[SLOT_U_BOUNDS] * len(x0), options={"maxiter": 500})
    return (np.full(H, r.x[0]) if pooled else r.x), float(r.fun)


def _fit_T(arr, u, priors, law_mode: bool, start, temps):
    """The temperature given pi: the law (a, b) by L-BFGS-B from `start`, or the best grid value >= 1."""
    from scipy.optimize import minimize
    x = arr[5]

    def obj_T(T):
        return _slot_objective(arr[:5], T, u, priors, need_grad=False)[0]
    if not law_mode:
        vals = [(obj_T(T), float(T)) for T in temps if T >= 1.0]
        return min(vals)[1], min(vals)[0]
    lo, hi = np.log(T_MIN) - 3.0, np.log(100.0)
    f = lambda v: obj_T(np.maximum(T_MIN, np.exp(v[0] + v[1] * x)))
    r = minimize(f, x0=[start.a, start.b], method="L-BFGS-B", bounds=[(lo, hi), (0.0, 1.0)], options={"maxiter": 100})
    return SitesLaw(float(r.x[0]), float(r.x[1])), float(r.fun)


def _fit_slot(arr, priors, kappa, pi0, law_mode, temps, u0, T0, pooled=False, max_rounds=20, tol=1e-6, T_fixed=None):
    u, T, prev, it = np.asarray(u0, float), T0, np.inf, 0
    x = arr[5]
    for it in range(1, max_rounds + 1):
        if T_fixed is None:
            T, _ = _fit_T(arr, u, priors, law_mode, T, temps)
        Tv = (T(np.exp(x) * N_REF_SITES) if law_mode else T) if T_fixed is None else T_fixed
        u, val = _fit_u(arr, Tv, priors, u, kappa, pi0, pooled)
        if abs(prev - val) < tol or T_fixed is not None:
            break
        prev = val
    return u, T, val, it


def fit_slot_model(ff: FragmentFit, law_mode: bool = True, p_cal: float = 0.1, kappa=None, eta: float = ETA_REL,
                   folds: int = 5, objective: str = "legacy", test_b: str = "decidable", log=print) -> SlotModel:
    """The slot model fitted on the calibration pieces: pi_0 by the pooled fit (kappa = inf), kappa by 5-fold CV over
    source contigs (folds stratified by bin, the temperature held at the pooled full-data fit), then pi_h and the
    temperature jointly at the chosen kappa (alternating; deterministic starts: pi = 0.02, T from the pooled fit)."""
    arr = _slot_arrays(ff, eta)
    if test_b == "all":   # design C: every kept piece in test B (the undecidable ones are the sibling evidence)
        arr = arr[:4] + (np.ones(len(arr[0]), bool),) + arr[5:]
    elif test_b != "decidable":
        raise ValueError(f"unknown slot test_b {test_b!r}")
    H = arr[0].shape[1]
    u0 = np.full(H, np.log(0.02 / 0.98))
    if objective not in ("legacy", "averaged"):
        raise ValueError(f"unknown slot objective {objective!r}")
    priors = ((p_cal, 1.0, 1.0),) if objective == "legacy" else tuple(
        (p, (1 - p) / len(PRIOR_GRID), p / len(PRIOR_GRID)) for p in PRIOR_GRID)
    if law_mode:
        T_start = SitesLaw(float(np.log(3.0)), 0.0)
    else:
        T_start = 3.0
    u_pool, T_pool, val_pool, it_pool = _fit_slot(arr, priors, np.inf, 0.0, law_mode, ff.temps, u0, T_start, pooled=True)
    pi0 = float(1.0 / (1.0 + np.exp(-u_pool[0])))
    cv_rows = []
    if kappa is None:
        k = np.flatnonzero(ff.kept)
        contig = ff.contig[k] if ff.contig is not None else np.arange(len(k))
        fold = np.zeros(len(k), int)
        for b in np.unique(ff.fbin[k]):           # contigs of each bin dealt round-robin in sorted order
            cs = np.unique(contig[ff.fbin[k] == b])
            fid = {c: j % folds for j, c in enumerate(cs)}
            sel = np.flatnonzero(ff.fbin[k] == b)
            fold[sel] = [fid[c] for c in contig[sel]]
        x = arr[5]
        Tv_pool = T_pool(np.exp(x) * N_REF_SITES) if law_mode else T_pool
        for kap in SLOT_KAPPAS:
            tot, n_tot = 0.0, 0
            for f in range(folds):
                tr, te = fold != f, fold == f
                sub = lambda a: tuple(v[tr] for v in a)
                Ttr = Tv_pool[tr] if law_mode else Tv_pool
                u_f, _ = _fit_u(sub(arr), Ttr, priors, u_pool if np.isinf(kap) else u_pool.copy(), kap, pi0, np.isinf(kap))
                Tte = Tv_pool[te] if law_mode else Tv_pool
                v_te, _ = _slot_objective(tuple(v[te] for v in arr[:5]), Tte, u_f, priors, need_grad=False, sums=True)
                tot += v_te
                n_tot += int(te.sum())
            cv_rows.append({"kappa": kap, "cv_log_loss_per_piece": tot / max(n_tot, 1)})
        kappa = min(cv_rows, key=lambda r: r["cv_log_loss_per_piece"])["kappa"]
    if np.isinf(kappa):
        u, T, val, it = u_pool, T_pool, val_pool, it_pool
    else:
        u, T, val, it = _fit_slot(arr, priors, kappa, pi0, law_mode, ff.temps, u_pool.copy(), T_pool)
    pi = 1.0 / (1.0 + np.exp(-u))
    m = SlotModel(pi=pi, pi0=pi0, kappa=float(kappa), law=T if law_mode else None, T=None if law_mode else float(T),
                  eta=eta, objective=val, rounds=it, cv=cv_rows)
    tdesc = (f"log T = {T.a:.4f} + {T.b:.4f} log(sites / {N_REF_SITES:g}) (T at 300 / 3,000 / 30,000 sites "
             f"{T(300):.3g} / {T(3000):.3g} / {T(30000):.3g})") if law_mode else f"T {T:g}"
    log(f"  slot model: pi_0 {pi0:.4f}, kappa {kappa:g}" + (" (CV: " + ", ".join(f"{r['kappa']:g} {r['cv_log_loss_per_piece']:.5f}" for r in cv_rows) + ")" if cv_rows else "")
        + f"; pi_h median {np.median(pi):.4f}, >= 0.15 in {np.mean(pi >= 0.15):.1%} of bins, >= 0.3 in {int((pi >= 0.3).sum())} bins; {tdesc}; {it} rounds")
    return m


def slot_decide(contigs: ContigScores, model: SlotModel, rules: dict, thr_c, thr_i, share_rows=None, log=print):
    """Decisions under the slot model: p_c by the EM at T = 1 over "community draw" against "some slot" (bin and
    relative together), then refine() at each contig's temperature with p_unknown = p_c, the bins' scores carrying
    log(1 - pi_h) and the unknown term = community draw + every bin's relative. Returns (Refinement, p_c, T, extra)."""
    from .refine import em_weight, refine
    R = contigs.R[model.eta]
    S, Sc = contigs.S, contigs.Su_comm
    N, H = S.shape
    lpi, l1pi = np.log(np.maximum(model.pi, 1e-300)), np.log1p(-np.minimum(model.pi, 1 - 1e-12))
    unb = np.flatnonzero((contigs.given < 0) & contigs.decidable) if share_rows is None else np.asarray(share_rows)
    slot1 = logsumexp(np.logaddexp(S[unb] + l1pi[None], R[unb] + lpi[None]), axis=1) - np.log(H)
    prior = rules.get("p_unknown_prior", (1.0, 1.0))
    p_c = float(em_weight(slot1, Sc[unb], *prior)[0])
    T = model.temperature(contigs.n_sites)
    Tv = np.broadcast_to(np.asarray(T, float), (N,))
    S2 = S + Tv[:, None] * l1pi[None, :]
    zu = model.unknown_logit(R, Sc, Tv, p_c)
    ref = refine(S2, None, contigs.given, contigs.testable, Tv, thr_c, thr_i, p_unknown=p_c, decidable=contigs.decidable,
                 Zu=zu, **rules)
    # reports: slot masses and the unknown share including relatives, over the decided unbinned contigs
    u_all = np.flatnonzero((contigs.given < 0) & contigs.decidable)
    Tu = Tv[u_all][:, None]
    A = S[u_all] / Tu + l1pi[None]
    B = R[u_all] / Tu + lpi[None]
    slot = np.logaddexp(A, B)
    lc = np.log(p_c) + Sc[u_all] / Tv[u_all]
    ls = np.log1p(-p_c) - np.log(H) + slot
    tot = np.logaddexp(lc, logsumexp(ls, axis=1))
    P_slot = np.exp(ls - tot[:, None])
    rel = np.exp(B - slot)
    mass = P_slot.sum(0)
    p_u = float(np.mean(np.exp(lc - tot) + (P_slot * rel).sum(1)))
    pi_bar = float((mass * model.pi).sum() / max(mass.sum(), 1e-300))
    extra = {"p_c": p_c, "p_unknown_with_relatives": p_u, "pi_bar_slot_mass": pi_bar,
             "slot_mass_share_pi_at_least_0.3": float(mass[model.pi >= 0.3].sum() / max(mass.sum(), 1e-300))}
    log(f"  slot model decisions: p_c {p_c:.4f} (EM at T = 1), p_unknown with relatives {p_u:.4f}, slot-mass pi_bar "
        f"{pi_bar:.4f}, slot mass at pi >= 0.3 {extra['slot_mass_share_pi_at_least_0.3']:.1%}")
    return ref, p_c, T, extra


def slot_hidden_relative(contigs: ContigScores, model: SlotModel, p_c: float, T, best_bin) -> np.ndarray:
    """Per contig: P(an unbinned relative of best_bin explains it) under the slot model (report only)."""
    S, R, Sc = contigs.S, contigs.R[model.eta], contigs.Su_comm
    N, H = S.shape
    Tv = np.broadcast_to(np.asarray(T, float), (N,))[:, None]
    lpi, l1pi = np.log(np.maximum(model.pi, 1e-300)), np.log1p(-np.minimum(model.pi, 1 - 1e-12))
    ls = np.log1p(-p_c) - np.log(H) + np.logaddexp(S / Tv + l1pi[None], R / Tv + lpi[None])
    tot = np.logaddexp(np.log(p_c) + Sc / Tv[:, 0], logsumexp(ls, axis=1))
    rows = np.arange(N)
    return np.exp(np.log1p(-p_c) - np.log(H) + lpi[best_bin] + R[rows, best_bin] / Tv[:, 0] - tot)
