"""Per-contig scoring of refinement calls, with the breakdowns a referee asks for.

benchmarks/evaluate_refinement.py gives one row of pooled metrics per run and leaves out the contigs the tool
did not judge. This module keeps the same truth and the same definitions but works per contig, so the same
calls can be reported

  * with undecided contigs counted as misses (the universe is every planted error, not the decided ones);
  * per coverage class (>= 5x, >= 10x) and per length class;
  * as precision-recall curves of any score (EpiRefine's posterior or a trivial baseline), with the average
    precision and the recall reached at a required precision;
  * with intervals from a cluster bootstrap: over isolates (pieces of one isolate are not independent, and the
    same isolate sits in several communities) or over scenarios (for paired differences between two settings).

`tally()` reproduces evaluate()'s counts exactly when no contig is left out (tests/test_evidence.py).
"""
from __future__ import annotations

import numpy as np
import polars as pl

POSITIVE = ("contaminant", "orphan_contaminant")       # binned contigs that should be flagged
WANTED = ("missing", "contaminant")                    # contigs that should end up in their true bin
STAY = ("orphan", "unknown", "orphan_contaminant")     # contigs whose genome has no bin
LENGTH_CLASSES = ((0, 5_000, "2-5 kb"), (5_000, 10_000, "5-10 kb"), (10_000, 20_000, "10-20 kb"),
                  (20_000, 50_000, "20-50 kb"), (50_000, 10**12, ">= 50 kb"))
COUNTS = ("tp", "fp", "fn", "want", "included_ok", "new", "new_wrong", "stay", "stay_wrong")


def tally(truth: pl.DataFrame, flagged, final_bin: dict) -> pl.DataFrame:
    """One row per contig with evaluate_refinement.evaluate()'s events as 0/1 columns.

    truth: truth_actions.tsv (contig, length, true_bin, given_bin, category). flagged: contigs called
    contamination. final_bin: contig -> bin after refinement (absent = unbinned)."""
    t = truth.with_columns(pl.col("true_bin", "given_bin").fill_null(""))
    t = t.with_columns(pl.col("contig").is_in(list(flagged)).alias("flagged"),
                       pl.col("contig").replace_strict(final_bin, default="").alias("final_bin"))
    binned = pl.col("given_bin") != ""
    pos = pl.col("category").is_in(POSITIVE)
    new = (pl.col("final_bin") != "") & (pl.col("final_bin") != pl.col("given_bin"))
    return t.with_columns(
        (binned & pos & pl.col("flagged")).cast(pl.Int32).alias("tp"),
        (binned & ~pos & pl.col("flagged")).cast(pl.Int32).alias("fp"),
        (binned & pos & ~pl.col("flagged")).cast(pl.Int32).alias("fn"),
        pl.col("category").is_in(WANTED).cast(pl.Int32).alias("want"),
        (pl.col("category").is_in(WANTED) & (pl.col("final_bin") == pl.col("true_bin"))).cast(pl.Int32).alias("included_ok"),
        new.cast(pl.Int32).alias("new"),
        (new & (pl.col("final_bin") != pl.col("true_bin"))).cast(pl.Int32).alias("new_wrong"),
        pl.col("category").is_in(STAY).cast(pl.Int32).alias("stay"),
        (pl.col("category").is_in(STAY) & (pl.col("final_bin") != "")).cast(pl.Int32).alias("stay_wrong"))


def rates(c: dict) -> dict:
    """The benchmark's ratios from summed counts (nan when a denominator is empty)."""
    def div(a, b):
        return a / b if b else float("nan")
    return {"contam_precision": div(c["tp"], c["tp"] + c["fp"]), "contam_recall": div(c["tp"], c["tp"] + c["fn"]),
            "inclusion_recall": div(c["included_ok"], c["want"]),
            "placement_precision": div(c["new"] - c["new_wrong"], c["new"]),
            "no_bin_placed": div(c["stay_wrong"], c["stay"])}


def length_class(length) -> np.ndarray:
    length = np.asarray(length)
    out = np.empty(len(length), dtype=object)
    for lo, hi, name in LENGTH_CLASSES:
        out[(length >= lo) & (length < hi)] = name
    return out


def cluster_bootstrap(counts: np.ndarray, stat, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(summed counts) when whole clusters are resampled.

    counts: [clusters, k] sums per cluster (an isolate, or a scenario); stat: vector of k sums -> float."""
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)          # how often each cluster is drawn
    vals = np.array([stat(row) for row in w @ counts], float)
    vals = vals[~np.isnan(vals)]
    return (float("nan"), float("nan")) if not len(vals) else tuple(np.percentile(vals, [2.5, 97.5]))


def pr_curve(score, correct, positive=None, weight=None):
    """Precision and recall at every distinct threshold of `score` (highest first).

    correct: the call is right if made (a true contaminant; a contig placed in its true bin). positive: counts
    in the recall denominator (default: correct); items with score -inf or nan are never called but still count
    as positives. weight: per-item multiplicity (bootstrap). Returns thresholds, precision, recall."""
    score = np.asarray(score, float)
    correct = np.asarray(correct, float)
    positive = correct if positive is None else np.asarray(positive, float)
    w = np.ones(len(score)) if weight is None else np.asarray(weight, float)
    n_pos = float((w * positive).sum())
    ok = np.isfinite(score) & (w > 0)
    order = np.argsort(-score[ok], kind="stable")
    s, tp, n = score[ok][order], np.cumsum((w * correct)[ok][order]), np.cumsum(w[ok][order])
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1] if len(s) else np.array([], int)   # end of each tie block
    return s[last], tp[last] / np.maximum(n[last], 1e-12), tp[last] / n_pos if n_pos else np.full(len(last), np.nan)


def average_precision(score, correct, positive=None, weight=None) -> float:
    """Area under the precision-recall curve (step-wise: sum of precision x gain in recall)."""
    _, p, r = pr_curve(score, correct, positive, weight)
    return float(np.sum(p * np.diff(np.r_[0.0, r]))) if len(p) else 0.0


def recall_at_precision(score, correct, min_precision: float, positive=None, weight=None) -> tuple[float, float]:
    """(best recall of any threshold whose precision is at least min_precision, that threshold); (0, inf) if none.

    The threshold is chosen with the truth in hand, so this flatters the scored method."""
    s, p, r = pr_curve(score, correct, positive, weight)
    ok = p >= min_precision - 1e-12
    if not ok.any():
        return 0.0, float("inf")
    i = int(np.argmax(np.where(ok, r, -1)))
    return float(r[i]), float(s[i])


def precision_at_recall(score, correct, min_recall: float, positive=None, weight=None) -> float:
    """Precision at the strictest threshold that reaches min_recall (nan when no threshold does)."""
    _, p, r = pr_curve(score, correct, positive, weight)
    ok = np.nonzero(r >= min_recall - 1e-12)[0]
    return float(p[ok[0]]) if len(ok) else float("nan")
