"""Bin refinement: contamination detection for binned contigs and inclusion of unbinned ones.

Inputs are the calibrated log-likelihoods of each contig under every bin (binned contigs scored
against their own bin leave-one-out) and under an unbinned genome ("unknown"):
    Z[i, h] = S[i, h] / T,   Zu[i] = Su[i] / T.

Two mixtures, with their weights estimated from the data by EM (MAP with a Beta prior):

  unbinned contig i:  from bin h with prob (1 - p_u) / H, from an unbinned genome with prob p_u
                      p_u = fraction of unbinned contigs whose genome has no bin
  binned contig i in bin b:  belongs to b with prob 1 - eps; otherwise it is foreign, coming from
                      another bin or an unbinned genome in the proportions above
                      eps = contamination rate of the binning

Decisions:
  * contaminant  if P(foreign | data) >= contamination_threshold;
  * a contaminant or an unbinned contig is included into bin h if P(h | data, not its old bin)
    >= include_threshold ("high_confidence"); 0.8 / 0.5 mark medium / low confidence, as in
    Nanomotif's include_contigs, and are reported but not applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.special import expit, logsumexp

from .model import credible_set

CONFIDENCE = ((0.95, "high_confidence"), (0.8, "medium_confidence"), (0.5, "low_confidence"))


def em_weight(l_in: np.ndarray, l_out: np.ndarray, a: float = 1.0, b: float = 1.0, iters: int = 500,
              tol: float = 1e-8) -> tuple[float, np.ndarray]:
    """MAP weight w of the mixture (1 - w) P_in + w P_out from per-item log-likelihoods.

    Beta(a, b) prior on w. Returns w and each item's responsibility P(out | data).
    """
    if len(l_in) == 0:
        return (a - 1) / max(a + b - 2, 1e-12) if a + b > 2 else 0.5, np.zeros(0)
    d = l_out - l_in
    w = 0.5
    for _ in range(iters):
        r = expit(np.log(w) - np.log1p(-w) + d)
        new = (r.sum() + a - 1) / (len(r) + a + b - 2)
        new = float(np.clip(new, 1e-6, 1 - 1e-6))
        if abs(new - w) < tol:
            w = new
            break
        w = new
    return w, expit(np.log(w) - np.log1p(-w) + d)


@dataclass
class Refinement:
    p_unknown: float
    eps: float
    p_foreign: np.ndarray      # [N] P(foreign | data) for testable binned contigs, nan otherwise
    best_alt: np.ndarray       # [N] best explanation other than the contig's own bin (H = unknown)
    p_best_alt: np.ndarray     # [N] its probability given that the contig is foreign / unbinned
    contaminant: np.ndarray    # [N] bool (P(foreign) above threshold and a methylome difference)
    candidate: np.ndarray      # [N] bool: unbinned, or flagged contaminant
    include_bin: np.ndarray    # [N] bin index to include into (high confidence), -1 otherwise


def refine(S: np.ndarray, Su: np.ndarray, given: np.ndarray, testable: np.ndarray, temperature: float,
           contamination_threshold: float = 0.5, include_threshold: float = 0.95,
           eps_prior=(1.0, 19.0), p_unknown_prior=(1.0, 1.0), p_unknown: float | None = None,
           methylome_difference: np.ndarray | None = None, min_difference: float = 0.3) -> Refinement:
    """given[i] = bin index of contig i, -1 if unbinned; testable[i] = binned and scored leave-one-out.

    methylome_difference[i]: over motif groups, the largest distance between the bin's methylated
    fraction and contig i's 95% credible interval for its own. A contaminant must clear its bin by at
    least min_difference in some group: a foreign genome has a different set of MTases (a motif
    methylated in one, not the other), whereas small consistent shifts over many sites are
    within-genome variation that the likelihood alone reads as overwhelming evidence (dev log D20).
    """
    N, H = S.shape
    Z, Zu = S / temperature, Su / temperature
    unb = given < 0
    # p_u from the unbinned contigs: "some bin, uniformly" vs "an unbinned genome"
    l_bins = logsumexp(Z, axis=1) - np.log(H)
    if p_unknown is None:
        p_unknown, _ = em_weight(l_bins[unb], Zu[unb], *p_unknown_prior)
    lpu, lpb = np.log(p_unknown), np.log1p(-p_unknown)

    # alternatives to the contig's own bin: other bins (uniform) and unknown
    own = np.where(given >= 0, given, 0)
    Zx = Z.copy()
    Zx[np.arange(N)[given >= 0], own[given >= 0]] = -np.inf  # exclude the own bin
    n_alt = np.where(given >= 0, H - 1, H)
    alt_logits = np.concatenate([Zx + lpb - np.log(n_alt)[:, None], (Zu + lpu)[:, None]], axis=1)
    l_alt = logsumexp(alt_logits, axis=1)
    p_alt = np.exp(alt_logits - l_alt[:, None])
    best_alt = p_alt.argmax(1)
    p_best_alt = p_alt.max(1)

    p_foreign = np.full(N, np.nan)
    t = testable & (given >= 0)
    eps, r = em_weight(Z[t, own[t]], l_alt[t], *eps_prior)
    p_foreign[t] = r
    contaminant = np.zeros(N, bool)
    contaminant[t] = r >= contamination_threshold
    if methylome_difference is not None:
        contaminant &= methylome_difference >= min_difference

    candidate = unb | contaminant
    include_bin = np.where(candidate & (best_alt < H) & (p_best_alt >= include_threshold), best_alt, -1)
    return Refinement(float(p_unknown), float(eps), p_foreign, best_alt, p_best_alt, contaminant, candidate,
                      include_bin)


def confidence(p: float) -> str:
    return next((name for lo, name in CONFIDENCE if p >= lo), "")


def bin_evidence_table(bins: list[str], summary, n_sites_group: np.ndarray, given: np.ndarray,
                       lo_methylated: float = 0.5, hi_unmethylated: float = 0.5) -> pl.DataFrame:
    """Per bin: is there methylation evidence for its methylome?

    methylated               some motif group's 95% interval lies above lo_methylated
    no_methylation_detected  every group with sites is confidently below hi_unmethylated (an
                             unmethylated genome: still a profile other bins can be told from)
    no_methylation_evidence  neither: the intervals are too wide to say (too few sites or reads)
    """
    mean, lo, hi, _ = summary
    sites = np.zeros((len(bins), n_sites_group.shape[1]), np.int64)
    for b in range(len(bins)):
        sel = given == b
        if sel.any():
            sites[b] = n_sites_group[sel].sum(0)
    rows = []
    for b, name in enumerate(bins):
        has = sites[b] > 0
        meth = has & (lo[b] >= lo_methylated)
        unmeth = has & (hi[b] < hi_unmethylated)
        status = ("methylated" if meth.any() else
                  "no_methylation_detected" if has.any() and unmeth[has].all() else "no_methylation_evidence")
        rows.append({"bin": name, "status": status, "groups_with_sites": int(has.sum()),
                     "groups_methylated": int(meth.sum()), "groups_confidently_unmethylated": int(unmeth.sum()),
                     "groups_undetermined": int((has & ~meth & ~unmeth).sum()), "sites": int(sites[b].sum())})
    return pl.DataFrame(rows)


def write_outputs(out, contigs: list[str], lengths: dict, given: np.ndarray, bins: list[str], ref: Refinement,
                  n_sites: np.ndarray, sole: set, P: np.ndarray, methylome_difference: np.ndarray | None = None,
                  redundancy: np.ndarray | None = None, evidence: np.ndarray | None = None, min_evidence: float = 3.0,
                  coverage: np.ndarray | None = None, low_coverage: np.ndarray | None = None):
    """Nanomotif-compatible files + pbinner's per-contig decision table."""
    H = len(bins)
    labels = bins + ["unknown"]
    given_lbl = np.array([bins[g] if g >= 0 else "" for g in given], dtype=object)
    length = np.array([lengths[c] for c in contigs])
    bin_len = {b: 0 for b in bins}
    bin_n = {b: 0 for b in bins}
    for g, L in zip(given_lbl, length):
        if g:
            bin_len[g] += int(L); bin_n[g] += 1

    ci = np.flatnonzero(ref.contaminant)
    pl.DataFrame({
        "contig": [contigs[i] for i in ci], "bin": [given_lbl[i] for i in ci], "method": ["pbinner"] * len(ci),
        "cluster": [labels[ref.best_alt[i]] for i in ci], "bin_cluster": [given_lbl[i] for i in ci],
        "bin_length": [bin_len[given_lbl[i]] for i in ci], "n_contigs_bin": [bin_n[given_lbl[i]] for i in ci],
        "fraction_contigs": [1 / bin_n[given_lbl[i]] for i in ci],
        "fraction_length": [length[i] / bin_len[given_lbl[i]] for i in ci],
        "p_contaminant": ref.p_foreign[ci], "p_cluster": ref.p_best_alt[ci],
    }, schema_overrides={"contig": pl.Utf8, "bin": pl.Utf8, "method": pl.Utf8, "cluster": pl.Utf8,
                         "bin_cluster": pl.Utf8}).write_csv(out / "bin_contamination.tsv", separator="\t",
                                                            float_precision=4)

    kept = [(c, given_lbl[i]) for i, c in enumerate(contigs) if given_lbl[i] and not ref.contaminant[i]]
    pl.DataFrame(kept, schema={"contig": pl.Utf8, "bin": pl.Utf8}, orient="row").sort("bin", "contig").write_csv(
        out / "decontaminated_contig_bin.tsv", separator="\t")

    rows = []
    for i in np.flatnonzero(ref.candidate & (ref.best_alt < H) & (ref.p_best_alt >= CONFIDENCE[-1][0])):
        p = float(ref.p_best_alt[i])
        rows.append({"contig": contigs[i], "bin": given_lbl[i] or "unbinned", "assigned_bin": labels[ref.best_alt[i]],
                     "method": "pbinner", "prob": p, "mean_prob": p, "confidence": confidence(p)})
    pl.DataFrame(rows, schema={"contig": pl.Utf8, "bin": pl.Utf8, "assigned_bin": pl.Utf8, "method": pl.Utf8,
                               "prob": pl.Float64, "mean_prob": pl.Float64, "confidence": pl.Utf8}).sort(
        "confidence", "contig").write_csv(out / "include_contigs.tsv", separator="\t", float_precision=4)

    new = kept + [(contigs[i], bins[ref.include_bin[i]]) for i in np.flatnonzero(ref.include_bin >= 0)]
    pl.DataFrame(new, schema={"contig": pl.Utf8, "bin": pl.Utf8}, orient="row").sort("bin", "contig").write_csv(
        out / "new_contig_bin.tsv", separator="\t")

    status = np.where(given >= 0, "kept", "unbinned").astype(object)
    status[ref.contaminant] = "contaminant"
    for i, c in enumerate(contigs):
        if c in sole:
            status[i] = "sole_contig_of_bin"
    if redundancy is not None:
        status[~np.isnan(redundancy) & (ref.include_bin < 0)] = "redundant"
    status[(ref.include_bin >= 0) & (given < 0)] = "included"
    status[(ref.include_bin >= 0) & (given >= 0)] = "moved"
    if evidence is not None:
        # kept / unbinned only because nothing could be seen: say so (expected log BF vs a random
        # community genome below min_evidence, or no sites at all)
        weak = np.isin(status, ["kept", "unbinned"]) & ~(np.nan_to_num(evidence, nan=0.0) >= min_evidence)
        status[weak] = "no_methylation_evidence"
    if low_coverage is not None:
        # binned contigs below the coverage cutoff were not tested for contamination
        status[low_coverage & np.isin(status, ["kept", "no_methylation_evidence"])] = "low_coverage"
    new_bin = np.array([bins[ref.include_bin[i]] if ref.include_bin[i] >= 0
                        else (given_lbl[i] if given[i] >= 0 and not ref.contaminant[i] else "")
                        for i in range(len(contigs))], dtype=object)
    pl.DataFrame({
        "contig": contigs, "length": length, "n_sites": n_sites, "given_bin": given_lbl, "status": status,
        "new_bin": new_bin, "p_contaminant": ref.p_foreign,
        "methylome_difference": methylome_difference if methylome_difference is not None else np.full(len(contigs), np.nan),
        "redundancy_with_bin": redundancy if redundancy is not None else np.full(len(contigs), np.nan),
        "best_alternative": [labels[k] for k in ref.best_alt], "p_best_alternative": ref.p_best_alt,
        "map": [labels[k] for k in P.argmax(1)], "p_map": P.max(1), "p_unknown": P[:, -1],
        "credible_set_95": [",".join(credible_set(row, labels)) for row in P],
        "expected_evidence": evidence if evidence is not None else np.full(len(contigs), np.nan),
        "coverage": coverage if coverage is not None else np.full(len(contigs), np.nan),
    }).write_csv(out / "contig_decisions.tsv", separator="\t", float_precision=4)
