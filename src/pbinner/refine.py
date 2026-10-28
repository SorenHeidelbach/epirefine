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
           contamination_threshold: float = 0.5, include_threshold: float = 0.9,
           eps_prior=(1.0, 19.0), p_unknown_prior=(1.0, 1.0), p_unknown: float | None = None,
           methylome_difference: np.ndarray | None = None, min_difference: float = 0.4,
           decidable: np.ndarray | None = None, unmethylated_bins: np.ndarray | None = None,
           include_threshold_unmethylated: float = 0.95) -> Refinement:
    """given[i] = bin index of contig i, -1 if unbinned; testable[i] = binned and scored leave-one-out.
    decidable[i] = False: no decision about contig i at all (e.g. coverage too low): it is neither
    flagged nor included and doesn't enter the estimates of p_unknown and the contamination rate.

    unmethylated_bins[h] = True for bins with no detected methylation: inclusion into them needs
    include_threshold_unmethylated. Such a bin is matched by any contig that shows no methylation at
    the listed motifs, including contigs of genomes without a bin whose own motifs are not in the list
    (Nanomotif sees only bins): all 4 wrong placements on the monoculture set were orphan genomes put
    into methylation-poor bins at P 0.905-0.918.

    methylome_difference[i]: over motif groups, the largest distance between the bin's methylated
    fraction and contig i's 95% credible interval for its own. A contaminant must clear its bin by at
    least min_difference in some group: a foreign genome has a different set of MTases (a motif
    methylated in one, not the other), whereas small consistent shifts over many sites are
    within-genome variation that the likelihood alone reads as overwhelming evidence (dev log D20).
    """
    N, H = S.shape
    T = np.asarray(temperature, float)  # scalar or one per contig
    Z, Zu = S / (T[:, None] if T.ndim == 1 else T), Su / T
    dec = np.ones(N, bool) if decidable is None else np.asarray(decidable, bool)
    testable = testable & dec
    unb = (given < 0) & dec
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

    candidate = unb | contaminant  # unb already excludes contigs without a decision
    thr = np.full(N, include_threshold)
    if unmethylated_bins is not None:
        to_bin = best_alt < H
        thr[to_bin] = np.where(np.asarray(unmethylated_bins)[best_alt[to_bin]], include_threshold_unmethylated, include_threshold)
    include_bin = np.where(candidate & (best_alt < H) & (p_best_alt >= thr), best_alt, -1)
    return Refinement(float(p_unknown), float(eps), p_foreign, best_alt, p_best_alt, contaminant, candidate,
                      include_bin)


def regional_pattern(llrs: list[np.ndarray], switch: float = 0.05, iters: int = 100, weights=None):
    """Is a long contig its bin's, foreign, or a mixture of the two along its length?

    llrs[u]: per 50 kb chunk of contig u, in order, log p(chunk | foreign) - log p(chunk | its bin) (from
    the robust host-profile fit, bin profile without the chunk). Three explanations:
        member   every chunk belongs to the bin            log L = 0
        foreign  every chunk is foreign                     log L = sum llr
        mixed    a two-state chain along the contig (switch probability `switch` per chunk), excluding
                 the two pure paths
    with mixture weights estimated by EM over all contigs. On ZymoFecal every flag that Hi-C contradicts
    on a contig >= 150 kb had foreign and bin-like chunks (17-83% foreign: islands, chimeric joins), while
    the Hi-C-confirmed contaminants were foreign in every chunk; whole-contig scores mixed the two.
    weights: fixed prior weights of the three explanations (no EM), e.g. tied to the model's p_unknown.
    Returns (posterior [U, 3] for member/foreign/mixed, per contig the chunk posterior of 'foreign' under
    the mixed explanation, the weights).
    """
    from scipy.special import logsumexp as lse
    ls, l1s = np.log(switch), np.log1p(-switch)
    T = np.array([[l1s, ls], [ls, l1s]])
    L = np.zeros((len(llrs), 3))
    chunk_post = []
    for u, x in enumerate(llrs):
        x = np.asarray(x, float)
        n = len(x)
        em = np.stack([np.zeros(n), x], 1)  # emissions: bin, foreign
        a = np.zeros((n, 2)); b = np.zeros((n, 2))
        a[0] = np.log(0.5) + em[0]
        for k in range(1, n):
            a[k] = lse(a[k - 1][:, None] + T, axis=0) + em[k]
        for k in range(n - 2, -1, -1):
            b[k] = lse(T + (em[k + 1] + b[k + 1])[None, :], axis=1)
        total = lse(a[-1])
        pure = np.logaddexp(np.log(0.5) + (n - 1) * l1s, np.log(0.5) + (n - 1) * l1s + x.sum())
        mixed = total + np.log(max(-np.expm1(min(pure - total, 0.0)), 1e-300))
        L[u] = (0.0, x.sum(), mixed)
        g = a + b
        chunk_post.append(np.exp(g[:, 1] - lse(g, axis=1)))
    if weights is None:
        w = np.full(3, 1 / 3)
        for _ in range(iters):
            z = L + np.log(w)[None]
            r = np.exp(z - lse(z, axis=1, keepdims=True))
            w = np.clip(r.mean(0), 1e-4, None); w /= w.sum()
    else:
        w = np.clip(np.asarray(weights, float), 1e-6, None); w /= w.sum()
    z = L + np.log(w)[None]
    return np.exp(z - lse(z, axis=1, keepdims=True)), chunk_post, w


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
                  coverage: np.ndarray | None = None, low_coverage: np.ndarray | None = None,
                  too_short: np.ndarray | None = None, regional: dict | None = None):
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
        # coverage too low: no decision was made about these contigs (binned or not)
        status[low_coverage] = "low_coverage"
    if too_short is not None:
        # shorter than min_contig_length: no decision either
        status[too_short] = "too_short"
    if regional is not None:
        # flagged by the whole-contig score, but along its length partly the bin's and partly foreign:
        # kept in its bin, foreign segments reported
        status[regional["mixed"]] = "mixed"
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
        "p_regional_foreign": regional["p"][:, 1] if regional is not None else np.full(len(contigs), np.nan),
        "p_regional_mixed": regional["p"][:, 2] if regional is not None else np.full(len(contigs), np.nan),
        "foreign_segments": regional["segments"] if regional is not None else [""] * len(contigs),
        "segment_bins": regional["segment_bins"] if regional is not None else [""] * len(contigs),
    }).write_csv(out / "contig_decisions.tsv", separator="\t", float_precision=4)
