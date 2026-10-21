"""Self-calibration with pseudo-plasmids.

Host contigs are cut into fragments of plasmid-like lengths. Each fragment is scored as if it were
a plasmid, with the chunks it overlaps removed from its own bin's profile (leave-one-out), so the
true answer is known without external ground truth.

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

Caveat: fragments are chromosomal. Plasmid-specific effects (own MTases, under-methylation,
copy number) are not represented; that is v2.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .assignability import margin_vs_competitors
from .model import FittedModel, posterior, summed_scores, unknown_score

DEFAULT_LENGTHS = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000)
DEFAULT_TAUS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4, float("inf"))
DEFAULT_ETAS = (0.02, 0.05, 0.1, 0.2, 0.35)
DEFAULT_LAMS = (0.1, 0.5, 0.9, 1.0)  # 1.0 = community-only unknown (the earlier model)
DEFAULT_TEMPS = tuple(np.round(np.exp(np.linspace(np.log(0.2), np.log(20), 31)), 3))


def fragments_per_bin(budget: int, n_lengths: int, n_bins: int, cap: int) -> int:
    """Fragments per (bin, length) for a total budget: calibration error depends on the number of
    fragments, not on the number of bins, so the total stays fixed as bins grow."""
    return int(max(1, min(cap, round(budget / max(1, n_lengths * n_bins)))))


def select_fragments(host_contigs: pl.DataFrame, lengths, n_per_bin: int, rng: np.random.Generator) -> pl.DataFrame:
    """Pseudo-plasmids: up to n_per_bin random tiles of each length per bin.

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


def calibrate(L_chunks, chunk_bin, n_bins, group_modtype, grid, frags: pl.DataFrame, L_frag,
              chunks_of, p_unknown, rho_base=0.02, H_frag=None, taus=DEFAULT_TAUS, temps=DEFAULT_TEMPS,
              lookalike_dist=0.15, decidable_bf=3.0, etas=DEFAULT_ETAS, lams=DEFAULT_LAMS,
              allow_sharpening=False, log=print, **model_kw):
    fbin = frags["bin_idx"].to_numpy()
    H = n_bins
    rho = np.zeros(L_frag.shape[:2])  # pseudo-contigs are chromosomal: no own-MTase explanation

    def own_and_scores(model):
        sc = model.score(L_frag, rho, plasmid=False)
        A_own = np.empty(L_frag.shape[:2])
        for i in range(len(fbin)):
            A_own[i] = model.loo_rescore(sc, i, fbin[i], chunks_of[i], rho[i])
        # fragments from chunks the robust model calls foreign to their bin carry a wrong label
        keep = np.array([bool((model.w[ch] >= 0.5).all()) for ch in chunks_of])
        return sc, A_own, keep

    tau_rows, best, warm, group_ll = [], None, {}, []
    for tau in taus:
        model = FittedModel(L_chunks, chunk_bin, n_bins, group_modtype, grid, tau, **warm, **model_kw)
        warm = {"init_w": model.w, "init_pi": model.raw_pi}  # not the model: at most two are held at once
        sc, A_own, keep = own_and_scores(model)
        ll = float(sc.S[keep, fbin[keep]].sum())
        group_ll.append(A_own[keep].sum(0))
        tau_rows.append({"tau": float(tau), "label": f"{tau:g}", "loo_loglik": ll, "n_fragments": int(keep.sum())})
        log(f"  tau={tau:>8g}  LOO log-lik={ll:.1f}")
        if best is None or ll > best[0]:
            best = (ll, tau, sc, A_own, model, keep)
        del model
    # per-group tau: each group's own leave-one-out term is maximised separately (groups with
    # intermediate, variable methylation need a finite tau; groups at 0 or 1 an infinite one)
    group_ll = np.array(group_ll)  # [n_taus, M]
    tau_g = np.array([taus[k] for k in group_ll.argmax(0)], dtype=float)
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
    tau_groups = pl.DataFrame({"group": np.arange(len(tau_g)), "tau": tau_g,
                               "loo_loglik_gain_vs_inf": group_ll.max(0) - group_ll[-1]})
    _, tau, sc_best, A_own_best, best_model, keep = best
    if (~keep).any():
        log(f"  {int((~keep).sum())} fragments excluded: from chunks flagged foreign to their bin")
    kidx = np.flatnonzero(keep)
    S, fbin, frags_k = sc_best.S[kidx], fbin[kidx], frags[kidx]
    Su_comm, cu, A_own_k = sc_best.Su_comm[kidx], sc_best.cu[kidx], A_own_best[kidx]
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
    if H_frag is not None and H > 1:
        m_exp, _, _ = margin_vs_competitors(best_model, *H_frag, fbin, False, 1.0)
        decidable = m_exp >= decidable_bf
    distinct = ~f_look & decidable
    log(f"  test B uses {int(distinct.sum())} of {len(fbin)} fragments (bin decidable from its nearest competitor)")

    # Unknown host = community draw (weight lam) or an unbinned relative of a bin, which takes each
    # group's level from that bin with prob 1 - eta (weight 1 - lam). (eta, lam, T) are fitted together
    # on test A + decidable test B; in test B the removed bin is not offered as a relative either.
    def relatives(eta):
        _, R = summed_scores(sc_best.Lp[kidx], best_model.log_prof, sc_best.O[kidx], rho[kidx], rel=(eta, cu))
        R[np.arange(len(fbin)), fbin] = np.logaddexp(np.log1p(-eta) + A_own_k, np.log(eta) + cu).sum(1)  # leave-out
        return R

    def logloss(T, Su_a, Su_b):
        Pa = posterior(S, Su_a, p_unknown, T)
        la = -np.log(np.maximum(Pa[np.arange(len(fbin)), truth_a], 1e-12)).mean()
        if not distinct.any():
            return la, la, np.nan
        Pb = posterior(S[distinct], Su_b[distinct], p_unknown, T, host_mask=mask_b[distinct])
        lb = -np.log(np.maximum(Pb[:, H], 1e-12)).mean()
        return la + lb, la, lb

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
    # Never sharpen: pseudo-plasmids are chromosomal and lack plasmid-specific uncertainty, so they
    # can reveal overconfidence but cannot justify making plasmid posteriors more confident.
    if not allow_sharpening:
        fit = min((r for r in temp_rows if r["temperature"] >= 1.0), key=lambda r: r["log_loss"])
    temperature, eta, lam = fit["temperature"], fit["eta"], fit["lam"]
    Su, Su_b = Su_by[(eta, lam)]
    log(f"  unknown model: eta={eta:g}, lam={lam:g}; temperature fitted {fitted_temperature:g} -> used {temperature:g}")

    rows, rel = [], []
    for label, T in (("uncalibrated", 1.0), ("fitted_temperature", fitted_temperature), ("calibrated", temperature)):
        Pa = posterior(S, Su, p_unknown, T)
        Pb = posterior(S, Su_b, p_unknown, T, host_mask=mask_b)
        for Lf in sorted(frags_k["length"].unique().to_list()) + ["all"]:
            sel = np.ones(len(fbin), bool) if Lf == "all" else (frags_k["length"].to_numpy() == Lf)
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
    # does the assignability score (computed without looking at methylation) predict accuracy?
    assign_rows, frag_extra = [], {}
    if H_frag is not None:
        Pa = posterior(S, Su, p_unknown, temperature)
        margin, nearest, vs_unk = margin_vs_competitors(best_model, *H_frag, fbin, False, temperature)
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
    return {"tau": tau, "eta": float(eta), "lam": float(lam), "temperature": float(temperature), "fitted_temperature": float(fitted_temperature), "summary": pl.DataFrame(rows), "lookalikes": lookalikes,
            "tau_fit": pl.DataFrame(tau_rows), "tau_groups": tau_groups, "temperature_fit": pl.DataFrame(temp_rows),
            "reliability": pl.DataFrame(rel), "assignability": pl.DataFrame(assign_rows),
            "fragments": frags.with_columns(pl.Series("foreign_excluded", ~keep), **{
                k: pl.Series(_expand(v, kidx, len(keep))) for k, v in frag_extra.items()}), "model": best_model}


def _expand(values, idx, n):
    out = np.full(n, np.nan)
    out[idx] = values
    return out
