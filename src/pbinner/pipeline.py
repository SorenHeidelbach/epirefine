"""End-to-end run: motifs -> sites -> detection model -> robust host profiles -> calibration ->
plasmid parameters -> MTase-aware scoring -> assignability and goodness of fit.

Data flow at scale: motif groups are found on a random sample of the sequence; one parallel pass
over the pileup writes a compact site cache (sites.extract_sites); a second parallel pass turns it
into per-unit likelihood tables (sites.unit_tables). Nothing proportional to assembly size x
number of motifs is held in memory.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logsumexp

from . import io as pio
from .assignability import detectability, margin_vs_competitors
from .calibrate import DEFAULT_LENGTHS, calibrate, fragment_chunks, fragments_per_bin, select_fragments
from .detection import detection_table, fit_detection
from .grid import Grid
from .model import FittedModel, credible_set, posterior, propagate
from .motifs import build_records, describe_groups, group_motifs
from .refine import refine, write_outputs
from .mtase import RhoParams, annotation_levels, bin_levels, estimate_rho, read_mtases
from .sites import EncodedAssembly, extract_sites, unit_tables

UNKNOWN = "unknown"
DEFAULT_DELTAS = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
DEFAULT_TAU_P = (10.0, 30.0, 100.0, 300.0, 1000.0, float("inf"))


def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def _write(df: pl.DataFrame, path, precision=4):
    df.write_csv(path, separator="\t", float_precision=precision)


def fit_plasmid_params(model: FittedModel, L, rho, p_unknown, min_sites_mask, deltas=DEFAULT_DELTAS,
                       taus=DEFAULT_TAU_P, max_units=1000, rng=None, log=print):
    """Type-II ML of (delta, tau_p) over plasmid candidates, marginalising over their host.

    At most `max_units` informative candidates are used (random subset): enough for two parameters.
    """
    idx = np.flatnonzero(min_sites_mask)
    if len(idx) > max_units:
        idx = np.sort((rng or np.random.default_rng(0)).choice(idx, max_units, replace=False))
    rows = []
    H = model.n_bins
    prior = np.r_[np.full(H, np.log((1 - p_unknown) / H)), np.log(p_unknown)]
    for d in deltas:
        for t in taus:
            model.set_plasmid_params(d, t)
            sc = model.score(L[idx], rho[idx], plasmid=True)
            ll = logsumexp(np.concatenate([sc.S, sc.Su[:, None]], 1) + prior, axis=1).sum()
            rows.append({"delta": d, "tau_p": t, "marginal_loglik": float(ll), "n_units": len(idx)})
    best = max(rows, key=lambda r: r["marginal_loglik"])
    log(f"  plasmid params: delta={best['delta']}, tau_p={best['tau_p']:g} ({len(idx)} candidates)")
    return best["delta"], best["tau_p"], pl.DataFrame(rows)


def _sample_contigs(contigs: dict[str, str], names, sample_bp: float, rng) -> dict[str, str]:
    """Random contigs up to sample_bp of sequence (all of them for small assemblies)."""
    names = list(names)
    if sum(len(contigs[c]) for c in names) <= sample_bp:
        return {c: contigs[c] for c in names}
    out, acc = {}, 0
    for i in rng.permutation(len(names)):
        c = names[i]
        out[c] = contigs[c]; acc += len(contigs[c])
        if acc >= sample_bp:
            break
    return out


def _manifest(**kw) -> dict:
    out = {}
    for k, v in kw.items():
        if isinstance(v, (str, Path)) and Path(v).exists():
            st = Path(v).stat()
            v = [str(Path(v).resolve()), st.st_size, int(st.st_mtime)]
        out[k] = v
    return json.loads(json.dumps(out, default=str))


def _same_partition(a, b) -> bool:
    """Same grouping of motifs, whatever the group numbers."""
    if len(a) != len(b):
        return False
    fa, fb = {}, {}
    return [fa.setdefault(x, len(fa)) for x in a] == [fb.setdefault(x, len(fb)) for x in b]


def _batched_summary(grid, logp_fn, n, batch=2000):
    """grid.summarize over units in batches; logp_fn(slice) -> log densities [b, M, G]."""
    parts = [grid.summarize(logp_fn(slice(s, s + batch))) for s in range(0, n, batch)]
    return [np.concatenate([p[i] for p in parts]) for i in range(4)]


def run(assembly, pileup, contig_bin, motifs_path, out, plasmids=None, mtases=None, score_all=True, min_cov=1,
        chunk_size=50_000, grid_size=64, p_unknown=None, contamination_threshold=0.5, include_threshold=0.95, do_calibrate=True, frag_lengths=DEFAULT_LENGTHS,
        frags_per_bin=30, frag_budget=6000, tau=None, temperature=None, delta=None, tau_p=None, rho_base=0.02,
        contam_eps=0.05, min_plasmids_for_fit=5, allow_sharpening=False, min_containment=0.5,
        uninformative_log_bf=3.0, misfit_log_bf=3.0, group_sample_bp=50e6, workers=None, cache_dir=None,
        detect_top=20, seed=1):
    out = Path(out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cache = Path(cache_dir) if cache_dir else out / "cache"

    contigs = pio.read_fasta(assembly)
    lengths = {c: len(s) for c, s in contigs.items()}
    cb = {c: b for c, b in pio.read_contig_bin(contig_bin).items() if c in contigs}
    bins = sorted(set(cb.values()))
    bin_idx = {b: i for i, b in enumerate(bins)}
    candidates = pio.read_contig_list(plasmids) & set(contigs) if plasmids else None
    host_contigs = sorted(c for c in cb if candidates is None or c not in candidates)
    host_set = set(host_contigs)
    # every contig is scored: binned ones leave-one-out (contamination), unbinned ones against all bins
    # (inclusion); plasmid parameters apply only to the given plasmid candidates
    targets = sorted(contigs)
    is_plasmid = np.array([candidates is not None and c in candidates for c in targets])
    H = len(bins)
    _log(f"{len(contigs)} contigs, {H} bins, {len(host_contigs)} host contigs, {len(targets)} targets")

    # motif groups, from the records of a random sample of the sequence
    motifs = pio.read_motifs(motifs_path)
    relevant = sorted(host_set | set(targets))
    sample = _sample_contigs(contigs, relevant, group_sample_bp, rng)
    records = build_records(sample, motifs)
    group_of = group_motifs(records, motifs, min_containment)
    rank_d = dict(records.group_by("motif").len().iter_rows())
    motif_rank = [int(rank_d.get(i, 0)) for i in range(len(motifs))]
    del records
    # a cache written with another numbering of the same groups is still valid: keep its labels
    man_path = cache / "manifest.json"
    cached = json.loads(man_path.read_text()) if man_path.exists() else None
    if cached and _same_partition(cached["inputs"].get("group_of", []), group_of):
        group_of = cached["inputs"]["group_of"]
    n_groups = max(group_of) + 1
    group_modtype = [next(motifs[i].mod_type for i in range(len(motifs)) if group_of[i] == g) for g in range(n_groups)]
    group_label = [",".join(motifs[i].key for i in range(len(motifs)) if group_of[i] == g) for g in range(n_groups)]
    _log(f"{len(motifs)} motifs -> {n_groups} groups (containment from {sum(map(len, sample.values())) / 1e6:.0f} Mb)")

    # pass 1: pileup -> site cache (reused when inputs are unchanged)
    man = _manifest(assembly=assembly, pileup=pileup, motifs=[m.key for m in motifs], group_of=group_of,
                    motif_rank=motif_rank, relevant=len(relevant))
    if cached and cached.get("inputs") == man:
        parts = [str(cache / "sites" / Path(f).name) for f in cached["parts"]]  # wherever the cache now lives
        asm = EncodedAssembly(cache / "asm")
        _log(f"reusing site cache {cache} ({len(parts)} parts)")
    else:
        asm = EncodedAssembly.write(contigs, cache / "asm")
        for f in (cache / "sites").glob("*.parquet") if (cache / "sites").exists() else []:
            f.unlink()
        wanted = None if len(relevant) == len(contigs) else {asm.index[c] for c in relevant}
        parts = extract_sites(asm, pileup, motifs, group_of, motif_rank, cache / "sites", wanted=wanted,
                              workers=workers, log=_log)
        man_path.write_text(json.dumps({"inputs": man, "parts": parts}))
    del contigs, sample
    cid = asm.index

    counts = (pl.scan_parquet(parts).filter(pl.col("n") >= min_cov)
              .group_by("group", "n", "k").agg(pl.len().alias("w")).collect())
    _log(f"{int(counts['w'].sum())} sites with coverage >= {min_cov}")
    _write(describe_groups(motifs, group_of, counts), out / "motif_groups.tsv")
    det = fit_detection(counts.select("group", "n", "k", "w"), dict(enumerate(group_modtype)),
                        workers=workers or min(16, os.cpu_count() or 1))
    _write(detection_table(det), out / "detection_model.tsv")
    _log("detection model fitted")

    # units: host chunks (from contig lengths), targets, calibration fragments
    grid = Grid(grid_size)
    n_ids = len(asm.names)
    chunk_offset = np.full(n_ids, -1, np.int64)
    n_chunks_of = {c: -(-lengths[c] // chunk_size) for c in host_contigs}
    off = 0
    for c in host_contigs:
        chunk_offset[cid[c]] = off; off += n_chunks_of[c]
    n_chunks_all = off
    chunk_contig = np.repeat([cid[c] for c in host_contigs], [n_chunks_of[c] for c in host_contigs])
    chunk_no = np.concatenate([np.arange(n_chunks_of[c]) for c in host_contigs]) if host_contigs else np.zeros(0, int)
    chunk_bin_all = np.repeat([bin_idx[cb[c]] for c in host_contigs], [n_chunks_of[c] for c in host_contigs])
    target_of = np.full(n_ids, -1, np.int64)
    for i, c in enumerate(targets):
        target_of[cid[c]] = i
    frags = None
    if do_calibrate:
        hc = pl.DataFrame({"contig": [cid[c] for c in host_contigs], "bin_idx": [bin_idx[cb[c]] for c in host_contigs],
                           "length": [lengths[c] for c in host_contigs]}, schema={"contig": pl.Int32, "bin_idx": pl.Int64, "length": pl.Int64})
        per_bin = fragments_per_bin(frag_budget, len(frag_lengths), H, frags_per_bin)
        frags = select_fragments(hc, frag_lengths, per_bin, rng)
        _log(f"{frags.height} calibration fragments ({per_bin} per bin and length)")

    # pass 2: site cache -> likelihood tables
    tabs = unit_tables(parts, n_groups=n_groups, det=det, grid_size=grid_size, min_cov=min_cov,
                       chunk_offset=chunk_offset, chunk_size=chunk_size, n_chunks=n_chunks_all, target_of=target_of,
                       n_targets=len(targets), frags=frags, workers=workers, log=_log)
    keep = np.flatnonzero(tabs["chunk_sites"] > 0)  # chunks without sites carry no information
    remap = np.full(n_chunks_all, -1, np.int64); remap[keep] = np.arange(len(keep))
    L_chunks = tabs.pop("L_chunks")[keep]
    chunk_bin, chunk_contig, chunk_no = chunk_bin_all[keep], chunk_contig[keep], chunk_no[keep]
    L_t, (H1_t, H0_t), n_sites, n_sites_group = tabs["L_t"], tabs["H_t"], tabs["n_sites_t"], tabs["n_sites_group_t"]
    _log(f"{len(keep)} host chunks of {chunk_size} bp; tables {L_chunks.nbytes / 1e9:.1f} GB (chunks), "
         f"{3 * L_t.nbytes / 1e9:.1f} GB (targets)")

    def chunk_index(contig_id, c):
        return int(remap[chunk_offset[contig_id] + c]) if chunk_offset[contig_id] >= 0 and c < n_chunks_of[asm.names[contig_id]] else -1

    p_cal = 0.1 if p_unknown is None else p_unknown  # prior used before p_unknown is estimated (refinement)
    params = {"chunk_size": chunk_size, "grid_size": grid_size, "p_unknown": p_unknown, "min_cov": min_cov,
              "contam_eps": contam_eps}
    model_kw = {"contam_eps": contam_eps}
    if do_calibrate:
        _log("calibration with pseudo-plasmids")
        chunks_of = fragment_chunks(frags, chunk_size, chunk_index)
        cal = calibrate(L_chunks, chunk_bin, H, group_modtype, grid, frags, tabs["L_frag"], chunks_of,
                        p_cal, rho_base=rho_base, H_frag=tabs["H_frag"], allow_sharpening=allow_sharpening,
                        log=_log, **model_kw)
        names = dict(enumerate(bins))
        cal["lookalikes"] = cal["lookalikes"].select(
            pl.col("bin_idx").replace_strict(names, return_dtype=pl.Utf8).alias("bin"),
            pl.col("nearest_bin_idx").replace_strict(names, return_dtype=pl.Utf8).alias("nearest_bin"),
            "max_phi_difference", "has_lookalike")
        cal["fragments"] = cal["fragments"].with_columns(
            pl.col("bin_idx").replace_strict(names, return_dtype=pl.Utf8).alias("bin"),
            pl.col("contig").replace_strict(dict(enumerate(asm.names)), return_dtype=pl.Utf8)).drop("bin_idx")
        for name in ("summary", "tau_fit", "tau_groups", "temperature_fit", "reliability", "lookalikes",
                     "assignability", "fragments"):
            _write(cal[name], out / "calibration" / f"{name}.tsv")
        tau = cal["tau"] if tau is None else tau
        temperature = cal["temperature"] if temperature is None else temperature
        params["fitted_temperature"] = cal["fitted_temperature"]
        params["n_fragments"] = frags.height
        warm = {"init_w": cal["model"].w, "init_pi": cal["model"].raw_pi}
        unknown_params = (cal["eta"], cal["lam"])
        params.update(unknown_eta=cal["eta"], unknown_lam=cal["lam"])
        del cal
        _log(f"calibrated: tau={'per-group' if np.ndim(tau) else f'{tau:g}'}, temperature={temperature:g}")
    else:
        warm, unknown_params = {}, (None, 1.0)
    tabs = None
    tau = 100.0 if tau is None else tau
    temperature = 1.0 if temperature is None else temperature

    model = FittedModel(L_chunks, chunk_bin, H, group_modtype, grid, tau, **warm, **model_kw)
    model.set_unknown_params(*unknown_params)
    p_foreign = 1 - model.w
    _write(pl.DataFrame({"contig": [asm.names[i] for i in chunk_contig], "start": chunk_no * chunk_size,
                         "bin": [bins[b] for b in chunk_bin], "p_foreign": p_foreign}),
           out / "host_chunk_outliers.tsv")
    _log(f"robust host profiles: {int((p_foreign > 0.5).sum())} of {len(p_foreign)} chunks flagged as foreign to their bin")

    # MTase annotations -> rho
    p_present = np.exp(model.log_prof)[..., grid.x > 0.5].sum(-1)
    rho_params, rho_report = RhoParams(base=rho_base), {}
    if mtases:
        ann = read_mtases(mtases)
        rho_params, rho_report = estimate_rho(bin_levels(ann, {c: cb[c] for c in host_contigs}, bins, motifs,
                                                         group_of, group_modtype), p_present, base=rho_base)
        levels_t = annotation_levels(ann, targets, motifs, group_of, group_modtype)
        _log(f"MTase annotations: {int((levels_t > 0).any(1).sum())} targets annotated; "
             f"rho motif={rho_params.motif:.2f} modtype={rho_params.modtype:.2f} base={rho_params.base:.3f}")
    else:
        levels_t = np.zeros((len(targets), n_groups), np.int8)
    rho_t = rho_params.matrix(levels_t)
    # own-MTase explanations are for plasmids: a chromosome fragment is methylated by its genome's
    # MTases (they act in trans), so it cannot carry a methylome of its own
    rho_t[~is_plasmid] = 0.0

    # plasmid-specific delta and tau_p
    fit_mask = is_plasmid & (n_sites >= 20)
    if (delta is None or tau_p is None) and fit_mask.sum() >= min_plasmids_for_fit:
        d_fit, t_fit, pfit = fit_plasmid_params(model, L_t, rho_t, p_cal, fit_mask, rng=rng, log=_log)
        _write(pfit, out / "plasmid_param_fit.tsv")
        delta = d_fit if delta is None else delta
        tau_p = t_fit if tau_p is None else tau_p
    delta = 1.0 if delta is None else delta
    tau_p = tau if tau_p is None else tau_p
    model.set_plasmid_params(delta, tau_p)
    params.update(tau=tau, temperature=temperature, delta=delta, tau_p=tau_p, rho=rho_report or
                  {"rho_motif": rho_params.motif, "rho_modtype": rho_params.modtype, "rho_base": rho_params.base})

    # score: plasmid candidates with plasmid parameters, everything else with chromosome ones; binned
    # contigs leave-one-out, so their own bin's profile is built without them
    sc = model.score(L_t, rho_t, plasmid=False)
    if is_plasmid.any():
        idx = np.flatnonzero(is_plasmid)
        sc_p = model.score(L_t[idx], rho_t[idx], plasmid=True)
        for f in ("S", "Su", "Au", "O", "Pwu", "Lp"):
            getattr(sc, f)[idx] = getattr(sc_p, f)
        del sc_p
    sole_contig, loo = set(), {}
    bin_chunks = np.bincount(chunk_bin, minlength=H)
    own_chunks: dict[int, list[int]] = {}
    for u, c in enumerate(chunk_contig):
        own_chunks.setdefault(int(c), []).append(u)
    for i, c in enumerate(targets):
        if c not in host_set:
            continue
        own = np.array(own_chunks.get(cid[c], []), dtype=np.int64)
        b = bin_idx[cb[c]]
        if len(own) and len(own) == bin_chunks[b]:
            sole_contig.add(c)  # leave-one-out would empty the bin; keep the full (non-independent) score
            continue
        model.loo_rescore(sc, i, b, own, rho_t[i])
        loo[i] = (b, own)

    # bin refinement: contamination of binned contigs, inclusion of unbinned ones
    given = np.array([bin_idx[cb[c]] if c in cb else -1 for c in targets])
    testable = np.array([c in host_set and c not in sole_contig and n_sites[i] > 0 for i, c in enumerate(targets)])
    ref = refine(sc.S, sc.Su, given, testable, temperature, contamination_threshold, include_threshold,
                 p_unknown=p_unknown)
    p_unknown = ref.p_unknown
    params.update(p_unknown=p_unknown, contamination_rate=ref.eps)
    P = posterior(sc.S, sc.Su, p_unknown, temperature)
    write_outputs(out, targets, lengths, given, bins, ref, n_sites, sole_contig, P)
    _log(f"refinement: p_unknown={p_unknown:.3f}, contamination rate={ref.eps:.3f}; "
         f"{int(ref.contaminant.sum())} of {int(testable.sum())} testable binned contigs flagged, "
         f"{int(((ref.include_bin >= 0) & (given < 0)).sum())} of {int((given < 0).sum())} unbinned included, "
         f"{int(((ref.include_bin >= 0) & (given >= 0)).sum())} contaminants moved to another bin")
    labels = bins + [UNKNOWN]
    logl = np.concatenate([sc.S, sc.Su[:, None]], 1) / temperature
    map_idx = P.argmax(1)
    hosted = map_idx < H
    truth = np.where(hosted, map_idx, 0)
    _log(f"scored {len(targets)} targets: {int(hosted.sum())} with a bin as MAP")

    # assignability: margins for units with a host MAP, detectability for units with MAP "unknown"
    margin = np.full(len(targets), np.nan); nearest = np.zeros(len(targets), int); vs_unk = np.full(len(targets), np.nan)
    detect = np.zeros(len(targets))
    for flag in (True, False):
        sel = np.flatnonzero((is_plasmid == flag) & hosted)
        if len(sel):
            margin[sel], nearest[sel], vs_unk[sel] = margin_vs_competitors(
                model, H1_t[sel], H0_t[sel], truth[sel], flag, temperature)
        sel = np.flatnonzero((is_plasmid == flag) & ~hosted & (n_sites > 0))
        if len(sel):
            detect[sel] = detectability(model, H1_t[sel], H0_t[sel], flag, temperature,
                                        n_sites_group=n_sites_group[sel], top=detect_top)[0]
    _log("assignability computed")

    # goodness of fit per group, for the MAP explanation
    rows_h = np.flatnonzero(hosted)
    A_map, Pw_map = sc.Au.copy(), sc.Pwu.copy()
    if len(rows_h):
        A_h, Pw_h = model.host_terms(sc, rows_h, hosts=truth[rows_h])
        for j, i in enumerate(rows_h):
            if i in loo and loo[i][0] == truth[i]:
                prof = model.loo_profile(*loo[i])
                A_h[j:j + 1], Pw_h[j:j + 1] = model.host_terms(sc, np.array([i]), log_prof=prof)
        A_map[rows_h], Pw_map[rows_h] = A_h, Pw_h
    misfit = sc.Au - A_map  # log BF: "some community host" vs MAP host, per group
    phi_obs = _batched_summary(grid, lambda sl: L_t[sl].astype(np.float64), len(targets))

    def pred(sl):
        lp = np.where(hosted[sl, None, None], model.log_prof[truth[sl]], model.log_pi[None])
        return propagate(lp, model.Tp)
    phi_pred = _batched_summary(grid, pred, len(targets))

    rows, long_rows = [], []
    for i, c in enumerate(targets):
        order = np.argsort(-P[i])
        b1, b2 = order[0], order[1] if len(order) > 1 else order[0]
        cset = credible_set(P[i], labels)
        if c in sole_contig:
            status = "sole_contig_of_bin"  # scored against a profile built from itself: not an independent test
        elif b1 < H:
            status = "assigned" if cset == [labels[b1]] else "ambiguous"
        else:
            status = "host_not_binned" if detect[i] >= uninformative_log_bf else "uninformative"
        has = n_sites_group[i] > 0
        own = [group_label[g] for g in np.flatnonzero(has & (Pw_map[i] > 0.5))]
        bad = [group_label[g] for g in np.flatnonzero(has & (misfit[i] > misfit_log_bf) & (Pw_map[i] <= 0.5))]
        rows.append({
            "contig": c, "length": lengths[c], "binned_as": cb.get(c, ""), "n_sites": int(n_sites[i]),
            "status": status, "assignment": labels[b1], "posterior": P[i, b1], "p_unknown": P[i, -1],
            "credible_set_95": ",".join(cset), "runner_up": labels[b2],
            "log_bf_best_vs_runner_up": logl[i, b1] - logl[i, b2], "log_bf_best_vs_unknown": logl[i, b1] - logl[i, -1],
            "expected_log_bf_vs_nearest_bin": margin[i] if b1 < H else np.nan,
            "nearest_competitor": bins[nearest[i]] if b1 < H and H > 1 else "",
            "expected_log_bf_vs_unknown": vs_unk[i] if b1 < H else np.nan,
            "detectability": detect[i], "own_mtase_motifs": ";".join(own), "misfit_motifs": ";".join(bad),
            "mtase_annotated": bool((levels_t[i] > 0).any()),
        })
        long_rows.extend({"contig": c, "candidate": labels[j], "posterior": P[i, j], "loglik": logl[i, j]}
                         for j in order if P[i, j] > 1e-6)
    _write(pl.DataFrame(rows), out / "assignments.tsv")
    _write(pl.DataFrame(long_rows), out / "posteriors.tsv", 6)
    ui, gi = np.nonzero(n_sites_group)
    _write(pl.DataFrame({
        "contig": pl.Series(targets)[ui], "explained_by": pl.Series(labels)[map_idx[ui]], "group": gi,
        "motifs": pl.Series(group_label)[gi], "n_sites": n_sites_group[ui, gi],
        "phi_observed": phi_obs[0][ui, gi], "phi_observed_lo95": phi_obs[1][ui, gi],
        "phi_observed_hi95": phi_obs[2][ui, gi], "phi_expected": phi_pred[0][ui, gi],
        "phi_expected_lo95": phi_pred[1][ui, gi], "phi_expected_hi95": phi_pred[2][ui, gi],
        "misfit_log_bf": misfit[ui, gi] / temperature, "p_own_mtase": Pw_map[ui, gi],
        "annotation_level": levels_t[ui, gi].astype(np.int64)}), out / "motif_fit.tsv")

    mean, lo, hi, pp = grid.summarize(np.concatenate([model.log_prof, model.log_pi[None]], 0))
    nb = H + 1
    _write(pl.DataFrame({
        "bin": np.repeat(bins + ["_population_prior_"], n_groups), "group": np.tile(np.arange(n_groups), nb),
        "motifs": np.tile(group_label, nb), "phi_mean": mean.ravel(), "phi_lo95": lo.ravel(), "phi_hi95": hi.ravel(),
        "p_phi_gt_0.5": pp.ravel()}), out / "host_profiles.tsv")
    (out / "params.json").write_text(json.dumps(params, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    _log(f"done -> {out}")
    return out
