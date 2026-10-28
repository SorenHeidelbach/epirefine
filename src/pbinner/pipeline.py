"""End-to-end run: motif groups -> site cache -> caller model -> robust bin profiles -> calibration
-> scoring every contig -> contamination and inclusion decisions.

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

from . import io as pio
from .calibrate import (outlier_rates, DEFAULT_LENGTHS, DEFAULT_SIGMAS, DEFAULT_TAUS, calibrate, contig_chunks, fragment_chunks,
                        fragments_per_bin, select_fragments)
from .detection import detection_table, fit_detection_by_bin
from .grid import Grid
from .assignability import contig_evidence
from .model import FittedModel, posterior
from .motifs import build_records, describe_groups, group_motifs, noisy_groups, redundancy_weights
from .redundancy import inclusion_redundancy
from .refine import bin_evidence_table, refine, write_outputs
from .sites import EncodedAssembly, extract_sites, unit_tables



def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def _write(df: pl.DataFrame, path, precision=4):
    df.write_csv(path, separator="\t", float_precision=precision)


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


def run(assembly, pileup, contig_bin, motifs_path, out, min_cov=1, chunk_size=50_000, grid_size=64,
        p_unknown=None, contamination_threshold=0.5, include_threshold=0.9, min_methylome_difference=0.4,
        max_redundancy=0.3, min_evidence=3.0, min_contig_coverage=5.0, min_contig_length=1000, motif_noise_filter=True, group_weights=True, include_threshold_unmethylated=0.95, tau_mode="outlier", outlier_tau=10.0, do_calibrate=True, frag_lengths=DEFAULT_LENGTHS, frags_per_bin=30, frag_budget=6000,
        tau=None, temperature=None, contam_eps=0.05, allow_sharpening=False, min_jaccard=0.9,
        variation="beta", group_sample_bp=50e6, workers=None, cache_dir=None, write_motif_fit=False, seed=1):
    out = Path(out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cache = Path(cache_dir) if cache_dir else out / "cache"

    contigs = pio.read_fasta(assembly)
    lengths = {c: len(s) for c, s in contigs.items()}
    cb = {c: b for c, b in pio.read_contig_bin(contig_bin).items() if c in contigs}
    bins = sorted(set(cb.values()))
    bin_idx = {b: i for i, b in enumerate(bins)}
    host_contigs = sorted(cb)
    host_set = set(host_contigs)
    # every contig is scored: binned ones leave-one-out (contamination), unbinned ones against all bins
    # (inclusion)
    targets = sorted(contigs)
    H = len(bins)
    _log(f"{len(contigs)} contigs, {H} bins, {len(host_contigs)} host contigs, {len(targets)} targets")

    # motif groups, from the records of a random sample of the sequence
    motifs = pio.read_motifs(motifs_path)
    relevant = sorted(host_set | set(targets))
    sample = _sample_contigs(contigs, relevant, group_sample_bp, rng)
    records = build_records(sample, motifs)
    group_of = group_motifs(records, motifs, min_jaccard)
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
    _log(f"{len(motifs)} motifs -> {n_groups} groups (overlap from {sum(map(len, sample.values())) / 1e6:.0f} Mb)")

    # pass 1: pileup -> site cache (reused when inputs are unchanged)
    man = _manifest(assembly=assembly, pileup=pileup, motifs=[m.key for m in motifs], group_of=group_of,
                    motif_rank=motif_rank, relevant=len(relevant), shared_sites="most_specific_group")
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

    # contig coverage (median reads per motif site, all sites): contigs below min_contig_coverage get
    # no decision at all (status low_coverage) and don't feed the calibration fragments. Against Hi-C
    # on ZymoFecal, flags on contigs below 5x were confirmed for 12.5% (81% at >= 10x).
    cov_of = dict(pl.scan_parquet(parts).group_by("contig").agg(pl.col("n").median()).collect().iter_rows())
    # motif groups whose methylation is partial on many contigs of many genomes are caller noise:
    # they are dropped from every pass below (caller fit, likelihood tables, evidence)
    drop = []
    if motif_noise_filter:
        rich = [cid[c] for c in host_contigs if cov_of.get(cid[c], 0.0) >= 10]
        cb_id = pl.DataFrame({"contig": [cid[c] for c in host_contigs], "bin": [bin_idx[cb[c]] for c in host_contigs]},
                             schema={"contig": pl.Int32, "bin": pl.Int32})
        noise = noisy_groups(parts, cb_id, rich)
        drop = noise.filter("noisy")["group"].to_list()
        _write(noise.join(describe_groups(motifs, group_of).select(pl.col("group").cast(pl.Int32), "mod_type", "motifs"), on="group", how="left"),
               out / "motif_noise.tsv")
        _log(f"motif noise filter: {len(drop)} of {noise.height} measured groups dropped (intermediate calls at their sites in "
             f">= 20% of bins): {', '.join(str(g) for g in drop)}")
    keep_groups = ~pl.col("group").is_in(drop)
    counts = (pl.scan_parquet(parts).filter((pl.col("n") >= min_cov) & keep_groups)
              .group_by("group", "n", "k").agg(pl.len().alias("w")).collect())
    _log(f"{int(counts['w'].sum())} sites with coverage >= {min_cov}")
    _write(describe_groups(motifs, group_of, counts), out / "motif_groups.tsv")
    # caller model fitted with the bins as labels (a motif is methylated in some genomes, not in
    # others); the pooled fit collapses when few of many bins methylate a motif (dev log D19)
    host_bin = pl.DataFrame({"contig": [cid[c] for c in host_contigs], "bin": [bin_idx[cb[c]] for c in host_contigs]},
                            schema={"contig": pl.Int32, "bin": pl.Int32})
    counts_bin = (pl.scan_parquet(parts).filter((pl.col("n") >= min_cov) & keep_groups).join(host_bin.lazy(), on="contig")
                  .group_by("group", "bin", "n", "k").agg(pl.len().alias("w")).collect())
    # one methyltransferase seen through several motif variants is one piece of evidence (weights 1 / cluster size)
    group_w = np.ones(n_groups)
    if group_weights:
        group_w, wtab = redundancy_weights(counts_bin, motifs, group_of, n_groups)
        wtab = wtab.join(describe_groups(motifs, group_of).select(pl.col("group").cast(pl.Int32), "motifs"), on="group", how="left")
        _write(wtab, out / "group_weights.tsv")
        _log(f"redundant motif groups: {int((group_w < 1).sum())} groups in {wtab.filter(pl.col('cluster_size') > 1)['cluster'].n_unique()} "
             f"clusters down-weighted (effective groups {group_w.sum():.0f} of {n_groups})")
    det = fit_detection_by_bin(counts_bin, dict(enumerate(group_modtype)),
                               workers=workers or min(16, os.cpu_count() or 1))
    _write(detection_table(det), out / "detection_model.tsv")
    _log("detection model fitted")

    # between-contig variation per motif group: contigs match their bin's level, except for rare, large
    # departures (outlier kernel; rates from binned contigs >= 10x)
    tau_spread, outlier = None, None
    if tau_mode == "outlier" and tau is None and variation == "beta":
        rich_host = pl.DataFrame({"contig": [cid[c] for c in host_contigs if cov_of.get(cid[c], 0.0) >= 10],
                                  "bin": [bin_idx[cb[c]] for c in host_contigs if cov_of.get(cid[c], 0.0) >= 10]},
                                 schema={"contig": pl.Int32, "bin": pl.Int32})
        st = (pl.scan_parquet(parts).filter((pl.col("n") >= 5) & keep_groups).join(rich_host.lazy(), on="contig")
              .group_by("bin", "contig", "group").agg(pl.len().alias("N"), (pl.col("k") / pl.col("n") > 0.5).sum().alias("K"))
              .collect())
        eps, eps0, etab = outlier_rates(st, n_groups)
        _write(etab.join(describe_groups(motifs, group_of).select(pl.col("group").cast(pl.Int32), "motifs"), on="group", how="left"),
               out / "outlier_rates.tsv")
        outlier = (eps, outlier_tau)
        tau_spread = float("inf")
        _log(f"outlier kernel: overall departure rate {eps0:.4f}; per group "
             f"{', '.join(f'{int((eps == e).sum())} at {e:g}' for e in np.unique(eps))}; departures ~ Beta(tau={outlier_tau:g})")
    # units: host chunks (from contig lengths), targets, calibration fragments
    grid = Grid(grid_size, kernel=variation)
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
        hc_ok = [c for c in host_contigs if cov_of.get(cid[c], 0.0) >= min_contig_coverage]
        _log(f"calibration fragments from {len(hc_ok)} of {len(host_contigs)} host contigs with >= {min_contig_coverage:g}x")
        hc = pl.DataFrame({"contig": [cid[c] for c in hc_ok], "bin_idx": [bin_idx[cb[c]] for c in hc_ok],
                           "length": [lengths[c] for c in hc_ok]}, schema={"contig": pl.Int32, "bin_idx": pl.Int64, "length": pl.Int64})
        per_bin = fragments_per_bin(frag_budget, len(frag_lengths), H, frags_per_bin)
        frags = select_fragments(hc, frag_lengths, per_bin, rng)
        _log(f"{frags.height} calibration fragments ({per_bin} per bin and length)")

    # pass 2: site cache -> likelihood tables
    tabs = unit_tables(parts, n_groups=n_groups, det=det, grid_size=grid_size, min_cov=min_cov, drop_groups=drop,
                       chunk_offset=chunk_offset, chunk_size=chunk_size, n_chunks=n_chunks_all, target_of=target_of,
                       n_targets=len(targets), frags=frags, workers=workers, log=_log)
    if group_weights and (group_w < 1).any():
        wf = group_w.astype(np.float32)[None, :, None]
        for key in ("L_chunks", "L_t", "L_frag"):
            tabs[key] *= wf
        tabs["H_frag"] = (tabs["H_frag"][0] * wf, tabs["H_frag"][1] * wf)
    keep = np.flatnonzero(tabs["chunk_sites"] > 0)  # chunks without sites carry no information
    remap = np.full(n_chunks_all, -1, np.int64); remap[keep] = np.arange(len(keep))
    L_chunks = tabs.pop("L_chunks")[keep]
    chunk_bin, chunk_contig, chunk_no = chunk_bin_all[keep], chunk_contig[keep], chunk_no[keep]
    L_t, n_sites, n_sites_group = tabs["L_t"], tabs["n_sites_t"], tabs["n_sites_group_t"]
    _log(f"{len(keep)} host chunks of {chunk_size} bp; tables {L_chunks.nbytes / 1e9:.1f} GB (chunks), "
         f"{L_t.nbytes / 1e9:.1f} GB (contigs)")

    def chunk_index(contig_id, c):
        return int(remap[chunk_offset[contig_id] + c]) if chunk_offset[contig_id] >= 0 and c < n_chunks_of[asm.names[contig_id]] else -1

    p_cal = 0.1 if p_unknown is None else p_unknown  # prior used before p_unknown is estimated (refinement)
    params = {"chunk_size": chunk_size, "grid_size": grid_size, "p_unknown": p_unknown, "min_cov": min_cov,
              "contam_eps": contam_eps}
    model_kw = {"contam_eps": contam_eps, "outlier": outlier}
    if do_calibrate:
        _log("calibration with pseudo-contigs")
        chunks_of = fragment_chunks(frags, chunk_size, chunk_index)
        cal = calibrate(L_chunks, chunk_bin, H, group_modtype, grid, frags, tabs["L_frag"], chunks_of,
                        p_cal, H_frag=tabs["H_frag"], allow_sharpening=allow_sharpening,
                        leave_out=contig_chunks(frags["contig"].to_numpy(), chunk_contig),
                        taus=DEFAULT_TAUS if variation == "beta" else DEFAULT_SIGMAS, n_sites_frag=tabs["frag_sites"],
                        fixed_tau=tau_spread,
                        log=_log, **model_kw)
        names = dict(enumerate(bins))
        cal["lookalikes"] = cal["lookalikes"].select(
            pl.col("bin_idx").replace_strict(names, return_dtype=pl.Utf8).alias("bin"),
            pl.col("nearest_bin_idx").replace_strict(names, return_dtype=pl.Utf8).alias("nearest_bin"),
            "max_phi_difference", "has_lookalike")
        cal["fragments"] = cal["fragments"].with_columns(
            pl.col("bin_idx").replace_strict(names, return_dtype=pl.Utf8).alias("bin"),
            pl.col("contig").replace_strict(dict(enumerate(asm.names)), return_dtype=pl.Utf8)).drop("bin_idx")
        for name in ("summary", "tau_fit", "tau_groups", "temperature_fit", "temperature_by_length", "reliability", "lookalikes",
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
    if tau is None and tau_spread is not None:
        tau = tau_spread
    tau = (100.0 if variation == "beta" else 0.3) if tau is None else tau
    temperature = 1.0 if temperature is None else temperature

    model = FittedModel(L_chunks, chunk_bin, H, group_modtype, grid, tau, **warm, **model_kw)
    model.set_unknown_params(*unknown_params)
    p_foreign = 1 - model.w
    _write(pl.DataFrame({"contig": [asm.names[i] for i in chunk_contig], "start": chunk_no * chunk_size,
                         "bin": [bins[b] for b in chunk_bin], "p_foreign": p_foreign}),
           out / "host_chunk_outliers.tsv")
    _log(f"robust host profiles: {int((p_foreign > 0.5).sum())} of {len(p_foreign)} chunks flagged as foreign to their bin")

    params.update(tau=tau, temperature=temperature)

    # score every contig against every bin; binned contigs leave-one-out, so their own bin's profile
    # is built without them
    sc = model.score(L_t)
    sole_contig = set()
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
            sole_contig.add(c)  # leave-one-out would empty the bin: not an independent test
            continue
        model.loo_rescore(sc, i, b, own)

    # bin refinement: contamination of binned contigs, inclusion of unbinned ones
    given = np.array([bin_idx[cb[c]] if c in cb else -1 for c in targets])
    # expected methylation evidence (assignability) per contig, for its own bin (binned) or its best
    # bin (unbinned), and its coverage (median reads per motif site)
    best_bin = np.where(given >= 0, given, sc.S.argmax(1))
    evidence, _, coverage = contig_evidence(parts, {cid[c]: i for i, c in enumerate(targets)}, best_bin, model, det,
                                            grid, float(np.median(np.atleast_1d(temperature))), min_cov,
                                            group_weights=group_w if group_weights else None, drop_groups=drop)
    _log(f"methylation evidence: {int((np.nan_to_num(evidence, nan=0.0) < min_evidence).sum())} of {len(targets)} "
         f"contigs below {min_evidence:g} (expected log BF vs a community genome)")
    # contamination calls only on contigs with enough coverage: against Hi-C on ZymoFecal, flags were
    # confirmed for 81% of contigs at >= 10x, 36% at 5-9x and 12.5% below 5x (most sites there have
    # 1-2 reads); inclusions were right at every coverage, so they are not restricted
    coverage = np.array([cov_of.get(cid[c], 0.0) for c in targets])
    low_cov = coverage < min_contig_coverage
    # contigs shorter than min_contig_length get no decision either (status too_short): a 1 kb contig
    # holds a handful of motif sites, too few to tell genomes apart
    short = np.array([lengths[c] < min_contig_length for c in targets])
    _log(f"length: {int(short.sum())} of {len(targets)} contigs shorter than {min_contig_length} bp: no decision (too_short)")
    testable = np.array([c in host_set and c not in sole_contig and n_sites[i] > 0 for i, c in enumerate(targets)])
    _log(f"coverage: {int((low_cov & (given >= 0)).sum())} of {int((given >= 0).sum())} binned and "
         f"{int((low_cov & (given < 0)).sum())} of {int((given < 0).sum())} unbinned contigs below "
         f"{min_contig_coverage:g}x median site coverage: no decision (low_coverage)")
    # effect size a contamination call needs (D20): over motif groups, the largest distance between
    # the bin's methylated fraction and the contig's 95% credible interval for its own. Short
    # contigs with a real difference have wide intervals that still clear the bin's level; long
    # contigs with a small, precise shift do not.
    phi_obs = _batched_summary(grid, lambda sl: L_t[sl].astype(np.float64), len(targets))
    phi_bin = grid.summarize(model.log_prof)[0]
    diff = np.zeros(len(targets))
    has = given >= 0
    pb = phi_bin[given[has]]
    gap = np.maximum(np.maximum(phi_obs[1][has] - pb, pb - phi_obs[2][has]), 0.0)
    diff[has] = np.where(n_sites_group[has] > 0, gap, 0.0).max(1)
    bin_ev = bin_evidence_table(bins, grid.summarize(model.log_prof), n_sites_group, given)
    unmeth_bins = (bin_ev["status"] == "no_methylation_detected").to_numpy()
    ref = refine(sc.S, sc.Su, given, testable, temperature, contamination_threshold, include_threshold,
                 methylome_difference=diff, min_difference=min_methylome_difference,
                 p_unknown=p_unknown, decidable=~low_cov & ~short, unmethylated_bins=unmeth_bins, include_threshold_unmethylated=include_threshold_unmethylated)
    # don't include contigs whose sequence the target bin already holds (strain variants, haplotigs):
    # methylation can't see redundancy (D21)
    redundancy = np.full(len(targets), np.nan)
    if max_redundancy < 1:
        inc = np.flatnonzero(ref.include_bin >= 0)
        members: dict[int, list[int]] = {}
        for i, c in enumerate(targets):
            if given[i] >= 0 and not ref.contaminant[i]:
                members.setdefault(int(given[i]), []).append(cid[c])
        redundancy[inc] = inclusion_redundancy(asm, [(cid[targets[i]], int(ref.include_bin[i])) for i in inc], members)
        drop = inc[redundancy[inc] >= max_redundancy]
        ref.include_bin[drop] = -1
        _log(f"redundancy: {len(drop)} of {len(inc)} inclusions dropped (>= {max_redundancy:g} of their k-mers already in the bin)")
    p_unknown = ref.p_unknown
    params.update(p_unknown=p_unknown, contamination_rate=ref.eps)
    P = posterior(sc.S, sc.Su, p_unknown, temperature)
    write_outputs(out, targets, lengths, given, bins, ref, n_sites, sole_contig, P, diff, redundancy,
                  evidence=evidence, min_evidence=min_evidence, coverage=coverage, low_coverage=low_cov, too_short=short)
    _write(bin_ev, out / "bin_evidence.tsv")
    _log(f"refinement: p_unknown={p_unknown:.3f}, contamination rate={ref.eps:.3f}; "
         f"{int(ref.contaminant.sum())} of {int(testable.sum())} testable binned contigs flagged, "
         f"{int(((ref.include_bin >= 0) & (given < 0)).sum())} of {int((given < 0).sum())} unbinned included, "
         f"{int(((ref.include_bin >= 0) & (given >= 0)).sum())} contaminants moved to another bin")
    if write_motif_fit:
        # per contig and motif group: the contig's methylated fraction next to its bin's (binned
        # contigs) or its best alternative bin's (unbinned contigs)
        ref_bin = np.where(given >= 0, given, np.where(ref.best_alt < H, ref.best_alt, -1))
        ui, gi = np.nonzero(n_sites_group)
        rb = ref_bin[ui]
        _write(pl.DataFrame({
            "contig": pl.Series(targets)[ui], "group": gi, "motifs": pl.Series(group_label)[gi],
            "n_sites": n_sites_group[ui, gi], "phi_observed": phi_obs[0][ui, gi],
            "phi_observed_lo95": phi_obs[1][ui, gi], "phi_observed_hi95": phi_obs[2][ui, gi],
            "reference_bin": [bins[b] if b >= 0 else "" for b in rb],
            "phi_bin": np.where(rb >= 0, phi_bin[np.maximum(rb, 0), gi], np.nan)}), out / "motif_fit.tsv")

    mean, lo, hi, pp = grid.summarize(np.concatenate([model.log_prof, model.log_pi[None]], 0))
    nb = H + 1
    _write(pl.DataFrame({
        "bin": np.repeat(bins + ["_population_prior_"], n_groups), "group": np.tile(np.arange(n_groups), nb),
        "motifs": np.tile(group_label, nb), "phi_mean": mean.ravel(), "phi_lo95": lo.ravel(), "phi_hi95": hi.ravel(),
        "p_phi_gt_0.5": pp.ravel()}), out / "host_profiles.tsv")
    (out / "params.json").write_text(json.dumps(params, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    _log(f"done -> {out}")
    return out
