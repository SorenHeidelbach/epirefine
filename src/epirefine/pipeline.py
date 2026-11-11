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
from .calibrate import (DEFAULT_LENGTHS, DEFAULT_SIGMAS, DEFAULT_TAUS, calibrate, contig_chunks, fragment_chunks,
                        fragments_per_bin, select_fragments)
from .detection import detection_table, fit_detection_by_bin
from .grid import Grid
from .assignability import contig_evidence
from .model import FittedModel, posterior
from .motifs import build_records, describe_groups, group_motifs, noisy_groups, redundancy_weights
from .redundancy import inclusion_redundancy
from .refine import bin_evidence_table, refine, regional_pattern, write_outputs
from .sites import EncodedAssembly, extract_sites, unit_tables



def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def _write(df: pl.DataFrame, path, precision=4):
    df.write_csv(path, separator="\t", float_precision=precision)


def _sample_names(lengths: dict[str, int], names, sample_bp: float, rng) -> list[str]:
    """Random contigs up to sample_bp of sequence (all of them for small assemblies), in sampling order."""
    names = list(names)
    if sum(lengths[c] for c in names) <= sample_bp:
        return names
    out, acc = [], 0
    for i in rng.permutation(len(names)):
        c = names[i]
        out.append(c); acc += lengths[c]
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
    """grid.summarize over units in batches; logp_fn(slice) -> log densities [b, M, G].

    A batch is at most `batch` units and at most 200,000 (unit, group) cells (about five float64
    [b, M, G] arrays are alive at once: 4.7 GB for 913 groups at 2000 units)."""
    if n:
        batch = min(batch, max(1, 200_000 // logp_fn(slice(0, 1)).shape[1]))
    parts = [grid.summarize(logp_fn(slice(s, s + batch))) for s in range(0, n, batch)]
    return [np.concatenate([p[i] for p in parts]) for i in range(4)]


def run(assembly, pileup, contig_bin, motifs_path, out, min_cov=1, chunk_size=50_000, grid_size=64,
        p_unknown=None, contamination_threshold=0.5, include_threshold=0.9, min_methylome_difference=0.4,
        max_redundancy=0.3, min_evidence=3.0, min_contig_coverage=5.0, min_contig_length=1000, motif_noise_filter=True, group_weights=True, include_threshold_unmethylated=0.95, regional_model=True, regional_inclusion=False, do_calibrate=True, frag_lengths=DEFAULT_LENGTHS, frags_per_bin=30, frag_budget=6000,
        tau=None, temperature=None, contam_eps=0.05, allow_sharpening=False, min_jaccard=0.9,
        variation="beta", group_sample_bp=50e6, workers=None, cache_dir=None, write_motif_fit=False, seed=1):
    out = Path(out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cache = Path(cache_dir) if cache_dir else out / "cache"

    # the assembly is streamed (lengths, then the motif-grouping sample, then the encoded copy on disk): held
    # whole as Python strings it took about three times its size, 53 GB for an 18.9 GB soil assembly
    lengths = pio.fasta_lengths(assembly)
    cb = {c: b for c, b in pio.read_contig_bin(contig_bin).items() if c in lengths}
    bins = sorted(set(cb.values()))
    bin_idx = {b: i for i, b in enumerate(bins)}
    host_contigs = sorted(cb)
    host_set = set(host_contigs)
    # every contig is scored: binned ones leave-one-out (contamination), unbinned ones against all bins
    # (inclusion)
    targets = sorted(lengths)
    H = len(bins)
    _log(f"{len(lengths)} contigs, {H} bins, {len(host_contigs)} host contigs, {len(targets)} targets")

    # motif groups, from the records of a random sample of the sequence
    motifs = pio.read_motifs(motifs_path)
    relevant = sorted(host_set | set(targets))
    sample = pio.fasta_subset(assembly, _sample_names(lengths, relevant, group_sample_bp, rng))
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
        # the old manifest goes first: a rebuild that is killed must not leave it describing parts and an
        # encoded assembly that have been partly rewritten (a later run with the old inputs would reuse them)
        man_path.unlink(missing_ok=True)
        asm = EncodedAssembly.write(pio.iter_fasta(assembly), cache / "asm")
        for f in (cache / "sites").glob("*.parquet") if (cache / "sites").exists() else []:
            f.unlink()
        wanted = None if len(relevant) == len(lengths) else {asm.index[c] for c in relevant}
        parts = extract_sites(asm, pileup, motifs, group_of, motif_rank, cache / "sites", wanted=wanted,
                              workers=workers, log=_log)
        man_path.write_text(json.dumps({"inputs": man, "parts": parts}))
    del sample
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

    # units: host chunks (from contig lengths), targets, calibration fragments
    grid = Grid(grid_size, kernel=variation)
    n_ids = len(asm.names)
    chunk_offset = np.full(n_ids, -1, np.int64)
    n_chunks_of = {c: -(-lengths[c] // chunk_size) for c in host_contigs}
    off = 0
    for c in host_contigs:
        chunk_offset[cid[c]] = off; off += n_chunks_of[c]
    n_host_chunks = off
    # windows of long unbinned contigs (regional model for inclusion), scored like host chunks but kept
    # out of the bin profiles. Off by default: no benchmark gains an inclusion from it, and on ZymoFecal
    # (0.7.0) it included 58 long contigs the contig-level model gives P(unknown) 0.54-1 (42 above
    # 0.99), with Hi-C's best bin agreeing for 7 of 43 (contig-level inclusions: 135 of 267; dev log D40)
    win_contigs = ([c for c in targets if c not in cb and lengths[c] >= 2 * chunk_size and lengths[c] >= min_contig_length
                    and cov_of.get(cid[c], 0.0) >= min_contig_coverage] if regional_model and regional_inclusion else [])
    for c in win_contigs:
        n_chunks_of[c] = -(-lengths[c] // chunk_size)
        chunk_offset[cid[c]] = off; off += n_chunks_of[c]
    n_chunks_all = off
    unit_contigs = list(host_contigs) + win_contigs
    chunk_contig = np.repeat([cid[c] for c in unit_contigs], [n_chunks_of[c] for c in unit_contigs]).astype(np.int64)
    chunk_no = np.concatenate([np.arange(n_chunks_of[c]) for c in unit_contigs]) if unit_contigs else np.zeros(0, int)
    chunk_bin_all = np.concatenate([np.repeat([bin_idx[cb[c]] for c in host_contigs], [n_chunks_of[c] for c in host_contigs]),
                                    np.full(off - n_host_chunks, -1)]).astype(np.int64)
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
    L_all = tabs.pop("L_chunks")
    win = np.arange(n_host_chunks, n_chunks_all)[tabs["chunk_sites"][n_host_chunks:] > 0]
    L_win, win_contig, win_no = L_all[win], chunk_contig[win], chunk_no[win]
    keep = np.flatnonzero(tabs["chunk_sites"][:n_host_chunks] > 0)  # chunks without sites carry no information
    remap = np.full(n_chunks_all, -1, np.int64); remap[keep] = np.arange(len(keep))
    L_chunks = L_all[keep]
    del L_all
    chunk_bin, chunk_contig, chunk_no = chunk_bin_all[keep], chunk_contig[keep], chunk_no[keep]
    L_t, n_sites, n_sites_group = tabs["L_t"], tabs["n_sites_t"], tabs["n_sites_group_t"]
    _log(f"{len(keep)} host chunks of {chunk_size} bp; tables {L_chunks.nbytes / 1e9:.1f} GB (chunks), "
         f"{L_t.nbytes / 1e9:.1f} GB (contigs)")

    def chunk_index(contig_id, c):
        if chunk_offset[contig_id] < 0 or chunk_offset[contig_id] >= n_host_chunks or c >= n_chunks_of[asm.names[contig_id]]:
            return -1
        return int(remap[chunk_offset[contig_id] + c])

    p_cal = 0.1 if p_unknown is None else p_unknown  # prior used before p_unknown is estimated (refinement)
    params = {"chunk_size": chunk_size, "grid_size": grid_size, "p_unknown": p_unknown, "min_cov": min_cov,
              "contam_eps": contam_eps}
    model_kw = {"contam_eps": contam_eps}
    if do_calibrate:
        _log("calibration with pseudo-contigs")
        chunks_of = fragment_chunks(frags, chunk_size, chunk_index)
        cal = calibrate(L_chunks, chunk_bin, H, group_modtype, grid, frags, tabs["L_frag"], chunks_of,
                        p_cal, H_frag=tabs["H_frag"], allow_sharpening=allow_sharpening,
                        leave_out=contig_chunks(frags["contig"].to_numpy(), chunk_contig),
                        taus=DEFAULT_TAUS if variation == "beta" else DEFAULT_SIGMAS, n_sites_frag=tabs["frag_sites"],
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
    # (their Lp rows are the only ones the leave-one-out scores read: the rest is not kept)
    sc = model.score(L_t, lp_rows=[i for i, c in enumerate(targets) if c in host_set])
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
    # long binned contigs: whole contig foreign, or partly the bin's and partly foreign along its length?
    # (chunk-level evidence from the robust host-profile fit; flags on mixed contigs are withdrawn and
    # their foreign segments reported)
    regional = None
    if regional_model:
        llr_chunk = (np.log(np.clip(p_foreign, 1e-12, 1 - 1e-12)) - np.log(np.clip(1 - p_foreign, 1e-12, None))
                     - np.log(contam_eps / (1 - contam_eps)))
        order = np.lexsort((chunk_no, chunk_contig))
        by_contig: dict[int, list[int]] = {}
        for k in order:
            by_contig.setdefault(int(chunk_contig[k]), []).append(int(k))
        tidx = {cid[c]: i for i, c in enumerate(targets)}
        long_u = [u for u in by_contig if len(by_contig[u]) >= 2 and u in tidx and given[tidx[u]] >= 0 and not low_cov[tidx[u]]]
        post, chunk_post, w_flag = (regional_pattern([llr_chunk[by_contig[u]] for u in long_u]) if long_u
                                    else (np.zeros((0, 3)), [], np.array([1.0, 0.0, 0.0])))
        reg_p = np.full((len(targets), 3), np.nan)
        segments = [""] * len(targets)
        mixed = np.zeros(len(targets), bool)
        for j, u in enumerate(long_u):
            i = tidx[u]
            reg_p[i] = post[j]
            ks = by_contig[u]
            fk = [chunk_no[k] for k, pf in zip(ks, chunk_post[j]) if pf >= 0.5]
            segments[i] = ";".join(f"{c * chunk_size}-{min((c + 1) * chunk_size, lengths[targets[i]])}" for c in fk)
            if ref.contaminant[i] and post[j, 2] > post[j, 1]:
                mixed[i] = True
        ref.contaminant[mixed] = False
        ref.include_bin[mixed & (given >= 0)] = -1
        ref.candidate[mixed] = False
        # long unbinned contigs: windows scored against every bin and the unknown genome; the candidate bin
        # is the one expected to explain the most sequence (sum over windows of its posterior among bins and
        # the unknown genome: a strongly methylated genome covering 40% of a chimera must not outvote a
        # weakly methylated one covering 60%, which a summed score did on the chimera benchmark), and the three
        # explanations (all windows the bin's / all unknown / a mixture along the contig) are weighed as
        # for flags. Included when P(the bin explains the contig) x P(that bin among bins) reaches the
        # inclusion threshold.
        reg_inc = 0
        if len(win):
            ws = model.score(L_win)
            T = float(np.median(np.atleast_1d(temperature)))
            Sw, Suw = ws.S / T, ws.Su / T
            by_win: dict[int, list[int]] = {}
            for k in np.lexsort((win_no, win_contig)):
                by_win.setdefault(int(win_contig[k]), []).append(int(k))
            cand_u, llrs, h_star, p_bin = [], [], [], []
            for u, ks in by_win.items():
                i = tidx.get(u)
                if i is None or given[i] >= 0 or len(ks) < 2 or ref.include_bin[i] >= 0 or low_cov[i] or short[i]:
                    continue
                pu = min(max(ref.p_unknown, 1e-3), 1 - 1e-3)
                logits = np.concatenate([Sw[ks] + np.log((1 - pu) / H), (Suw[ks] + np.log(pu))[:, None]], 1)
                pw = np.exp(logits - logits.max(1, keepdims=True)); pw /= pw.sum(1, keepdims=True)
                share = pw[:, :H].sum(0)                                        # expected windows per bin
                h = int(share.argmax())
                robust = np.maximum(Sw[ks], Suw[ks][:, None]).sum(0)
                pb = np.exp(robust - robust.max()); pb /= pb.sum()
                cand_u.append(i); h_star.append(h); p_bin.append(pb[h])
                llrs.append(Suw[ks] - Sw[ks, h])
            if cand_u and ref.p_unknown >= 1 - 1e-6:
                # the model's own estimate says unbinned contigs have no bin here (simulated look-alikes):
                # windows that each lean slightly toward a bin must not reopen that
                _log("regional inclusion: p_unknown = 1, no unbinned contig has a bin; none included")
                cand_u = []
            if cand_u:
                # prior weights tied to the model: p_unknown from the contig-level fit, the share of mixed
                # contigs from the binned ones. Windows that can't tell the bin from an unknown genome
                # (look-alike strains) then leave the decision to p_unknown, as at contig level; weights
                # estimated from the long unbinned contigs alone let 'mixed' absorb them (simulated
                # look-alikes: ~40 inclusions per scenario, 9 wrong).
                pu = min(max(ref.p_unknown, 1e-6), 1 - 1e-6)
                m = float(w_flag[2])
                post_u, cpost, _ = regional_pattern(llrs, weights=((1 - pu) * (1 - m), pu * (1 - m), m))
                for j, i in enumerate(cand_u):
                    # a mixed contig counts toward the bin for the share of its windows the bin explains
                    share_bin = float(np.mean(1 - cpost[j]))
                    p_in = (post_u[j, 0] + post_u[j, 2] * share_bin) * p_bin[j]
                    thr = include_threshold_unmethylated if unmeth_bins[h_star[j]] else include_threshold
                    reg_p[i] = post_u[j]
                    if p_in >= thr:
                        ref.include_bin[i] = h_star[j]; ref.candidate[i] = True; reg_inc += 1
                        u = cid[targets[i]]
                        fk = [win_no[k] for k, pf in zip(by_win[u], cpost[j]) if pf >= 0.5 and post_u[j, 2] > post_u[j, 0]]
                        segments[i] = ";".join(f"{c * chunk_size}-{min((c + 1) * chunk_size, lengths[targets[i]])}" for c in fk)
            _log(f"regional inclusion: {len(cand_u)} long unbinned contigs scored in windows; {reg_inc} included")
        # segment map: for contigs called mixed or included regionally, which bin (or an unknown genome)
        # best explains each run of chunks along the contig, with its posterior. Reported only: nothing is
        # split, so a user can see whether a mixed contig joins two bins' sequences.
        seg_bins = [""] * len(targets)
        T0 = float(np.median(np.atleast_1d(temperature)))
        pu0 = min(max(ref.p_unknown, 1e-3), 1 - 1e-3)
        lbl = bins + ["unknown"]

        def segment_map(L_rows, nos, i):
            sc_ = model.score(L_rows)
            lg = np.concatenate([sc_.S / T0 + np.log((1 - pu0) / H), (sc_.Su / T0 + np.log(pu0))[:, None]], 1)
            pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            best, pbest = pr.argmax(1), pr.max(1)
            runs, start = [], 0
            for k in range(1, len(best) + 1):
                if k == len(best) or best[k] != best[start]:
                    lo, hi = int(nos[start]) * chunk_size, min((int(nos[k - 1]) + 1) * chunk_size, lengths[targets[i]])
                    runs.append(f"{lo}-{hi}:{lbl[best[start]]}({pbest[start:k].mean():.2f})")
                    start = k
            return ";".join(runs)
        for j, u in enumerate(long_u):
            i = tidx[u]
            if mixed[i]:
                ks = by_contig[u]
                seg_bins[i] = segment_map(L_chunks[ks], chunk_no[ks], i)
        if len(win):
            for u, ks in by_win.items():
                i = tidx.get(u)
                if i is not None and given[i] < 0 and ref.include_bin[i] >= 0 and segments[i]:
                    seg_bins[i] = segment_map(L_win[ks], win_no[ks], i)
        regional = {"p": reg_p, "segments": segments, "mixed": mixed, "segment_bins": seg_bins}
        _log(f"regional: {len(long_u)} binned contigs with >= 2 chunks; {int(mixed.sum())} flags withdrawn as mixed "
             f"(partly the bin's, partly foreign along the contig)")
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
                  evidence=evidence, min_evidence=min_evidence, coverage=coverage, low_coverage=low_cov, too_short=short,
                  regional=regional)
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
