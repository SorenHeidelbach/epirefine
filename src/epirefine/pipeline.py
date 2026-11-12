"""End-to-end run: motif groups -> site cache -> caller model -> robust bin profiles -> calibration
-> scoring every contig -> contamination and inclusion decisions.

Data flow at scale: motif groups are found on a random sample of the sequence; one parallel pass
over the pileup writes a compact site cache (sites.extract_sites); a second parallel pass turns it
into per-unit likelihood tables (sites.unit_tables). Nothing proportional to assembly size x
number of motifs is held in memory.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from . import io as pio
from . import runinfo
from .calibrate import (DEFAULT_LENGTHS, DEFAULT_SIGMAS, DEFAULT_TAUS, ETA_REL, ContigScores, CoreOptions, SitesLaw,
                        hidden_relative,
                        calibrate, calibrate_decisions, contig_chunks, fragment_chunks,
                        fragments_per_bin, parse_features, select_fragments)
from .detection import detection_table, fit_detection_by_bin
from .grid import Grid
from .assignability import contig_evidence
from .model import (LOO_BATCH, LOO_THREADS_MIN, FittedModel, _cores, _n_threads, log_bins, logsumexp, posterior, set_tempered_bin_prior, summed_scores, unbinned_state_frequencies, unknown_logit,
                    unknown_score)
from .motifs import (build_records, describe_groups, group_motifs, noisy_groups, redundancy_weights,
                     specificity_ranks)
from .redundancy import inclusion_redundancy
from .refine import (bin_evidence_table, em_weight, flagged_share_per_bin, refine, regional_pattern, suspected_merges,
                     withdraw_as_mixed,
                     write_outputs)
from .sites import EncodedAssembly, extract_sites, same_pairs, unit_tables



def _log(msg):
    runinfo.log(msg)  # elapsed time since the start of the run, peak memory so far, message


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


def _cached_assembly(cache: Path, assembly):
    """The encoded assembly of a site cache that was written from this very file (the path, size and modification
    time its manifest holds), or None. With it, contig lengths and the motif-grouping sample are taken from the
    cache: reading the FASTA for them is two passes over the file (2.6 GB for ZymoFecal: 12 s each when the file
    is in memory, 160 s from a cold disk), and the encoding holds the same names, lengths and bases."""
    try:
        cached = json.loads((cache / "manifest.json").read_text())
        if cached["inputs"].get("assembly") != _manifest(assembly=assembly)["assembly"]:
            return None
        return EncodedAssembly(cache / "asm")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _prefetch_cache(cache: Path):
    """Read the site-cache parts of a manifest in a background thread, so that they are in the page cache when the
    first pass over them (the motif noise filter) starts: on ZymoFecal that pass took 27 s from a cold disk and 10 s
    from memory, and motif grouping before it leaves the disk idle for 15-40 s. Only reads (nothing is kept by this
    process), and only when the parts fit in half the available memory. Returns the thread or None."""
    import threading
    try:
        files = [cache / "sites" / Path(f).name for f in json.loads((cache / "manifest.json").read_text())["parts"]]
        size = sum(f.stat().st_size for f in files)
        with open("/proc/meminfo") as fh:
            avail = next(int(ln.split()[1]) * 1024 for ln in fh if ln.startswith("MemAvailable:"))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None
    if size > avail // 2:
        return None

    def read():
        buf = bytearray(1 << 24)
        for f in files:
            try:
                with open(f, "rb", buffering=0) as fh:
                    while fh.readinto(buf):
                        pass
            except OSError:
                pass
    th = threading.Thread(target=read, daemon=True)
    th.start()
    return th


_LETTERS = np.frombuffer(b"ACGTN", np.uint8)


def _decoded(asm, names) -> dict[str, str]:
    """Sequences of the named contigs from the encoded assembly (anything but A, C, G, T reads N, which is what
    the motif scan makes of it)."""
    return {c: _LETTERS[asm.seq(asm.index[c])].tobytes().decode("ascii") for c in names}


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
    # every unit is summarised from its own row, so the batches run on threads, each a share of `batch` (together
    # they hold what one batch held); it was one thread's pass over the tables of all contigs
    threads = _n_threads()
    step = max(1, batch // threads)
    slices = [slice(s, s + step) for s in range(0, n, step)]
    if threads <= 1 or len(slices) <= 1:
        parts = [grid.summarize(logp_fn(sl)) for sl in slices]
    else:
        with ThreadPoolExecutor(threads) as ex:
            parts = list(ex.map(lambda sl: grid.summarize(logp_fn(sl)), slices))
    return [np.concatenate([p[i] for p in parts]) for i in range(4)]


SPILL_MIN_BYTES = 256 << 20  # tables from this size on are moved to a temporary file once only rows of them are read


def _spill(a: np.ndarray, directory) -> np.ndarray:
    """a as a read-only memory map of an unlinked temporary file in `directory` (the same values; its memory is freed
    when the caller drops its last reference to the array), or a itself when it is small or the file cannot be written."""
    if a.nbytes < SPILL_MIN_BYTES:
        return a
    import tempfile
    try:
        fh = tempfile.TemporaryFile(dir=directory)
        a.tofile(fh)
        fh.flush()
        return np.memmap(fh, dtype=a.dtype, mode="r", shape=a.shape)
    except OSError:
        return a



def _bin_prior_scope(f):
    """Runs f with the form of the bin prior that its calibration features ask for (tempered-bin-prior) and puts
    the previous form back afterwards, also when the run fails."""
    import functools
    import inspect
    sig = inspect.signature(f)

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        try:
            on = "tempered-bin-prior" in parse_features(bound.arguments["calibration"])
        except ValueError:
            on = False  # run() reports the bad value
        old = set_tempered_bin_prior(on)
        try:
            return f(*args, **kwargs)
        finally:
            set_tempered_bin_prior(old)
    return wrapped


@_bin_prior_scope
def run(assembly, pileup, contig_bin, motifs_path, out, min_cov=1, chunk_size=50_000, grid_size=64,
        p_unknown=None, contamination_threshold=0.5, include_threshold=0.9, min_methylome_difference=0.4,
        max_redundancy=0.3, min_evidence=3.0, min_contig_coverage=5.0, min_contig_length=1000, motif_noise_filter=True, group_weights=True, include_threshold_unmethylated=0.95, regional_model=True, regional_inclusion=False, do_calibrate=True, frag_lengths=DEFAULT_LENGTHS, frags_per_bin=30, frag_budget=6000,
        tau=None, temperature=None, contam_eps=0.05, allow_sharpening=False, min_jaccard=0.9,
        variation="beta", group_sample_bp=50e6, workers=None, cache_dir=None, write_motif_fit=False, seed=1,
        shared_sites="sample", complement_pairs="sequence", strand_sites="sum", noise_max_off_share=1.0,
        robust_fit="alternating", max_flagged_share=1.0, regional_leave_out="chunk", regional_withdraw="vs_foreign",
        regional_tempered=False, gate_leave_out="none", gate_unweighted=False, eps_gated=False,
        redundancy_order="independent", unlisted_genome=False, inclusion_support="none", two_state_warning=0.2, tau_support="all",
        unknown_prior="bins", unknown_prior_strength=5.0,
        calibration="default", siteless_bin_guard=False,
        tau_select="per-group", npmle_start="warm", calibration_exclusion=True, unknown_share="deployed",
        temperature_law="grid", evidence_temperature="median", relatives="switch"):
    """calibration: 'default' (the calibration of 0.10 with the parsimony test of the unknown model resampled over
    bins, since 0.11.1), 'legacy' (the calibration of 0.10 and 0.11.0), 'consistent', 'coherent', or a comma-separated
    list of calibrate.FEATURES; see calibrate.parse_features.

    Calibration core options (WP3, experimental; each default is the behaviour of 0.11.1):
    tau_select: 'per-group' (the tau scan and a tau per motif group) or 'inf' (no between-contig variation: the
    calibration fits tau = inf only; the seed's main route into the scores, dev log model_diagnostics.md 6a).
    npmle_start: 'warm' or 'cold' (every population-prior fit from the uniform; model.FittedModel).
    unknown_share: 'deployed' (the EM at the deployed temperature) or 't1' (the EM at T = 1, the decisions with that
    share and the untempered bin prior). temperature_law: 'grid' (one temperature) or 'sites' (log T linear in
    log(sites), refitted under the chosen unknown model). evidence_temperature: 'median' or 'contig' (each contig's own
    temperature in the methylation-evidence cut-off). relatives: 'switch' (one unknown model for all bins, chosen by
    the 2-SE rule) or 'per-bin' (one mixture with a relatives weight per bin, fitted on the pieces; writes
    calibration/relatives_by_bin.tsv and hidden_relative.tsv, report only). See calibrate.calibrate_decisions.
    calibration_exclusion: False keeps the calibration fragments from chunks called foreign (diagnostic only)."""
    args = dict(locals())  # first statement: the options as given
    runinfo.begin(args, _log)  # first log lines (version, command, inputs) and option ranges, before any work
    # the inputs are checked before anything is read or written: a missing file or a wrong table must not
    # cost the minutes (hours on a large sample) it takes to reach the place where it is used
    runinfo.check_inputs(args)
    provenance = runinfo.provenance(args)
    features = parse_features(calibration)
    core = CoreOptions(tau_select=tau_select, npmle_start=npmle_start, unknown_share=unknown_share,
                       temperature_law=temperature_law, evidence_temperature=evidence_temperature,
                       relatives=relatives)
    if not core.decisions_default and features & {"temper-then-mix", "sites-temperature", "prior-matched",
                                                   "prior-averaged", "unit-temperature", "deployed-unknown"}:
        raise ValueError("--unknown-share, --temperature-law and --evidence-temperature cannot be combined with the "
                         "calibration features temper-then-mix, sites-temperature, prior-matched, prior-averaged, "
                         "unit-temperature or deployed-unknown")
    if not core.decisions_default and (unlisted_genome or regional_inclusion or not do_calibrate):
        raise ValueError("--unknown-share, --temperature-law and --evidence-temperature need the calibration and cannot "
                         "be combined with --unlisted-genome or --regional-inclusion")
    if "deployed-unknown" in features and features & {"prior-matched", "sites-temperature"}:
        raise ValueError("the calibration feature deployed-unknown checks one fit with one temperature; it cannot be "
                         "combined with prior-matched (which refits the unknown model) or sites-temperature")
    if "deployed-unknown" in features and unlisted_genome:
        raise ValueError("the calibration feature deployed-unknown cannot be combined with --unlisted-genome")
    out = Path(out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    runinfo.write_params(out / "params.json", provenance)  # a run that does not finish leaves its settings
    rng = np.random.default_rng(seed)
    cache = Path(cache_dir) if cache_dir else out / "cache"

    # the assembly is streamed (lengths, then the motif-grouping sample, then the encoded copy on disk): held
    # whole as Python strings it took about three times its size, 53 GB for an 18.9 GB soil assembly
    motifs = pio.read_motifs(motifs_path, log=_log)
    cb_given = pio.read_contig_bin(contig_bin, log=_log)
    pileup_first = runinfo.check_pileup_head(pileup)
    asm_cached = _cached_assembly(cache, assembly)
    if asm_cached is not None:
        _prefetch_cache(cache)
    lengths = (dict(zip(asm_cached.names, asm_cached.lengths.tolist())) if asm_cached is not None
               else pio.fasta_lengths(assembly))
    cb = runinfo.match_contig_bin(cb_given, lengths, contig_bin, log=_log)
    if pileup_first not in lengths:
        _log(f"WARNING: the first contig of the pileup, {pileup_first!r}, is not in the assembly: was the pileup "
             f"made from reads mapped to this assembly?")
    bins = sorted(set(cb.values()))
    bin_idx = {b: i for i, b in enumerate(bins)}
    host_contigs = sorted(cb)
    host_set = set(host_contigs)
    # every contig is scored: binned ones leave-one-out (contamination), unbinned ones against all bins
    # (inclusion)
    targets = sorted(lengths)
    H = len(bins)
    _log(f"{len(lengths)} contigs, {H} bins, {len(host_contigs)} host contigs, {len(targets)} targets")
    if H == 1:
        _log("WARNING: the contig-bin table names one bin only: contamination detection is skipped (a contaminant is "
             "recognised by fitting its own bin worse than other bins or a genome without a bin, which needs at "
             "least two bins); binned contigs are kept untested, unbinned contigs are still tested for inclusion")

    # motif groups, from the records of a random sample of the sequence
    relevant = sorted(host_set | set(targets))
    sample_names = _sample_names(lengths, relevant, group_sample_bp, rng)
    sample = _decoded(asm_cached, sample_names) if asm_cached is not None else pio.fasta_subset(assembly, sample_names)
    del asm_cached
    if strand_sites not in ("sum", "split", "auto"):
        raise ValueError(f"strand_sites must be 'sum', 'split' or 'auto', not {strand_sites!r}")
    records = build_records(sample, motifs, pairing=complement_pairs)
    group_of = group_motifs(records, motifs, min_jaccard, pairing=complement_pairs)
    rank_d = dict(records.group_by("motif").len().iter_rows())
    motif_rank = [int(rank_d.get(i, 0)) for i in range(len(motifs))]
    del records
    if shared_sites not in ("sample", "motif"):
        raise ValueError(f"shared_sites must be 'sample' or 'motif', not {shared_sites!r}")
    # a cache written with another numbering of the same groups is still valid: keep its labels
    man_path = cache / "manifest.json"
    cached = json.loads(man_path.read_text()) if man_path.exists() else None
    if cached and _same_partition(cached["inputs"].get("group_of", []), group_of):
        group_of = cached["inputs"]["group_of"]
    n_groups = max(group_of) + 1
    # which group a position matched by several groups goes to: "sample" orders the groups by their record counts
    # in the sample (seed-dependent for motifs of similar frequency); "motif" by the motifs' expected frequency
    group_rank = None
    if shared_sites == "motif":
        motif_rank, group_rank = specificity_ranks(motifs, group_of)
    group_modtype = [next(motifs[i].mod_type for i in range(len(motifs)) if group_of[i] == g) for g in range(n_groups)]
    group_label = [",".join(motifs[i].key for i in range(len(motifs)) if group_of[i] == g) for g in range(n_groups)]
    _log(f"{len(motifs)} motifs -> {n_groups} groups (overlap from {sum(map(len, sample.values())) / 1e6:.0f} Mb)")

    # pass 1: pileup -> site cache (reused when inputs are unchanged)
    man = _manifest(assembly=assembly, pileup=pileup, motifs=[m.key for m in motifs], group_of=group_of,
                    motif_rank=motif_rank, relevant=len(relevant), shared_sites="most_specific_group")
    if group_rank is not None:  # (keys are added only for non-default rules, so existing caches stay valid)
        man.update(shared_sites="motif_frequency", group_rank=group_rank)
    if not same_pairs(motifs, complement_pairs):  # a rule that pairs this motif list as "sequence" does gives the same sites
        man.update(complement_pairs=complement_pairs)
    if strand_sites != "sum":
        man.update(strand_sites=strand_sites + ("+half" if strand_sites == "auto" else ""))
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
                              workers=workers, log=_log, group_rank=group_rank, pairing=complement_pairs,
                              strand_sites=strand_sites)
        man_path.write_text(json.dumps({"inputs": man, "parts": parts}))
    del sample
    cid = asm.index

    if not parts:
        raise runinfo.no_sites_error(pileup, pileup_first, lengths)
    # contig coverage (median reads per motif site, all sites): contigs below min_contig_coverage get
    # no decision at all (status low_coverage) and don't feed the calibration fragments. Against Hi-C
    # on ZymoFecal, flags on contigs below 5x were confirmed for 12.5% (81% at >= 10x).
    # With --strand-sites auto some occurrences are stored as two single-strand sites (column half); their reads
    # are doubled here, so that the coverage of a contig does not depend on how many of its sites were cut.
    n_site = pl.col("n") * (1 + pl.col("half")) if "half" in pl.scan_parquet(parts).collect_schema().names() else pl.col("n")
    cov_of = dict(pl.scan_parquet(parts).group_by("contig").agg(n_site.median()).collect().iter_rows())
    # motif groups whose methylation is partial on many contigs of many genomes are caller noise:
    # they are dropped from every pass below (caller fit, likelihood tables, evidence)
    drop = []
    if motif_noise_filter:
        rich = [cid[c] for c in host_contigs if cov_of.get(cid[c], 0.0) >= 10]
        cb_id = pl.DataFrame({"contig": [cid[c] for c in host_contigs], "bin": [bin_idx[cb[c]] for c in host_contigs]},
                             schema={"contig": pl.Int32, "bin": pl.Int32})
        noise = noisy_groups(parts, cb_id, rich, max_off_bin_share=noise_max_off_share)
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
                               workers=workers or min(16, _cores()))
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
        runinfo.check_calibration_contigs([lengths[c] for c in hc_ok], len(host_contigs), min_contig_coverage, frag_lengths)
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

    # prior used by the calibration before p_unknown is estimated: 0.1 in the legacy calibration; with
    # prior-matched a neutral 0.5 (the fit is redone at the estimate); prior-averaged does not use it
    p_cal = (0.5 if "prior-matched" in features else 0.1) if p_unknown is None else p_unknown
    # (provenance holds --calibration as given; calibration_features the features it stands for)
    params = {**provenance, "chunk_size": chunk_size, "grid_size": grid_size, "p_unknown": p_unknown, "min_cov": min_cov,
              "contam_eps": contam_eps, "calibration_features": sorted(features) or ["legacy"]}
    ffit = fit = None
    user_temperature = temperature
    model_kw = {"contam_eps": contam_eps, "robust_fit": robust_fit}
    if npmle_start != "warm":
        model_kw["prior_start"] = npmle_start
    cal_taus = DEFAULT_TAUS if variation == "beta" else DEFAULT_SIGMAS
    if tau_select == "inf":  # no between-contig variation (beta: tau = inf; logit-normal: sigma = 0)
        cal_taus = (float("inf"),) if variation == "beta" else (0.0,)
    if do_calibrate:
        _log("calibration with pseudo-contigs")
        chunks_of = fragment_chunks(frags, chunk_size, chunk_index)
        cal = calibrate(L_chunks, chunk_bin, H, group_modtype, grid, frags, tabs["L_frag"], chunks_of,
                        p_cal, H_frag=tabs["H_frag"], allow_sharpening=allow_sharpening,
                        leave_out=contig_chunks(frags["contig"].to_numpy(), chunk_contig),
                        taus=cal_taus, n_sites_frag=tabs["frag_sites"], exclude_foreign=calibration_exclusion,
                        log=_log, features=features, tau_support=tau_support, **model_kw)
        ffit, fit = cal["fragment_fit"], cal["fit"]
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
        if temperature is not None:  # a temperature given by the user overrides the fit, and any curve
            fit = dict(fit, temperature=float(temperature), curve=None)
        temperature = cal["temperature"] if temperature is None else temperature
        params["fitted_temperature"] = cal["fitted_temperature"]
        params["n_fragments"] = frags.height
        warm = {"init_w": cal["model"].w, "init_pi": cal["model"].raw_pi}
        unknown_params = (cal["eta"], cal["lam"])
        params.update(unknown_eta=cal["eta"], unknown_lam=cal["lam"])
        community_fit = cal["community"]
        del cal
        _log(f"calibrated: tau={'per-group' if np.ndim(tau) else f'{tau:g}'}, temperature={temperature:g}")
    else:
        warm, unknown_params = {}, (None, 1.0)
        community_fit = None
    tabs = None
    tau = (100.0 if variation == "beta" else 0.3) if tau is None else tau
    temperature = 1.0 if temperature is None else temperature

    model = FittedModel(L_chunks, chunk_bin, H, group_modtype, grid, tau, **warm, **model_kw)
    if core.relatives == "per-bin":   # the contigs' relative scores at the eta of the per-bin mixture
        unknown_params = (ETA_REL, 1.0)
    model.set_unknown_params(*unknown_params)
    # from here on the chunk tables are read only for the few mixed contigs of the regional step: kept in a temporary
    # file (memory-mapped, removed when the run ends) instead of memory, next to the model's own marginal copy
    L_chunks = _spill(L_chunks, out)
    p_foreign = 1 - model.w
    _write(pl.DataFrame({"contig": [asm.names[i] for i in chunk_contig], "start": chunk_no * chunk_size,
                         "bin": [bins[b] for b in chunk_bin], "p_foreign": p_foreign}),
           out / "host_chunk_outliers.tsv")
    _log(f"robust host profiles: {int((p_foreign > 0.5).sum())} of {len(p_foreign)} chunks flagged as foreign to their bin")
    # a bin whose chunk weights still moved in the last round has no settled profile (two methylomes in
    # comparable shares under the alternating scheme): its chunk calls and its contigs' scores are not reliable
    unconverged = model.robust_unconverged()
    two_share, two_group = model.two_state_share()
    _log(f"two-state screen: {int((two_group >= 0).sum())} of {H} bins with a second methylation state in a motif group")
    if unconverged.any():
        _log(f"WARNING: robust host profile did not converge for {int(unconverged.sum())} of {H} bins "
             f"({', '.join(bins[b] for b in np.flatnonzero(unconverged)[:10])}{', ...' if unconverged.sum() > 10 else ''}): "
             f"the bin may hold two genomes in comparable shares (bin_evidence.tsv, column robust_fit)")

    params.update(tau=tau, temperature=temperature)

    # score every contig against every bin; binned contigs leave-one-out, so their own bin's profile
    # is built without them
    # (their Lp rows are the only ones the leave-one-out scores read: the rest is not kept, unless an option
    # reads the tables of every contig afterwards: prior-matched rebuilds the relatives' scores from Lp and the
    # leave-out terms when it refits the unknown model; the unbinned prior, the unlisted genome and the
    # inclusion support read the unbinned contigs' rows)
    refit_unknown = "prior-matched" in features
    all_lp_rows = refit_unknown or unknown_prior == "unbinned" or unlisted_genome or inclusion_support != "none"
    sc = model.score(L_t, lp_rows=None if all_lp_rows else [i for i, c in enumerate(targets) if c in host_set])
    _log(f"scores: {len(targets)} contigs against {H} bins and an unbinned genome")
    if unknown_prior == "unbinned":
        # the unbinned-genome model draws each motif group's state at the frequency seen among the unbinned
        # contigs that get a decision, not among the bins (model.unbinned_state_frequencies)
        ub = np.array([c not in host_set and cov_of.get(cid[c], 0.0) >= min_contig_coverage
                       and lengths[c] >= min_contig_length for c in targets])
        off, on, q_bins = model.state_terms(sc)
        c_unb = unbinned_state_frequencies(off[ub], on[ub], q_bins, unknown_prior_strength)
        sc.Su_comm = np.logaddexp(np.log1p(-c_unb)[None] + off, np.log(c_unb)[None] + on).sum(1)
        sc.Su = unknown_score(sc.R, sc.Su_comm, model.eta, model.lam)
        _write(pl.DataFrame({"group": np.arange(n_groups), "motifs": group_label, "methylated_among_bins": q_bins,
                             "methylated_among_unbinned": c_unb,
                             "unbinned_contigs_with_sites": (np.abs(on[ub] - off[ub]) > 1e-9).sum(0)}),
               out / "unbinned_prior.tsv")
        _log(f"unbinned-genome prior from {int(ub.sum())} unbinned contigs: methylated share per group "
             f"median {np.median(c_unb):.3f} (among bins {np.median(q_bins):.3f})")
        del off, on
    elif unknown_prior != "bins":
        raise ValueError(f"unknown unknown_prior {unknown_prior!r}")
    # each contig's methylated fraction per group with its 95% interval (for the effect-size gate below). Group
    # weights were multiplied into the tables; gate_unweighted takes them out again, so that a group sharing its
    # weight with variants of the same methyltransferase keeps the interval its own sites give it
    gw = (group_w.astype(np.float64)[None, :, None] if gate_unweighted and group_weights and (group_w < 1).any() else None)
    phi_obs = _batched_summary(grid, lambda sl: L_t[sl].astype(np.float64) / (1.0 if gw is None else gw), len(targets))
    table_cells = L_t.shape[1] * L_t.shape[2]
    # the contigs' tables are not read after this point (the leave-one-out scores use the kept marginal rows):
    # freed before the evidence step, where the main process had its peak (6.3 GB on ZymoFecal)
    del L_t
    diff = np.zeros(len(targets))
    llr_loo = np.zeros(len(chunk_bin))
    sole_contig = set()
    loo: dict[int, tuple] = {}  # contig -> (own bin, per-group leave-out terms); filled for refit_unknown only
    bin_chunks = np.bincount(chunk_bin, minlength=H)
    own_chunks: dict[int, list[int]] = {}
    for u, c in enumerate(chunk_contig):
        own_chunks.setdefault(int(c), []).append(u)

    # leave-one-out scores of the binned contigs, LOO_BATCH contigs at a time in whole-array steps (each contig
    # writes its own rows and chunks only: the batches run on threads)
    jobs = []
    for i, c in enumerate(targets):
        if c not in host_set:
            continue
        own = np.array(own_chunks.get(cid[c], []), dtype=np.int64)
        b = bin_idx[cb[c]]
        if len(own) and len(own) == bin_chunks[b]:
            sole_contig.add(c)  # leave-one-out would empty the bin: not an independent test
            continue
        jobs.append((i, b, own))

    def leave_out(batch):
        rows = np.array([i for i, _, _ in batch], np.int64)
        bs = np.array([b for _, b, _ in batch], np.int64)
        profs = model.loo_profiles(bs, [own for _, _, own in batch])
        A_loo = model.loo_rescore_many(sc, rows, bs, profs)
        if refit_unknown:
            for k, i in enumerate(rows):
                loo[int(i)] = (int(bs[k]), A_loo[k])
        if gate_leave_out == "contig":
            # distance between the two 95% intervals: the contig's and that of its bin without the contig
            _, lo_b, hi_b, _ = grid.summarize(profs)
            gap = np.maximum(np.maximum(phi_obs[1][rows] - hi_b, lo_b - phi_obs[2][rows]), 0.0)
            diff[rows] = np.where(n_sites_group[rows] > 0, gap, 0.0).max(1)
        if regional_leave_out == "contig":
            for k, (_, _, own) in enumerate(batch):
                if len(own) >= 2:
                    llr_loo[own] = model.chunk_llr(own, profs[k:k + 1])
    batches = [jobs[s:s + LOO_BATCH] for s in range(0, len(jobs), LOO_BATCH)]
    binned = [i for i, c in enumerate(targets) if c in host_set]
    if table_cells < LOO_THREADS_MIN:  # small tables: Python-bound, threads only contend
        for batch in batches:
            leave_out(batch)
    else:
        with ThreadPoolExecutor(_n_threads()) as ex:
            list(ex.map(leave_out, batches))
    loo = dict(sorted(loo.items()))  # in contig order, as the loop filled it
    _log(f"leave-one-out scores: {len(binned) - len(sole_contig)} binned contigs ({len(sole_contig)} sole contigs of their bin)")

    # bin refinement: contamination of binned contigs, inclusion of unbinned ones
    given = np.array([bin_idx[cb[c]] if c in cb else -1 for c in targets])
    if ("deployed-unknown" in features and ffit is not None and fit is not None and fit["lam"] < 1
            and community_fit is not None and user_temperature is None and p_unknown is None):
        # The relatives model was chosen on the fragments at the reference prior; check it at the prior it would be
        # deployed with. p_unknown is estimated exactly as refine() will (EM over the unbinned contigs that get a
        # decision), once under each model, and both models are scored on the fragments at their own estimate.
        ub = np.array([c not in host_set and cov_of.get(cid[c], 0.0) >= min_contig_coverage
                       and lengths[c] >= min_contig_length for c in targets])
        prior_u = (2.0, 2.0) if "em-priors" in features else (1.0, 1.0)
        ttm = "temper-then-mix" in features

        def deployed_p(f):
            T = f["temperature"]
            zu = unknown_logit(sc.R[ub] if f["lam"] < 1 else None, sc.Su_comm[ub], f["eta"], f["lam"], T, temper_then_mix=ttm)
            return em_weight(logsumexp(sc.S[ub] / T, axis=1) - log_bins(H, T), zu, *prior_u)[0]
        p_rel, p_com = deployed_p(fit), deployed_p(community_fit)
        gain, se = ffit.deployed_gain((fit["temperature"], fit["eta"], fit["lam"]), p_rel,
                                      (community_fit["temperature"], community_fit["eta"], 1.0), p_com)
        keep = gain > 2 * se
        _log(f"unknown model at its deployed prior: relatives (T={fit['temperature']:g}, eta={fit['eta']:g}, lam={fit['lam']:g}) "
             f"p_unknown={p_rel:.3f} vs community only (T={community_fit['temperature']:g}) p_unknown={p_com:.3f}: "
             f"log-loss gain {gain:.4f} +- {se:.4f} -> {'relatives' if keep else 'community only'}")
        params["unknown_model_deployed_check"] = {"p_unknown_relatives": p_rel, "p_unknown_community": p_com,
                                                  "gain": gain, "se": se, "kept_relatives": bool(keep)}
        if not keep:
            fit = dict(fit, temperature=community_fit["temperature"], eta=community_fit["eta"], lam=1.0, curve=None)
            temperature = fit["temperature"]
            model.set_unknown_params(fit["eta"], 1.0)
            sc.Su = sc.Su_comm.copy()
            params.update(temperature=temperature, unknown_eta=fit["eta"], unknown_lam=1.0)
    # expected methylation evidence (assignability) per contig, for its own bin (binned) or its best
    # bin (unbinned), and its coverage (median reads per motif site)
    best_bin = np.where(given >= 0, given, sc.S.argmax(1))

    def methylation_evidence(T_used, per_contig=False):
        # the expected log BF is linear in 1 / T: per_contig computes it at T = 1 and divides by each contig's T
        ev, _, cov_ = contig_evidence(parts, {cid[c]: i for i, c in enumerate(targets)}, best_bin, model, det,
                                      grid, 1.0 if per_contig else float(np.median(np.atleast_1d(T_used))), min_cov,
                                      group_weights=group_w if group_weights else None, drop_groups=drop, log=_log)
        if per_contig:
            ev = ev / np.asarray(T_used, float)
        _log(f"methylation evidence: {int((np.nan_to_num(ev, nan=0.0) < min_evidence).sum())} of {len(targets)} "
             f"contigs below {min_evidence:g} (expected log BF vs a community genome)")
        return ev, cov_
    # the core options that change the temperature fix it in the decision step below; the evidence follows it there
    core_late = do_calibrate and not core.decisions_default
    if not core_late:
        evidence, coverage = methylation_evidence(temperature)
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
    # gate_leave_out "none" (up to 0.10.x): the bin's level from the profile that still holds the contig, as a
    # point. "contig": the bin's 95% interval without the contig (computed in the loop above), as for the score:
    # a contig that is a large share of its bin otherwise pulls the bin's level toward itself and clears itself.
    phi_bin = grid.summarize(model.log_prof)[0]
    has = given >= 0
    if gate_leave_out != "contig":
        pb = phi_bin[given[has]]
        gap = np.maximum(np.maximum(phi_obs[1][has] - pb, pb - phi_obs[2][has]), 0.0)
        diff[has] = np.where(n_sites_group[has] > 0, gap, 0.0).max(1)
    bin_ev = bin_evidence_table(bins, grid.summarize(model.log_prof), n_sites_group, given)
    unmeth_bins = (bin_ev["status"] == "no_methylation_detected").to_numpy()
    new_path = bool(features & {"temper-then-mix", "sites-temperature", "prior-matched", "prior-averaged", "unit-temperature"})
    em_kw = dict(p_unknown_prior=(2.0, 2.0), eps_after_gate=True) if "em-priors" in features else {}
    if fit is None:  # no calibration: one temperature, the unknown model as set
        fit = {"temperature": float(temperature), "eta": model.eta, "lam": model.lam, "curve": None}
    # options of the decision rules that both calibration paths pass on
    rule_kw = dict(eps_gated=eps_gated, empty_bins=bin_chunks == 0 if siteless_bin_guard else None,
                   S_unlisted=model.unlisted_score(sc) if unlisted_genome else None)
    if unlisted_genome and new_path:
        raise ValueError("--unlisted-genome cannot be combined with the calibration features temper-then-mix, "
                         "sites-temperature, prior-matched, prior-averaged or unit-temperature")
    T_contig, Zu = temperature, None
    relatives_weights = None

    def decide(fit):
        """Decisions under one calibration fit: the temperature (per contig when it depends on the number of
        sites) and the unknown hypothesis' tempered score under the fit's unknown model."""
        T = fit["curve"](n_sites) if fit.get("curve") is not None else fit["temperature"]
        R = None
        if fit["lam"] < 1 and fit["eta"]:
            R = sc.R if fit["eta"] == (model.eta if model.eta else 0.05) else model.relatives(sc, fit["eta"], loo)
        zu = unknown_logit(R, sc.Su_comm, fit["eta"], fit["lam"], T, temper_then_mix="temper-then-mix" in features)
        r = refine(sc.S, None, given, testable, T, contamination_threshold, include_threshold,
                   methylome_difference=diff, min_difference=min_methylome_difference,
                   p_unknown=p_unknown, decidable=~low_cov & ~short, unmethylated_bins=unmeth_bins,
                   include_threshold_unmethylated=include_threshold_unmethylated, Zu=zu, **em_kw, **rule_kw)
        return r, T, zu

    if not new_path:
        # the one calibration function (calibrate.calibrate_decisions): at the default core options it is the
        # refine() call of 0.11 on the calibrated temperature and unknown model
        contigs = ContigScores(S=sc.S, Su_comm=sc.Su_comm, R={model.eta: sc.R} if model.eta else {}, given=given,
                               testable=testable, decidable=~low_cov & ~short, n_sites=np.asarray(n_sites, float),
                               Su=sc.Su, Su_eta_lam=(model.eta, model.lam))
        rules = dict(contamination_threshold=contamination_threshold, include_threshold=include_threshold,
                     methylome_difference=diff, min_difference=min_methylome_difference, p_unknown=p_unknown,
                     unmethylated_bins=unmeth_bins, include_threshold_unmethylated=include_threshold_unmethylated,
                     **em_kw, **rule_kw)
        core_fit = dict(fit, user_temperature=user_temperature)
        if core.relatives == "switch" and ((core_fit["lam"] < 1 and model.eta and core_fit["eta"] != model.eta)
                                           or core_fit.get("curve") is not None):
            raise ValueError("the calibration core path expects the unknown model the contigs were scored with")
        res = calibrate_decisions(ffit, contigs, rules, core, fit=core_fit, log=_log)
        ref = res.refinement
        if not core.decisions_default:
            temperature = res.T_contig
            fit = {k: v for k, v in res.fit.items() if k not in ("user_temperature", "relatives_weights")}
            params.update(temperature=float(np.median(np.atleast_1d(temperature))),
                          temperature_by_sites=res.law.table() if res.law is not None else None,
                          p_unknown_t1=res.p_unknown_t1, unknown_eta=fit["eta"], unknown_lam=fit["lam"])
        if res.weights is not None:
            Zu = None   # the decisions' posterior below is rebuilt from the mixture
            params["relatives_per_bin"] = res.weights.summary()
            _write(pl.DataFrame({"bin": bins, "weight": res.weights.w, "H_weight": H * res.weights.w}),
                   out / "calibration" / "relatives_by_bin.tsv")
            ph = hidden_relative(contigs, res.weights, ref.p_unknown, res.T_contig, best_bin)
            _write(pl.DataFrame({"contig": targets, "bin": [bins[b] for b in best_bin], "p_hidden_relative": ph}),
                   out / "hidden_relative.tsv")
            relatives_weights, T_rel = res.weights, res.T_contig
        del contigs
    else:
        ref, T_contig, Zu = decide(fit)
        if "prior-matched" in features and ffit is not None and p_unknown is None:
            # the calibration prior must be the prior of the decisions: refit (T, eta, lam) at the estimated
            # p_unknown and re-estimate, until the fit no longer changes (it lives on a grid, so it does, or cycles)
            key = lambda f: (f["temperature"], f["eta"], f["lam"], None if f.get("curve") is None else tuple(f["curve"].log_T))
            seen = [key(fit)]
            for it in range(5):
                new = ffit.fit(ref.p_unknown, log=_log)
                if user_temperature is not None:
                    new = dict(new, temperature=float(user_temperature), curve=None)
                _log(f"calibration at the estimated p_unknown={ref.p_unknown:.3f}: temperature={new['temperature']:g}, "
                     f"eta={new['eta']:g}, lam={new['lam']:g}")
                if key(new) == key(fit):
                    break
                cycle = key(new) in seen
                fit = new
                seen.append(key(fit))
                ref, T_contig, Zu = decide(fit)
                if cycle:
                    _log("calibration and p_unknown do not settle (cycle); keeping the last fit")
                    break
        temperature = fit["temperature"]
        params.update(temperature=temperature, unknown_eta=fit["eta"], unknown_lam=fit["lam"],
                      temperature_by_sites=fit["curve"].table() if fit.get("curve") is not None else None)
    if core_late:
        evidence, coverage = methylation_evidence(temperature, per_contig=core.evidence_temperature == "contig")
    if unlisted_genome:
        sc.Su = ref.Su
        params["unlisted_share"] = ref.unlisted_share
        _log(f"unbinned genomes: {ref.unlisted_share:.3f} without any listed motif (unlisted), the rest community draws")
    if inclusion_support == "methylated_groups":
        # An inclusion must be supported by the motif groups its bin methylates: the posterior of the bin is
        # recomputed from those groups alone and must reach the inclusion threshold too. Absence of methylation
        # at the other listed motifs can speak against a bin (through the full posterior) but not for it, since
        # a genome without a bin lacks them by construction (the list comes from the bins).
        on = grid.summarize(model.log_prof)[1] >= 0.5
        Tm = float(np.median(np.atleast_1d(temperature)))
        pu = min(max(ref.p_unknown, 1e-6), 1 - 1e-6)
        cand = np.flatnonzero(ref.include_bin >= 0)
        support = np.zeros(len(targets))
        for h in np.unique(ref.include_bin[cand]):
            rows = cand[ref.include_bin[cand] == h]
            ms = np.flatnonzero(on[h])
            S_on = (summed_scores(sc.Lp[rows][:, ms], model.log_prof[:, ms]) if len(ms) else np.zeros((len(rows), H)))
            n_alt = np.where(given[rows] >= 0, H - 1, H)
            lg = np.concatenate([S_on / Tm + np.log((1 - pu) / n_alt)[:, None],
                                 (sc.Au[rows][:, ms].sum(1) / Tm + np.log(pu))[:, None]], 1)
            b_ = given[rows] >= 0
            lg[np.flatnonzero(b_), given[rows][b_]] = -np.inf
            support[rows] = np.exp(lg[:, h] - logsumexp(lg, axis=1))
        thr_s = np.where(unmeth_bins[np.maximum(ref.include_bin, 0)], include_threshold_unmethylated, include_threshold)
        weak = (ref.include_bin >= 0) & (support < thr_s)
        _log(f"inclusion support: {int(weak.sum())} of {len(cand)} inclusions dropped (the bin's methylated motif groups "
             f"alone do not support them)")
        ref.include_bin[weak] = -1

    # long binned contigs: whole contig foreign, or partly the bin's and partly foreign along its length?
    # (chunk-level evidence from the robust host-profile fit; flags on mixed contigs are withdrawn and
    # their foreign segments reported)
    regional = None
    if regional_model and contam_eps > 0:  # at 0 the prior log-odds of a foreign chunk are infinite
        if regional_leave_out == "contig":
            # chunk evidence against the same profile as the contig's own score: the bin without the whole
            # contig. With only the chunk left out, the other chunks of a long foreign contig that makes up
            # much of a small bin vouch for it.
            llr_chunk = llr_loo / (float(np.median(np.atleast_1d(temperature))) if regional_tempered else 1.0)
        else:
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
            # a flag is withdrawn when "mixed" explains the contig best; "vs_foreign" (up to 0.10.x) compared it
            # with "foreign" only, so a contig the chunks call a plain member was labelled mixed as well
            if ref.contaminant[i] and withdraw_as_mixed(post[j], regional_withdraw)[0]:
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
                logits = np.concatenate([Sw[ks] + np.log1p(-pu) - log_bins(H, T), (Suw[ks] + np.log(pu))[:, None]], 1)
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

        def segment_map(L_rows, nos, i, sc_=None):
            sc_ = model.score(L_rows) if sc_ is None else sc_
            lg = np.concatenate([sc_.S / T0 + np.log1p(-pu0) - log_bins(H, T0), (sc_.Su / T0 + np.log(pu0))[:, None]], 1)
            pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            best, pbest = pr.argmax(1), pr.max(1)
            runs, start = [], 0
            for k in range(1, len(best) + 1):
                if k == len(best) or best[k] != best[start]:
                    lo, hi = int(nos[start]) * chunk_size, min((int(nos[k - 1]) + 1) * chunk_size, lengths[targets[i]])
                    runs.append(f"{lo}-{hi}:{lbl[best[start]]}({pbest[start:k].mean():.2f})")
                    start = k
            return ";".join(runs)
        # the chunks of all mixed contigs scored in one call (one call per contig cost about a second each on
        # ZymoFecal, mostly the profiles' exponentials and per-group products, for a few chunks)
        mixed_u = [u for u in long_u if mixed[tidx[u]]]
        if mixed_u:
            all_ks = np.concatenate([np.asarray(by_contig[u], np.int64) for u in mixed_u])
            sc_all = model.score(L_chunks[all_ks])
            at = 0
            for u in mixed_u:
                ks = by_contig[u]
                rows = slice(at, at + len(ks))
                at += len(ks)
                seg_bins[tidx[u]] = segment_map(None, chunk_no[ks], tidx[u],
                                                SimpleNamespace(S=sc_all.S[rows], Su=sc_all.Su[rows]))
        if len(win):
            for u, ks in by_win.items():
                i = tidx.get(u)
                if i is not None and given[i] < 0 and ref.include_bin[i] >= 0 and segments[i]:
                    seg_bins[i] = segment_map(L_win[ks], win_no[ks], i)
        regional = {"p": reg_p, "segments": segments, "mixed": mixed, "segment_bins": seg_bins}
        _log(f"regional: {len(long_u)} binned contigs with >= 2 chunks; {int(mixed.sum())} flags withdrawn as mixed "
             f"(partly the bin's, partly foreign along the contig)")
    # a bin in which the flagged contigs hold at least max_flagged_share of the sequence has no identifiable
    # majority genome: contamination is by definition the smaller part, so this is a merge of genomes (or a
    # profile that settled on the wrong one), and removing "the contaminants" would remove a genome. No flag
    # of such a bin is applied; the bin is reported (bin_evidence.tsv).
    L_arr = np.array([lengths[c] for c in targets], dtype=float)
    judged = testable & ~low_cov & ~short
    flagged_share = flagged_share_per_bin(given, ref.contaminant, judged, L_arr, H)
    # (only when asked for: at the default of 1 every flag is applied, as in 0.10. A share of exactly 1 is common
    # on a metagenome, where one or two contigs of a bin have the coverage to be judged and are flagged.)
    merged_bins = suspected_merges(flagged_share, max_flagged_share)
    if merged_bins.any():
        undo = ref.contaminant & merged_bins[np.maximum(given, 0)]
        ref.contaminant[undo] = False
        ref.include_bin[undo] = -1
        ref.candidate[undo] = False
        _log(f"WARNING: {int(merged_bins.sum())} bins look like a merge of genomes (flagged contigs hold >= "
             f"{max_flagged_share:g} of the bin): {int(undo.sum())} flags not applied "
             f"({', '.join(bins[b] for b in np.flatnonzero(merged_bins)[:10])})")
    bin_ev = bin_ev.with_columns(
        pl.Series("robust_fit", np.where(unconverged, "not_converged", "converged")),
        pl.Series("flagged_share", flagged_share),
        pl.Series("two_state_share", two_share),
        pl.Series("two_state_group", [group_label[g] if g >= 0 else "" for g in two_group]),
        pl.Series("warning", [";".join(w for w, on in (("merged_bin_suspected_flags_not_applied", merged_bins[b]),
                                                       ("two_methylation_states", two_share[b] >= two_state_warning),
                                                       ("large_flagged_share", flagged_share[b] >= two_state_warning and not merged_bins[b]),
                                                       ("profile_not_converged", unconverged[b])) if on)
                              for b in range(H)]))
    if (two_share >= two_state_warning).any():
        tb = np.flatnonzero(two_share >= two_state_warning)
        _log(f"WARNING: {len(tb)} bins show two methylation states in a motif group among their chunks (the smaller state "
             f"holds >= {two_state_warning:g} of the evidence; bin_evidence.tsv, two_state_share): two genomes or strains in the bin, or part of one genome in another state. "
             + ", ".join(f"{bins[b]} ({two_share[b]:.2f})" for b in tb[:10]))
    # don't include contigs whose sequence the target bin already holds (strain variants, haplotigs):
    # methylation can't see redundancy (D21)
    redundancy = np.full(len(targets), np.nan)
    if max_redundancy < 1:
        inc = np.flatnonzero(ref.include_bin >= 0)
        members: dict[int, list[int]] = {}
        for i, c in enumerate(targets):
            if given[i] >= 0 and not ref.contaminant[i]:
                members.setdefault(int(given[i]), []).append(cid[c])
        # redundancy_order "sequential": candidates of a bin in order of their probability, each accepted one
        # joining the bin before the next is compared (two unbinned copies of one sequence are not both included)
        redundancy[inc] = inclusion_redundancy(asm, [(cid[targets[i]], int(ref.include_bin[i])) for i in inc], members,
                                               priority=ref.p_best_alt[inc] if redundancy_order == "sequential" else None,
                                               max_redundancy=max_redundancy)
        drop = inc[redundancy[inc] >= max_redundancy]
        ref.include_bin[drop] = -1
        _log(f"redundancy: {len(drop)} of {len(inc)} inclusions dropped (>= {max_redundancy:g} of their k-mers already in the bin)")
    p_unknown = ref.p_unknown
    params.update(p_unknown=p_unknown, contamination_rate=ref.eps,
                  contamination_detection="skipped: one bin only" if H == 1 else "done")
    if relatives_weights is not None:
        P = posterior(sc.S, None, p_unknown, T_rel, Zu=relatives_weights.logit(sc.R, sc.Su_comm, T_rel))
    else:
        P = posterior(sc.S, sc.Su, p_unknown, temperature) if Zu is None else posterior(sc.S, None, p_unknown, T_contig, Zu=Zu)
    if ffit is not None:
        # how the deployed posteriors behave on the calibration fragments (true members scored without their
        # contig): false-flag rate per length, and reliability at the priors and temperature actually used
        dfit = dict(fit, eta=fit["eta"] if fit["eta"] else 0.05)
        if not new_path and not isinstance(fit.get("curve"), SitesLaw):
            dfit = dict(dfit, temperature=float(np.median(np.atleast_1d(temperature))), curve=None)
        _write(pl.DataFrame(ffit.false_flags(dfit, p_unknown, ref.eps, contamination_threshold, min_methylome_difference)),
               out / "calibration" / "false_flag_rate.tsv")
        _write(pl.DataFrame(ffit.deployed_reliability(dfit, p_unknown, ref.eps)), out / "calibration" / "reliability_deployed.tsv")
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
    runinfo.write_params(out / "params.json", runinfo.finish(params))
    _log(f"done -> {out}; peak memory {runinfo.peak_rss_gb():.2f} GB (main process), "
         f"{runinfo.peak_rss_gb(children=True):.2f} GB (largest worker process)")
    return out
