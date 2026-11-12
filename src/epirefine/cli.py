from __future__ import annotations

import argparse
import sys
import traceback

from .calibrate import DEFAULT_LENGTHS


class _Help(argparse.HelpFormatter):
    """Shows the default of every option that takes a value, unless its help text already states one
    (options that are estimated or calibrated when not given say so in words)."""

    def _get_help_string(self, action):
        text = action.help or ""
        if "default" in text or action.default in (None, argparse.SUPPRESS) or action.nargs == 0:
            return text
        d = action.default
        return f"{text} (default: {','.join(map(str, d)) if isinstance(d, (tuple, list)) else d})"


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    ap = argparse.ArgumentParser(prog="epirefine", description="Bin refinement from nanopore methylation: flags "
                                 "contaminating contigs in bins and places unbinned contigs into bins. "
                                 "Run 'epirefine run --help' for the inputs and options.")
    ap.add_argument("--version", action="version", version=f"epirefine {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="refine bins: contamination detection and contig inclusion", formatter_class=_Help,
                       description="Refine bins: flag binned contigs whose methylation contradicts their bin and "
                                   "place unbinned contigs into the bin their methylation fits. Progress is "
                                   "written to standard output.")
    g = r.add_argument_group("inputs and output")
    g.add_argument("--assembly", required=True,
                   help="assembly FASTA (.fasta or .fasta.gz); contig names are the first word of each header and "
                        "must be those used in the pileup and the contig-bin table")
    g.add_argument("--pileup", required=True, help="modkit pileup bedMethyl (.bed or .bed.gz) of reads mapped to the "
                                                   "assembly, rows of each contig together, no header line")
    g.add_argument("--contig-bin", required=True, help="contig<TAB>bin, tab-separated, no header")
    g.add_argument("--motifs", required=True,
                   help="Nanomotif bin-motifs.tsv, or any TSV with columns motif, mod_type (a = 6mA, m = 5mC, "
                        "21839 = 4mC, h = 5hmC) and mod_position (0-based); only the motif set is used")
    g.add_argument("--out", required=True, help="output directory (created; existing files in it are overwritten)")

    g = r.add_argument_group("decisions")
    g.add_argument("--contamination-threshold", type=float, default=0.5,
                   help="flag a binned contig as contamination when P(foreign) is at least this")
    g.add_argument("--min-methylome-difference", type=float, default=0.4,
                   help="a contaminant's 95%% credible interval for the methylated fraction of some motif group "
                        "must lie at least this far from its bin's level (0 = no effect-size requirement)")
    g.add_argument("--include-threshold", type=float, default=0.9,
                   help="include a contig into a bin when that bin's posterior is at least this")
    g.add_argument("--include-threshold-unmethylated", type=float, default=0.95,
                   help="include a contig into a bin with no detected methylation only at this posterior (such a bin matches "
                        "any contig whose own motifs are not in the motif list)")
    g.add_argument("--max-redundancy", type=float, default=0.3,
                   help="don't include a contig when at least this share of its k-mers is already in the bin "
                        "(1 = no redundancy check)")
    g.add_argument("--p-unknown", type=float, help="prior probability that an unbinned contig's genome has no bin "
                                                  "(default: estimated from the unbinned contigs)")

    g = r.add_argument_group("which contigs are judged")
    g.add_argument("--min-evidence", type=float, default=3.0,
                   help="contigs whose expected methylation evidence (log Bayes factor of their bin against a random "
                        "community genome) is below this get status no_methylation_evidence instead of kept/unbinned")
    g.add_argument("--min-contig-coverage", type=float, default=5.0,
                   help="contigs whose median reads per motif site is below this get no decision (neither "
                        "flagged nor included; status low_coverage) and give no calibration fragments; 0 decides all")
    g.add_argument("--min-contig-length", type=int, default=1000,
                   help="contigs shorter than this get no decision (status too_short); 0 decides all")
    g.add_argument("--min-cov", type=int, default=1,
                   help="use a motif site only when it has at least this many reads (per site; not to be confused "
                        "with --min-contig-coverage, the median over a contig's sites)")

    g = r.add_argument_group("motifs")
    g.add_argument("--no-group-weights", action="store_true",
                   help="count every motif group fully, even variants of one methyltransferase (default: groups that can "
                        "share sites and are methylated in the same bins share one group's weight; see group_weights.tsv)")
    g.add_argument("--no-motif-noise-filter", action="store_true",
                   help="keep motif groups whose methylation is partial on many contigs of many bins (by default "
                        "they are dropped as caller noise; see motif_noise.tsv)")
    g.add_argument("--min-jaccard", type=float, default=0.9,
                   help="merge motifs whose site sets overlap this much (Jaccard); positions shared by motifs "
                        "of different groups go to the most specific group")
    g.add_argument("--noise-max-off-share", type=float, default=1.0,
                   help="call a motif group noise only when at most this share of the bins that measure it leave it "
                        "unmethylated (a real motif is off in most genomes, a caller leak in none; 1 = not used)")
    g.add_argument("--shared-sites", choices=["sample", "motif"], default="sample",
                   help="which group gets a position that motifs of several groups match: the group with the fewest "
                        "occurrences in the sampled sequence (sample), or the one whose motif is least frequent in "
                        "random sequence (motif; independent of the sample and the seed)")
    g.add_argument("--complement-pairs", choices=["sequence", "position", "separate"], default="sequence",
                   help="a motif and its reverse complement in the motif list: one two-strand site per occurrence, paired "
                        "by sequence (sequence), paired only when each has a single modified position (position), or "
                        "kept as two motifs with single-strand sites (separate)")
    g.add_argument("--strand-sites", choices=["sum", "split", "auto"], default="sum",
                   help="reads of the two strands at one occurrence of a palindrome or complement pair: summed into one "
                        "site (sum), kept as one site per strand (split), or summed unless the strands can differ: "
                        "complement pairs, and occurrences where one strand is also a site of another motif group (auto)")
    g.add_argument("--group-sample-mb", type=float, default=50,
                   help="Mb of sequence sampled to measure motif overlap for grouping")

    g = r.add_argument_group("long contigs")
    g.add_argument("--no-regional-model", action="store_true",
                   help="judge long binned contigs only as a whole (default: a contig that is partly its bin's and partly "
                        "foreign along its length is reported as mixed, with its foreign segments, instead of flagged)")
    g.add_argument("--regional-inclusion", action="store_true",
                   help="also include long unbinned contigs scored in windows of --chunk-size (a bin explaining most "
                        "of the contig); experimental, no measured benefit; needs the regional model")
    g.add_argument("--chunk-size", type=int, default=50_000, help="binned contigs are split into chunks of this size (bp)")

    g = r.add_argument_group("model and calibration")
    g.add_argument("--variation", choices=["beta", "logit"], default="beta",
                   help="how a contig's methylated fraction varies around its bin's: Beta(tau) or "
                        "logit-normal(sigma); --tau then sets tau or sigma")
    g.add_argument("--tau", type=float, help="override the calibrated variation parameter: tau of --variation beta "
                                            "(larger = less variation) or sigma of --variation logit (default: calibrated)")
    g.add_argument("--temperature", type=float, help="override the calibrated temperature, the factor that divides all "
                                                    "log-likelihoods (default: calibrated)")
    g.add_argument("--allow-sharpening", action="store_true",
                   help="use a fitted temperature < 1 (default: clip at 1, never sharpen posteriors)")
    g.add_argument("--no-calibrate", action="store_true",
                   help="skip the calibration on pieces cut from the bins: probabilities are then uncalibrated "
                        "(temperature 1 and a fixed variation parameter unless --temperature / --tau are given)")
    g.add_argument("--calibration", default="default",
                   help="calibration design: 'default' (as in 0.10, plus bin-parsimony: an unknown model with "
                        "relatives must beat the community-only one by 2 SE taken over bins; since 0.11.1), "
                        "'legacy' (exactly as in 0.10 and 0.11.0), 'consistent' (temper-then-mix, common-tau, "
                        "em-priors), 'coherent' (the full proposed design), or a comma-separated list of features "
                        "(temper-then-mix, prior-averaged, prior-matched, conservative-temperature, full-test-b, "
                        "unit-temperature, sites-temperature, common-tau, em-priors, bin-parsimony, tempered-bin-prior, deployed-unknown; '-name' removes one). "
                        "See docs/development/method-notes.md")
    g.add_argument("--frag-lengths", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEFAULT_LENGTHS,
                   help="lengths (bp, comma-separated) of the pieces cut from binned contigs for calibration; at "
                        "least one binned contig must be as long as the shortest")
    g.add_argument("--frags-per-bin", type=int, default=30, help="cap on calibration fragments per bin and length")
    g.add_argument("--frag-budget", type=int, default=6000,
                   help="total calibration fragments; spread over bins and lengths (at most --frags-per-bin each)")
    g.add_argument("--contam-eps", type=float, default=0.05,
                   help="prior share of foreign chunks in a bin, used to keep contaminants out of the bin profiles "
                        "(0 = no such down-weighting and no regional model; must be below 1)")
    g.add_argument("--grid-size", type=int, default=64,
                   help="number of grid points for the methylated fraction of a motif group (finer = slower)")
    g.add_argument("--seed", type=int, default=1,
                   help="random seed (motif-grouping sample, calibration pieces); the same inputs and seed give "
                        "the same outputs")

    g = r.add_argument_group("speed and extra output")
    g.add_argument("--workers", type=int, help="processes for the pileup and table passes (default: CPUs, max 16)")
    g.add_argument("--cache-dir", help="site cache (default: <out>/cache); reused when inputs are unchanged")
    g.add_argument("--motif-fit", action="store_true",
                   help="also write motif_fit.tsv: each contig's methylated fraction per motif group next to its bin's")

    g = r.add_argument_group("rule variants",
                             "alternative designs of single rules; every default is the behaviour of 0.10")
    g.add_argument("--robust-fit", choices=["damped", "alternating"], default="alternating",
                   help="how chunk weights and bin profiles are iterated: damped steps to convergence, or the fixed four "
                        "alternating rounds of 0.10.x (which end in the pooled profile when a bin holds two methylomes)")
    g.add_argument("--max-flagged-share", type=float, default=1.0,
                   help="below 1: no flag of a bin is applied when its flagged contigs hold at least this share of the "
                        "bin's judged sequence (the bin is reported as a suspected merge of genomes); 1 = off, every "
                        "flag is applied")
    g.add_argument("--regional-leave-out", choices=["contig", "chunk"], default="chunk",
                   help="bin profile the chunks of a long contig are compared with in the regional model: without the "
                        "whole contig (as for the contig's own score) or without the chunk only (0.10.x)")
    g.add_argument("--regional-withdraw", choices=["largest", "vs_foreign"], default="vs_foreign",
                   help="withdraw a flag as mixed when 'mixed' is the most probable of member / foreign / mixed, or "
                        "whenever it is more probable than 'foreign' (0.10.x)")
    g.add_argument("--regional-tempered", action="store_true", help="divide the regional chunk evidence by the temperature")
    g.add_argument("--gate-leave-out", choices=["contig", "none"], default="none",
                   help="methylome-difference gate against the bin's 95%% interval without the contig, or against the "
                        "level of the full bin profile (0.10.x)")
    g.add_argument("--gate-unweighted", action="store_true",
                   help="contig intervals of the methylome-difference gate from unweighted site likelihoods")
    g.add_argument("--eps-gated", action="store_true",
                   help="estimate the contamination rate only from contigs that pass the methylome-difference gate")
    g.add_argument("--unlisted-genome", action="store_true",
                   help="unbinned genomes are community draws or genomes with none of the listed motifs (weights estimated)")
    g.add_argument("--inclusion-support", choices=["methylated_groups", "none"], default="none",
                   help="an inclusion must also reach the threshold on the motif groups its bin methylates alone")
    g.add_argument("--unknown-prior", choices=["unbinned", "bins"], default="bins",
                   help="an unbinned genome's state in each motif group is drawn at the frequency seen among the unbinned "
                        "contigs, or among the bins (0.10.x)")
    g.add_argument("--unknown-prior-strength", type=float, default=5.0,
                   help="with --unknown-prior unbinned: pseudo-contigs at the bins' frequency per motif group")
    g.add_argument("--tau-support", choices=["leave_bin_out", "all"], default="all",
                   help="a motif group gets between-contig variation (finite tau) only if the calibration fragments support "
                        "it without the bin that gains most, or on all fragments (0.10.x)")
    g.add_argument("--two-state-warning", type=float, default=0.2,
                   help="warn about a bin when, in some motif group, the smaller of two methylation states among its "
                        "chunks holds at least this share, or its flagged contigs hold at least this share of the "
                        "bin (bin_evidence.tsv, column warning; reporting only, no decision depends on it)")
    g.add_argument("--redundancy-order", choices=["sequential", "independent"], default="independent",
                   help="compare each inclusion candidate with the given bin only (independent), or with the bin plus "
                        "the inclusions already accepted, most probable first (sequential), so that two unbinned "
                        "copies of one sequence are not both included")
    g.add_argument("--siteless-bin-guard", action="store_true",
                   help="include nothing into a bin without a single motif site (such a bin has no profile; its "
                        "posterior is its prior share)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "run":
        from . import pipeline
        from .io import InputError
        from .runinfo import log
        try:
            pipeline.run(
                a.assembly, a.pileup, a.contig_bin, a.motifs, a.out, min_cov=a.min_cov, chunk_size=a.chunk_size,
                grid_size=a.grid_size, p_unknown=a.p_unknown, contamination_threshold=a.contamination_threshold,
                include_threshold=a.include_threshold, min_methylome_difference=a.min_methylome_difference,
                max_redundancy=a.max_redundancy, min_evidence=a.min_evidence, min_contig_coverage=a.min_contig_coverage, min_contig_length=a.min_contig_length, motif_noise_filter=not a.no_motif_noise_filter, group_weights=not a.no_group_weights, regional_model=not a.no_regional_model, regional_inclusion=a.regional_inclusion, include_threshold_unmethylated=a.include_threshold_unmethylated, do_calibrate=not a.no_calibrate, frag_lengths=a.frag_lengths,
                frags_per_bin=a.frags_per_bin, frag_budget=a.frag_budget, tau=a.tau, temperature=a.temperature,
                contam_eps=a.contam_eps, allow_sharpening=a.allow_sharpening, calibration=a.calibration,
                min_jaccard=a.min_jaccard,
                variation=a.variation, group_sample_bp=a.group_sample_mb * 1e6, workers=a.workers,
                cache_dir=a.cache_dir, write_motif_fit=a.motif_fit, seed=a.seed, shared_sites=a.shared_sites,
                complement_pairs=a.complement_pairs, strand_sites=a.strand_sites,
                noise_max_off_share=a.noise_max_off_share,
                two_state_warning=a.two_state_warning, siteless_bin_guard=a.siteless_bin_guard,
                robust_fit=a.robust_fit,
                max_flagged_share=a.max_flagged_share, regional_leave_out=a.regional_leave_out,
                regional_withdraw=a.regional_withdraw, regional_tempered=a.regional_tempered,
                gate_leave_out=a.gate_leave_out, gate_unweighted=a.gate_unweighted, eps_gated=a.eps_gated,
                redundancy_order=a.redundancy_order, unlisted_genome=a.unlisted_genome,
                inclusion_support=a.inclusion_support, tau_support=a.tau_support, unknown_prior=a.unknown_prior,
                unknown_prior_strength=a.unknown_prior_strength)
        except InputError as e:
            # a problem with the inputs or options: the message says what; no traceback needed
            log(f"ERROR: {e}")
            print(f"epirefine: error: {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            # progress goes to standard output; the traceback follows it there, so that a captured log holds
            # the error, and standard error gets one line
            sys.stdout.flush()
            log("ERROR: the run failed:\n" + traceback.format_exc().rstrip())
            print(f"epirefine: error: {type(e).__name__}: {e} (traceback in the log on standard output)", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
