from __future__ import annotations

import argparse

from .calibrate import DEFAULT_LENGTHS


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="pbinner", description="Bin refinement from nanopore methylation: flags "
                                 "contaminating contigs in bins and places unbinned contigs into bins")
    ap.add_argument("--version", action="version", version=f"pbinner {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="refine bins: contamination detection and contig inclusion")
    r.add_argument("--assembly", required=True)
    r.add_argument("--pileup", required=True, help="modkit pileup bedMethyl (.bed or .bed.gz)")
    r.add_argument("--contig-bin", required=True, help="contig<TAB>bin, no header")
    r.add_argument("--motifs", required=True, help="Nanomotif bin-motifs.tsv (only the motif set is used)")
    r.add_argument("--out", required=True)
    r.add_argument("--contamination-threshold", type=float, default=0.5,
                   help="flag a binned contig as contamination when P(foreign) is at least this")
    r.add_argument("--min-methylome-difference", type=float, default=0.4,
                   help="a contaminant's 95%% credible interval for the methylated fraction of some motif group "
                        "must lie at least this far from its bin's level (0 = no effect-size requirement)")
    r.add_argument("--include-threshold", type=float, default=0.9,
                   help="include a contig into a bin when that bin's posterior is at least this")
    r.add_argument("--include-threshold-unmethylated", type=float, default=0.95,
                   help="include a contig into a bin with no detected methylation only at this posterior (such a bin matches "
                        "any contig whose own motifs are not in the motif list)")
    r.add_argument("--min-evidence", type=float, default=3.0,
                   help="contigs whose expected methylation evidence (log BF of their bin vs a random community "
                        "genome) is below this get status no_methylation_evidence instead of kept/unbinned")
    r.add_argument("--min-contig-coverage", type=float, default=5.0,
                   help="contigs whose median reads per motif site is below this get no decision (neither "
                        "flagged nor included; status low_coverage) and give no calibration fragments; 0 decides all")
    r.add_argument("--min-contig-length", type=int, default=1000,
                   help="contigs shorter than this get no decision (status too_short); 0 decides all")
    r.add_argument("--no-group-weights", action="store_true",
                   help="count every motif group fully, even variants of one methyltransferase (default: groups that can "
                        "share sites and are methylated in the same bins share one group's weight; see group_weights.tsv)")
    r.add_argument("--no-motif-noise-filter", action="store_true",
                   help="keep motif groups whose methylation is partial on many contigs of many bins (by default "
                        "they are dropped as caller noise; see motif_noise.tsv)")
    r.add_argument("--tau-mode", choices=["outlier", "select"], default="outlier",
                   help="between-contig variation per motif group: contigs match their bin except for rare, large "
                        "departures whose rate is estimated per group (outlier, default; see outlier_rates.tsv), or a "
                        "Beta(tau) spread selected by calibration likelihood (select, the earlier method)")
    r.add_argument("--max-redundancy", type=float, default=0.3,
                   help="don't include a contig when at least this share of its k-mers is already in the bin "
                        "(1 = no redundancy check)")
    r.add_argument("--p-unknown", type=float, help="prior probability that an unbinned contig's genome has no bin "
                                                  "(default: estimated from the unbinned contigs)")
    r.add_argument("--variation", choices=["beta", "logit"], default="beta",
                   help="how a contig's methylated fraction varies around its bin's: Beta(tau) or "
                        "logit-normal(sigma); --tau then sets tau or sigma")
    r.add_argument("--tau", type=float, help="override the calibrated variation parameter (tau or sigma)")
    r.add_argument("--temperature", type=float, help="override the calibrated temperature")
    r.add_argument("--allow-sharpening", action="store_true",
                   help="use a fitted temperature < 1 (default: clip at 1, never sharpen posteriors)")
    r.add_argument("--no-calibrate", action="store_true")
    r.add_argument("--frag-lengths", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEFAULT_LENGTHS)
    r.add_argument("--frags-per-bin", type=int, default=30, help="cap on calibration fragments per bin and length")
    r.add_argument("--frag-budget", type=int, default=6000,
                   help="total calibration fragments; spread over bins and lengths (at most --frags-per-bin each)")
    r.add_argument("--contam-eps", type=float, default=0.05, help="prior share of foreign chunks in a bin (0 = off)")
    r.add_argument("--min-cov", type=int, default=1)
    r.add_argument("--chunk-size", type=int, default=50_000, help="binned contigs are split into chunks of this size")
    r.add_argument("--grid-size", type=int, default=64)
    r.add_argument("--min-jaccard", type=float, default=0.9,
                   help="merge motifs whose site sets overlap this much (Jaccard); positions shared by motifs "
                        "of different groups go to the most specific group")
    r.add_argument("--group-sample-mb", type=float, default=50,
                   help="Mb of sequence sampled to measure motif overlap for grouping")
    r.add_argument("--workers", type=int, help="processes for the pileup and table passes (default: CPUs, max 16)")
    r.add_argument("--cache-dir", help="site cache (default: <out>/cache); reused when inputs are unchanged")
    r.add_argument("--motif-fit", action="store_true",
                   help="also write motif_fit.tsv: each contig's methylated fraction per motif group next to its bin's")
    r.add_argument("--seed", type=int, default=1)

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run
        run(a.assembly, a.pileup, a.contig_bin, a.motifs, a.out, min_cov=a.min_cov, chunk_size=a.chunk_size,
            grid_size=a.grid_size, p_unknown=a.p_unknown, contamination_threshold=a.contamination_threshold,
            include_threshold=a.include_threshold, min_methylome_difference=a.min_methylome_difference,
            max_redundancy=a.max_redundancy, min_evidence=a.min_evidence, min_contig_coverage=a.min_contig_coverage, min_contig_length=a.min_contig_length, motif_noise_filter=not a.no_motif_noise_filter, group_weights=not a.no_group_weights, tau_mode=a.tau_mode, include_threshold_unmethylated=a.include_threshold_unmethylated, do_calibrate=not a.no_calibrate, frag_lengths=a.frag_lengths,
            frags_per_bin=a.frags_per_bin, frag_budget=a.frag_budget, tau=a.tau, temperature=a.temperature,
            contam_eps=a.contam_eps, allow_sharpening=a.allow_sharpening, min_jaccard=a.min_jaccard,
            variation=a.variation, group_sample_bp=a.group_sample_mb * 1e6, workers=a.workers,
            cache_dir=a.cache_dir, write_motif_fit=a.motif_fit, seed=a.seed)


if __name__ == "__main__":
    main()
