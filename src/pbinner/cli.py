from __future__ import annotations

import argparse

from .calibrate import DEFAULT_LENGTHS


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(prog="pbinner", description="Bin refinement (contamination detection, contig inclusion) "
                                                             "and plasmid-host assignment from nanopore methylation")
    ap.add_argument("--version", action="version", version=f"pbinner {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="assign plasmid contigs to host bins")
    r.add_argument("--assembly", required=True)
    r.add_argument("--pileup", required=True, help="modkit pileup bedMethyl (.bed or .bed.gz)")
    r.add_argument("--contig-bin", required=True, help="contig<TAB>bin, no header")
    r.add_argument("--motifs", required=True, help="Nanomotif bin-motifs.tsv (only the motif set is used)")
    r.add_argument("--plasmids", help="plasmid candidates: one contig per line or geNomad plasmid_summary.tsv. "
                                      "Default: score all unbinned contigs")
    r.add_argument("--mtases", help="MTase annotation TSV (contig, mod_type, motif; MTase-linker-like). "
                                    "Annotate host contigs too, so rho can be estimated")
    r.add_argument("--out", required=True)
    r.add_argument("--score-all", action="store_true", help="(kept for compatibility: every contig is always scored)")
    r.add_argument("--contamination-threshold", type=float, default=0.5,
                   help="flag a binned contig as contamination when P(foreign) is at least this")
    r.add_argument("--include-threshold", type=float, default=0.95,
                   help="include a contig into a bin when that bin's posterior is at least this")
    r.add_argument("--min-cov", type=int, default=1)
    r.add_argument("--chunk-size", type=int, default=50_000, help="host contigs are split into chunks of this size")
    r.add_argument("--grid-size", type=int, default=64)
    r.add_argument("--p-unknown", type=float, help="prior probability that an unbinned contig's genome has no bin "
                                                  "(default: estimated from the unbinned contigs)")
    r.add_argument("--no-calibrate", action="store_true")
    r.add_argument("--frag-lengths", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEFAULT_LENGTHS)
    r.add_argument("--frags-per-bin", type=int, default=30, help="cap on fragments per bin and length")
    r.add_argument("--frag-budget", type=int, default=6000,
                   help="total calibration fragments; spread over bins and lengths (at most --frags-per-bin each)")
    r.add_argument("--workers", type=int, help="processes for the pileup and table passes (default: CPUs, max 16)")
    r.add_argument("--cache-dir", help="site cache (default: <out>/cache); reused when inputs are unchanged")
    r.add_argument("--group-sample-mb", type=float, default=50,
                   help="Mb of sequence sampled to measure motif overlap for grouping")
    r.add_argument("--tau", type=float, help="override calibrated tau")
    r.add_argument("--temperature", type=float, help="override calibrated temperature")
    r.add_argument("--delta", type=float, help="override fitted plasmid under-methylation factor")
    r.add_argument("--tau-p", type=float, help="override fitted plasmid-vs-host concentration")
    r.add_argument("--rho-base", type=float, default=0.02, help="prior P(own MTase) for unannotated plasmid/group")
    r.add_argument("--contam-eps", type=float, default=0.05, help="prior fraction of foreign chunks in a bin (0 = off)")
    r.add_argument("--allow-sharpening", action="store_true",
                   help="use a fitted temperature < 1 (default: clip at 1, never sharpen posteriors)")
    r.add_argument("--min-containment", type=float, default=0.5, help="site overlap needed to merge motifs")
    r.add_argument("--seed", type=int, default=1)

    s = sub.add_parser("simulate", help="write a synthetic dataset with known truth")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=1)

    e = sub.add_parser("evaluate", help="compare assignments.tsv to a truth table (contig, expected)")
    e.add_argument("--assignments", required=True)
    e.add_argument("--truth", required=True)

    rp = sub.add_parser("report", help="write an interactive HTML report for a run directory")
    rp.add_argument("--run", required=True, help="output directory of `pbinner run`")
    rp.add_argument("--out", help="default: <run>/report.html")
    rp.add_argument("--truth", help="optional truth table (contig, expected) to mark the true host")
    rp.add_argument("--title")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import run
        run(a.assembly, a.pileup, a.contig_bin, a.motifs, a.out, plasmids=a.plasmids, mtases=a.mtases,
            score_all=True, contamination_threshold=a.contamination_threshold, include_threshold=a.include_threshold, delta=a.delta, tau_p=a.tau_p, rho_base=a.rho_base, contam_eps=a.contam_eps, allow_sharpening=a.allow_sharpening,
            min_cov=a.min_cov, chunk_size=a.chunk_size, grid_size=a.grid_size, p_unknown=a.p_unknown,
            do_calibrate=not a.no_calibrate, frag_lengths=a.frag_lengths, frags_per_bin=a.frags_per_bin,
            frag_budget=a.frag_budget, workers=a.workers, cache_dir=a.cache_dir, group_sample_bp=a.group_sample_mb * 1e6,
            tau=a.tau, temperature=a.temperature, min_containment=a.min_containment, seed=a.seed)
    elif a.cmd == "simulate":
        from .simulate import simulate
        simulate(a.out, seed=a.seed)
    elif a.cmd == "report":
        from .report import write_report
        print(write_report(a.run, a.out, a.truth, a.title))
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        print(evaluate(a.assignments, a.truth))


if __name__ == "__main__":
    main()
