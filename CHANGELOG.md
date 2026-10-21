# Changelog

Versions follow semantic versioning (see `CLAUDE.md`). Each entry lists what changed and its measured
effect on the benchmark suite (`scripts/benchmark_suite.py`; history in the dev log).

## 0.2.0 (2026-10-21): baseline under version control

Everything before this version was developed without version control; the reasoning is in the dev
log (`sessions/`, `DECISIONS.md`).

**Bin refinement (new focus).**
- Every contig is scored: binned contigs leave-one-out against their own bin, unbinned contigs
  against every bin. `refine.py` turns the posteriors into decisions:
  - contamination: P(foreign) from a mixture whose contamination rate is estimated from the data;
  - inclusion: a contig goes into a bin at posterior ≥ 0.95 (`--include-threshold`); the share of
    unbinned contigs from unbinned genomes is estimated from the data (`--p-unknown` overrides).
- Nanomotif-compatible outputs: `bin_contamination.tsv`, `decontaminated_contig_bin.tsv`,
  `include_contigs.tsv`, `new_contig_bin.tsv`; per-contig decisions in `contig_decisions.tsv`.
- Plasmid-specific parameters (δ, τ_p, own-MTase ρ) apply only to `--plasmids` candidates.
- Unbinned genomes are modelled as a community draw or a relative of a bin (switch probability η,
  weight λ, fitted in calibration with the temperature).
- τ (between-chunk variation) is chosen per motif group.
- Calibration's host-absent test uses only fragments that can tell their bin from its nearest
  competitor.

**Speed and memory (`docs/scalability.md`).**
- Shape-based motif matcher (identical records, 58x faster than regex).
- Parallel streaming pileup pass with a reusable site cache; no in-memory motif records.
- Lookup-table site likelihoods, float32 tables, vectorised NPMLE, summed scores without
  [units × bins × groups] tensors, threaded robust-profile step, fixed calibration budget.
- human_gut (1.73 Gb, 369 bins, 235 GB pileup): about 56 min from scratch, 45 min with the cache,
  against a projected 2 days for 0.1.0.

**Fixes.**
- Caller-model fit could stop at a degenerate optimum depending on row order.
- Motif-group numbering and fragment selection depended on polars group order.
- Fragment sampling assumed sorted input.
- Site-cache paths were relative to the directory the cache was made from.

**Benchmark (vs Nanomotif 1.1.2 include_contigs, same inputs).** ZymoOral, 6 scenarios:
contaminants found 100% (Nanomotif 15–42%), contigs put back 85–97% (46–81%), 0–1 contigs of
unbinned genomes placed in a bin (6–69). Simulated look-alike community: contaminants found
96–98%, almost nothing put back at the 0.95 threshold (methylation cannot separate sibling strains).

## 0.1.0 (2026-10-19)

Plasmid-host assignment from methylation: hierarchical Bayesian model (caller model, discrete
methylation grid, population prior, robust host profiles), pseudo-plasmid calibration,
MTase-aware scoring, assignability, HTML report.
