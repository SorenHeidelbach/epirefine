# Changelog

Versions follow semantic versioning (see `CLAUDE.md`). Each entry lists what changed and its measured
effect on the benchmark suite (`scripts/benchmark_suite.py`; history in the dev log).

## 0.3.0 (2026-10-21): refinement workflow with CheckM2

- `workflow/Snakefile`: per sample, runs Nanomotif 1.1.2 (`detect_contamination` +
  `include_contigs`) and pbinner on the same inputs, writes the original and refined bins as FASTA,
  and scores them with CheckM2 1.1.0 (database downloaded by the workflow). Conda environments are
  in `workflow/envs/`; per-sample `methods` chooses which bin sets to compare.
- `workflow/scripts/compare_bins.R`: per-bin and per-method CheckM2 comparison (quality tiers,
  completeness and contamination change per bin, sequence moved), as tables and a figure page.

**Benchmark effect:** none on pbinner (no pbinner code changed). Workflow results (CheckM2):

| Sample | Bins | High-quality: original → Nanomotif / pbinner | Median completeness | Median contamination |
|---|---|---|---|---|
| ZymoOral with planted errors | 10 | 0 → 4 / 9 | 71.4 → 91.8 / 99.8% | 5.7 → 5.7 / 1.7% |
| ZymoOral as binned | 12 | 11 → 11 / 11 | 100 → 100 / 100% | 0.14 → 0.14 / 0.31% |
| human_gut (Nanomotif skipped) | 369 | 198 → – / 157 | 93.7 → – / 92.1% | 0.93 → – / 1.06% |

On human_gut pbinner's refinement makes bins worse: its contamination flags mostly hit genuine
members (see the dev log, D18).

## 0.2.1 (2026-10-21): project framework

- `CLAUDE.md`: project rules (branches only, commit every change, merge with version bump,
  test gate, benchmark and report the effect of every change, dev log outside git).
- Pre-commit hook (`scripts/hooks/`): refuses commits on `main` and failing tests.
- `pbinner --version`.
- `scripts/benchmark_suite.py` with `benchmarks/suite.json`: records accuracy, wall time and peak
  memory per version in the dev log.

**Benchmark effect:** none; no code that changes results. First recorded baseline (0.2.0 code):

| Benchmark | Contaminants found | Flag precision | Put back | Wrongly placed / scenario | Completeness after | Wall time | Peak RAM |
|---|---|---|---|---|---|---|---|
| ZymoOral 5–150 kb | 100% | 96.5% | 92.6% | 1.0 | 98.7% | 55 s | 1.0 GB |
| ZymoOral 2–30 kb | 100% | 99.1% | 89.9% | 0.3 | 98.2% | 57 s | 1.1 GB |
| Simulated look-alikes | 97.0% | 100% | 0% | 0.7 | 76.1% | 145 s | 2.7 GB |

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
