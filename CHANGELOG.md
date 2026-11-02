# Changelog

Versions follow semantic versioning (see `CLAUDE.md`). Each entry lists what changed and its measured
effect on the benchmark suite (`benchmarks/benchmark_suite.py`; history in the dev log).

## 0.9.2 (2026-11-02): tests on GitHub for every supported Python

No change to results.
- `.github/workflows/tests.yml`: the unit tests run on every push, tag and pull request, on Python 3.10, 3.11,
  3.12, 3.13, 3.14 and 3.15, and once with the lowest dependency versions `pyproject.toml` allows.
- Fix: polars must be >= 1.2 (was >= 1.0). EpiRefine reads pileups with `read_csv(infer_schema=False)`, which
  polars 1.0 and 1.1 don't have; with them pileup parsing failed (2 tests). Found by the lowest-versions job.

**Benchmark effect:** none (no code change). All 35 tests pass locally on Python 3.10-3.15 and on
numpy 1.26.0 / scipy 1.11.1 / polars 1.2.1.

## 0.9.1 (2026-11-02): the repository holds only the tool

No change to the epirefine package or its outputs.
- `workflow/`, `hic/` and the benchmarks other than the monocultures (ZymoOral pieces, simulated
  look-alikes, chimeras, simulator, truth tables, Nanomotif comparison, threshold sweeps) moved to a
  separate repository that calls the tool (`epirefine_workflow`; their history up to here stays in this one).
- `benchmarks/benchmark_suite.py` runs the monoculture benchmark by default and appends a summary per
  version to `benchmarks/results.tsv` in the repository (seeded with the released versions 0.6.1, 0.7.2, 0.9.0).

**Benchmark effect:** none; mono-train identical to 0.9.0 (0.948 / 1.000 / 0.893 / 2.0).

## 0.9.0 (2026-11-02): renamed to EpiRefine

Breaking (name only; no model or result change).
- The tool is now **EpiRefine** (*epi*genetics + *refine*; formerly pbinner): package `epirefine`
  (`src/epirefine/`), command `epirefine` (`epirefine run`, `epirefine --version`). The `pbinner` command is gone.
- Output: the `method` column of the Nanomotif-compatible `bin_contamination.tsv` and
  `include_contigs.tsv` now says `epirefine`. Columns and every other value are unchanged.
- Workflow: the method is called `epirefine` (output folder `<sample>/epirefine/`, config keys `epirefine`,
  `epirefine_extra`); `benchmarks/compare_nanomotif.py --epirefine`. Data and cache folders on disk keep
  their old names.

**Benchmark effect:** none; every benchmark identical to 0.8.0 (mono-train 0.948 / 1.000 / 0.893 / 2.0;
ZymoOral 5-150 kb, 2-30 kb and the simulated look-alikes unchanged). Outputs byte-identical except the `method` value.

## 0.8.0 (2026-10-29): regional inclusion off by default

Changes default results and adds a CLI flag.
- Long unbinned contigs are no longer included from their 50 kb windows (the regional model) when
  the whole contig doesn't support the bin. `--regional-inclusion` turns it back on (experimental).
  Regional flagging of long binned contigs ("mixed") is unchanged.
- Why (dev log D40): no benchmark gains an inclusion from it. On ZymoFecal it added 69 inclusions
  (24.5 Mb); for 58 the whole-contig model's best explanation is "unknown" (P 0.54-1, 42 above 0.99).

**Benchmark effect:** none; every scenario is identical to 0.7.2 (mono-train, ZymoOral 5-150 kb and
2-30 kb, simulated look-alikes, chimeras).
**ZymoFecal:**
- inclusions 554 -> 485;
- strict Hi-C (>= 10x): flags 6/1/2 (TP/FP/FN) unchanged; inclusions 24 right / 2 wrong -> 23 / 1;
- peak memory 44.1 -> 40.9 GB.

## 0.7.2 (2026-10-28): monoculture benchmark sets (benchmarks/ and workflow/ only)

No change to the pbinner package.
- `benchmarks/monoculture_sets.py` builds synthetic communities from 40 isolates sequenced one at a
  time (so every read's origin is known):
  - genus-level split into training (6 communities), test (3) and validation (3);
  - 8 isolates per community;
  - pileups thinned to 3-60x;
  - assemblies cut into 2-100 kb pieces;
  - errors planted with the standard generator.
- `workflow/monocultures.smk` makes the missing modkit pileups and Nanomotif motifs.
- `benchmarks/suite_monocultures.json` lists the scenarios. Test and validation communities are held
  out for the final estimate.

**Benchmark effect:** none (no package change).

## 0.7.1 (2026-10-28): Hi-C membership by contact density (hic/ only)

No change to the pbinner package. `hic/density.py` calls contig membership from Hi-C contact density
relative to the bin's own internal density (ProxiMeta-style ratio):
- **member:** ratio >= 0.1, >= 5 links, to >= 2 of the bin's other contigs.
- **foreign:** another eligible bin is clearly preferred.
- **not linked:** own-bin contacts improbably few for a member: below the 1% quantile of a
  beta-binomial with the bin's pooled own-contact share and an over-dispersion fitted in-sample
  (ZymoFecal rho 0.28).
- Placed / not placed for unbinned contigs.

ZymoFecal, full library (416 M read pairs), binned contigs: 4,062 members, 10 foreign, 178 not
linked, 2,121 undetermined. The earlier calls had 5,047 members and 230 not linked.
**Benchmark effect:** none (no package change).

## 0.7.0 (2026-10-28): coverage cutoff, motif groups, regional model, and new statuses and columns

**Breaking for downstream parsers:** `contig_decisions.tsv` gains statuses `low_coverage`, `too_short`,
`no_methylation_evidence` and `mixed`, and columns `expected_evidence`, `coverage`,
`p_regional_foreign`, `p_regional_mixed`, `foreign_segments` and `segment_bins`. A new file,
`bin_evidence.tsv`, is written. The Nanomotif-compatible files keep their columns.

Model and decisions:
- **No decision below 5x** median coverage at motif sites, counted per strand (status `low_coverage`,
  `--min-contig-coverage`), or below 1 kb (status `too_short`, `--min-contig-length`).
- **Status `no_methylation_evidence`** for contigs whose sites cannot show where they belong
  (expected log Bayes factor below 3 against a random community genome).
- **Motif groups:** motifs are merged only when their sites are near-identical (Jaccard >= 0.9), and
  shared positions go to the most specific group. ZymoFecal: 952 motifs, 713 groups.
- **Noise filter:** groups whose sites show intermediate calls in >= 20% of bins are dropped
  (ZymoFecal: 8 4mC groups on CCNGG-type contexts). `--no-motif-noise-filter` turns it off.
- **Group weights:** variants of one methyltransferase share one group's weight (ZymoFecal: 81
  groups in 16 clusters). `--no-group-weights` turns them off.
- **Unknown-genome model:** community-only unless relatives fit significantly better (paired
  bootstrap); the per-group swap probability is capped at 0.05.
- **Thresholds:** flags need a methylome difference >= 0.4, and inclusion needs P >= 0.9, or >= 0.95
  for bins with no detected methylation (`--include-threshold-unmethylated`).
- **Regional model:** long binned contigs are judged along their length (member / foreign / mixed,
  a two-state chain over 50 kb chunks). Flags better explained as mixed are withdrawn, and the
  contig is reported with its foreign segments and a segment map. Long unbinned contigs are scored
  in windows, with inclusion tied to the model's share of genomes without a bin.
  `--no-regional-model` turns it off.
- Evidence pass: cache parts run serially, and the expectation table is shared (fixes a heap
  corruption from oversubscribed OpenBLAS threads on ZymoFecal).
- Benchmarks leave undecided contigs out of the metrics and count them separately. New chimera
  benchmark: `benchmarks/chimera_sets.py`, `benchmarks/evaluate_regional.py`.

**Benchmark effect** (39ab5ae, decided contigs; flag precision 1.00 on every benchmark):
- Monocultures: contaminants found 0.948, put back 0.893, 2.0 wrong placements per scenario.
- ZymoOral 5-150 kb: 1.000 / 0.927. ZymoOral 2-30 kb: 0.996 / 0.876.
- Simulated look-alikes: 0.967 found; nothing included.
- Chimeras: 33 of 40 reported as mixed instead of flagged; 20 of 20 long contaminants flagged.

**ZymoFecal against the strict full-library Hi-C (contigs >= 10x):**
- Flag precision 0.40 → 0.86 and recall 0.75 → 0.75 (TP/FP/FN 6/9/2 → 6/1/2) against the 0.6.1 candidate settings.
- Inclusions in the Hi-C bin: 24 of 26.
- Run: 86 min, 44 GB, 4 workers.

## 0.6.1 (2026-10-26): Hi-C validation of refinement (workflow and hic/ only)

No change to the pbinner package. Workflow and `hic/`:
- **Strict Hi-C calls.** Pairs need MAPQ >= 20 on both reads (`hic_mapq`; chromap's Hi-C preset
  used 1) and the `.pairs` file is kept. A contig's Hi-C bin is the bin with most contacts; a call
  is confident only with >= 20 contacts, >= 80% of the contig's contacts to bins, >= 3 distinct
  partner contigs, >= 10 contacts without its strongest partner, and P(random ligation) <= 1e-6.
  On ZymoFecal contigs of CheckM2 high-quality bins: 0 of 139 called to another bin (the earlier
  rule: 33 of 343, nearly all resting on a single partner contig).
- `contacts.py` counting and calling are separate steps (rules `hic_pairs`, `hic_contacts`).
- New analyses: `refine.py` (Hi-C-only refinement as a CheckM2 reference), `nonredundant.py`,
  `bin_quality.py`, `marker_visibility.py` (can CheckM2 see Hi-C-foreign contigs?),
  `within_vs_outside.py` (contacts inside vs outside the bin), `probability_vs_hic.py` (Hi-C
  posterior per contig), `calibrate_with_hic.py` (pbinner P(contaminant) -> Hi-C-foreign rate),
  with R plots and workflow rules.
- ZymoFecal sample in the config (original, pbinner and Hi-C bin sets; Nanomotif left out).

**Findings (ZymoFecal):** CheckM2 can't see most Hi-C-foreign sequence (78% carries no duplicated
single-copy marker); pbinner flags are Hi-C-foreign 36-44% of the time against a base rate of 11%.
**Benchmark effect:** none (no package change).

## 0.6.0 (2026-10-26): calibration leaves the whole contig out

- Calibration scores each pseudo-contig against its bin's profile with *every* chunk of its source
  contig removed, as contig decisions already did. Before, only the chunks the fragment overlapped
  were removed, so the rest of its own contig (a median 33-52% of the profile) stayed in, and
  calibration measured agreement within a contig. Fragments whose contig is its whole bin are skipped.

**Benchmark effect:** ZymoOral and simulated benchmarks identical (mock genomes have no
between-contig variation). ZymoFecal (real gut community, checked against Hi-C, MAPQ >= 20 and strict
calls): the unknown-genome model no longer absorbs everything (share of unbinned contigs from bin-less
genomes 0.995 -> 0.822); inclusions 5 -> 495; of the 380 unbinned contigs Hi-C places confidently, 42
included, all in Hi-C's bin (was 1); flags 460 -> 431, Hi-C precision of flags 31% -> 36%.
human_gut: CheckM2 203 vs 204 high-quality bins, median completeness 95.0 vs 95.1%, contamination 1.00%
(unchanged within noise); run time 25 vs 36 min.

## 0.5.0 (2026-10-23): bin refinement only; Hi-C validation in the workflow

**Breaking.**
- The package does bin refinement only (D22). Removed: plasmid-host assignment (`--plasmids`,
  `--mtases`, `assignments.tsv`, `posteriors.tsv`, `plasmid_param_fit.tsv`), the own-MTase model,
  detectability, and the `simulate`, `evaluate` and `report` subcommands. `pbinner run` is the only
  command.
- Repository layout: `src/pbinner/` (package), `workflow/`, `hic/`, `benchmarks/` (was
  `scripts/`), `dev/` (hooks, prototypes), `docs/`.

**New.**
- `--variation logit`: a logit-normal kernel for how a contig's methylated fraction varies around
  its bin's (default stays `beta`); `calibration/temperature_by_length.tsv` diagnostic.
- `hic/`: Hi-C contacts per contig and bin, a Hi-C bin call per contig, and scoring of each
  method's flags and inclusions against it. The workflow runs it (rules `hic_*`) for samples with
  `hic_r1`/`hic_r2`.

**Benchmark effect:** all 9 scenarios give the same status and bin for every contig as 0.4.0;
metrics equal to 1e-4. One ZymoOral contig (142 kb) changed P(foreign) 0.38 → 0.97 without a
status change (the own-MTase term removed; the methylome-difference gate keeps it in its bin).
The `--variation logit` option gives identical decisions on ZymoOral 2–30 kb; its human_gut test
is pending.

## 0.4.1 (2026-10-23): faster iteration

- Detectability and the redundancy check run threaded.
- `benchmarks/benchmark_suite.py --jobs N` runs scenarios in parallel (timings of such runs are
  marked and not compared with single-job runs).

**Benchmark effect:** results identical (same decisions on all scenarios). The suite takes about a
third of the wall time with `--jobs 3`; the human_gut timing of the threaded steps is reported with
the next human_gut run.

## 0.4.0 (2026-10-23): refinement that holds up on a real gut community

- **Caller model fitted with the bins as labels** (D19). The pooled two-component fit collapsed in
  365 of 367 motif groups on human_gut (a motif methylated in a few bins looks like noise in the
  pool); each group is now fitted with bin membership as labels.
- **A contamination call needs a methylome difference** (D20): besides P(foreign) ≥ 0.5, the
  contig's 95% credible interval for some motif group's methylated fraction must lie at least
  `--min-methylome-difference` (0.3) from its bin's level. New column `methylome_difference`.
- **No inclusion of sequence the bin already holds** (D21): FracMinHash containment of the contig in
  the bin ≥ `--max-redundancy` (0.3) gives status `redundant` instead of `included`. New column
  `redundancy_with_bin`.
- `contig_decisions.tsv` gains `credible_set_95`; `--motif-fit` writes `motif_fit.tsv`.

**Benchmark effect** (vs 0.2.0 code, the last recorded baseline; 3 scenarios each):

| Benchmark | Contaminants found | Flag precision | Put back | Wall time |
|---|---|---|---|---|
| ZymoOral 5–150 kb | 100 → 100% | 96.5 → 100% | 92.6 → 86.2% | 55 → 37 s (−32%) |
| ZymoOral 2–30 kb | 100 → 99.7% | 99.1 → 99.7% | 89.9 → 82.8% | 57 → 37 s (−34%) |
| Simulated look-alikes | 97.0 → 96.4% | 100 → 100% | 0 → 0% | 145 → 122 s (−16%) |

Fewer contigs are put back (the redundancy filter refuses assembly duplicates that the benchmark
truth counts as members). human_gut, CheckM2 (369 bins):

| | Original | 0.3.0 | 0.4.0 |
|---|---|---|---|
| High-quality bins | 198 | 157 | 204 |
| Low-quality bins | 1 | 46 | 3 |
| Median completeness | 93.7% | 92.1% | 95.1% |
| Median contamination | 0.93% | 1.06% | 1.00% |

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
