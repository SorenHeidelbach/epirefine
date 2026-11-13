# Changelog

Versions follow semantic versioning (see `CLAUDE.md`). Each entry lists what changed and its measured
effect on the benchmark suite (`benchmarks/benchmark_suite.py`; history in the dev log).

## Unreleased (0.12)

### Breaking: options that lost are removed
The command line no longer offers the options below. None was a default; each lost on the development benchmarks or
was a design that was tried and rejected (dev log U12, D63; the reason for each is in
`docs/development/method-notes.md` §7.6 and the dev log). At the defaults nothing changes: on the 15 development
scenarios of the small suites (monoculture training, ZymoOral, simulated look-alikes) every decision file is
byte-identical to the run before the removal. A command line that names one of them now fails with argparse's
usage error, and `epirefine rerun` of a `params.json` that used one at a non-default value stops with one line
naming it (the same option at its old default, as every 0.11.1 record holds, is accepted).
- `--calibration` with its sets (`default`, `legacy`, `consistent`, `coherent`) and features (`temper-then-mix`,
  `prior-averaged`, `prior-matched`, `conservative-temperature`, `full-test-b`, `unit-temperature`,
  `sites-temperature`, `common-tau`, `em-priors`, `tempered-bin-prior`, `deployed-unknown`). The 0.11.1 default
  (`default`: the relatives test's standard error over bins) is built in; the module-level switch of
  `tempered-bin-prior` is gone with it.
- `--robust-fit` (`damped`: no flag or inclusion changed on any development set).
- `--max-flagged-share` (withheld 8 of 12 correct flags in a merge stress set and fired in none of 42 merges). The
  `flagged_share` column and the `warning` column of `bin_evidence.tsv` stay; the warning
  `merged_bin_suspected_flags_not_applied` can no longer occur.
- `--tau-support` (`leave_bin_out`: no decision changed).
- `--unlisted-genome` and `--inclusion-support` (rejected inclusion designs: inclusion recall 89.3% -> 87.4% and
  58.4% on the training communities, development). `params.json` no longer has `unlisted_share`.
- `--regional-inclusion` (no benchmark gained an inclusion; on ZymoFecal it placed long contigs the contig-level
  model gives to an unknown genome).
- The calibration-core designs that lost (mission WP3): `--tau-select`, `--npmle-start`, `--relatives per-bin`,
  `--relatives slot` with `--slot-objective`, `--slot-test-b` and `--slot-kappa`; with them the report files
  `hidden_relative.tsv` and `calibration/relatives_by_bin.tsv` (written only by `per-bin` and `slot`) and the
  `params.json` keys `relatives_per_bin` and `relatives_slot`. The hidden diagnostic `--calibration-exclusion` too.
- `params.json` no longer holds the keys of these options (`calibration`, `calibration_features`, `robust_fit`,
  `max_flagged_share`, `tau_support`, `unlisted_genome`, `inclusion_support`, `regional_inclusion`, `tau_select`,
  `npmle_start`, `calibration_exclusion`, `slot_objective`, `slot_test_b`, `slot_kappa`) nor
  `unknown_model_deployed_check`. No column of any output table changed.
- Kept: `--relatives switch` (hidden, deprecated; the 0.11.1 path, for comparison with option E), now used
  automatically only with `--no-calibrate`; `--unknown-share`; the experimental `--temperature-law` and
  `--evidence-temperature`; and the rule variants not yet judged (`--regional-leave-out`, `--regional-withdraw`,
  `--regional-tempered`, `--gate-leave-out`, `--gate-unweighted`, `--eps-gated`, `--unknown-prior`,
  `--redundancy-order`, `--siteless-bin-guard`, and the site options).

### Changed
- **Method: the unknown-model switch is removed (new default, "option E").** Up to 0.11.1 EpiRefine chose, during
  calibration, between two models of a genome without a bin: a random draw from the community, or a relative of
  one of the bins, the latter kept when it beat the former by 2 standard errors over bins. That choice sat on a
  knife edge: on a fecal development sample (ZymoFecal, Nanomotif 1.3.0 motifs) the statistic was 0.4-1.9 SE on
  five sets of calibration pieces, and when the bins of the calibration were resampled it chose the relatives in
  2.5-20% of 200 resamples, every time with no or almost no inclusion (gate R4: 129-165 of 200 resamples kept the
  outcome). Now every run uses the community model (`--relatives community-only`, the default) and estimates the
  share of unbinned contigs from genomes without a bin at temperature 1 (`--unknown-share t1`, the default), where
  it does not rise with the temperature. With both, 200 of 200 resamples kept the outcome on each of the three
  development calibration sets (dev log D61-D63). The methylation-evidence check (`--min-evidence`) divides by the
  temperature of this community-only fit.
  Measured (development data): on ZymoFecal (calibration seeds 1-3) flags are unchanged (147), inclusions rise
  from 503 to 589 with the Hi-C agreement unchanged in rate, and the estimated unknown share falls from 0.69 to
  0.41; on the monoculture training set and ZymoOral every flag is unchanged and 2-3 more contigs per seed return to
  their own bin. On the simulated look-alikes (unbinned genomes that copy a bin's methylation) the change puts
  back 31% of the unbinned contigs of binned genomes (0.11.1: 0%), but also places 112 contigs of unbinned
  genomes into the bin of a same-methylome sister genome, and leaves 14-21% of contigs with no call: see Known
  limitations below and the FAQ.
- **Relatives statistic on every run.** The test the switch used is still computed and reported as a number, with
  no threshold: one log line (`relatives statistic: ...`) and `params.json` `relatives_statistic` (gain, standard
  error, their ratio `statistic`, and what the 0.11 switch would have chosen). Large positive values mean some
  calibration pieces are explained better by an unbinned relative of a bin than by a random community genome
  (development data: 4.4-5.4 on the simulated look-alikes, 0.4-1.9 on ZymoFecal). When the switch would have
  chosen the relatives, `params.json` also holds `unknown_model_forced`; the calibration tables (`summary.tsv`,
  `reliability.tsv`, `temperature_by_length.tsv`, `assignability.tsv`, `fragments.tsv`) and the log line `unknown
  model: ... -> used T` then describe the community-only fit the decisions use, and `fitted_temperature` is that
  fit's best temperature before the never-sharpen rule.
- `--relatives switch` (hidden, deprecated, for comparison only) restores the switch of 0.11.1 and, unless
  `--unknown-share` is given, its share estimate at the deployed temperature: it reproduces 0.11.1. A run with
  `--no-calibrate` uses it when `--relatives` is not given, and says so in the log. `params.json` records the
  `relatives` and `unknown_share` actually used; `epirefine rerun` of a `params.json` written before 0.12 uses the
  switch.

- **Guard for calibrations without a usable temperature.** When the calibrated temperature ends at the cap of the
  grid (20), the calibration pieces were not predicted better by any temperature the grid allows, and the fit
  says nothing about how far the scores can be trusted. Such a run now makes no call (no flag, inclusion or move;
  the posteriors are still written, for inspection only; `include_contigs.tsv` lists no candidate), writes one log line `WARNING: calibration guard: ...`
  and records `params.json` `calibration_guard` (`fired`, `reason`, `temperature`, `cap`, the numbers of
  calibration pieces and of test-B pieces, `fallback`). `--temperature` bypasses it. With the experimental
  `--temperature-law sites` it checks the law instead (its temperature at 1,000 sites at the law's cap, 100), since
  the decisions then use the law's temperature, not the grid's. It is checked on the default
  path only (not with the deprecated `--relatives switch`). Measured (development data): it fires on none of the
  normal development runs (temperature at most 5.9 on the monoculture training set, ZymoOral, the simulated
  look-alikes, ZymoFecal and its 600 bin resamples, fecal_inhouse) and on 3 of the 653 stress-set-2 runs that
  finish (every
  other one has a temperature of at most 1.7), all three of which made no call under option E anyway; decisions
  are unchanged everywhere. The number of calibration pieces does not trigger it: from a single piece the
  calibration usually lands at temperature 1.08 and the calls are right.

### Known limitations (new in 0.12)
- **Unbinned genomes with a bin's methylation.** Contigs from a genome that has no bin but has the same, or nearly
  the same, methylation as a binned genome (a strain or close sister that methylates the same motifs) are placed
  into that genome's bin. Methylation cannot tell the two apart; with the switch of 0.11.1 such samples usually
  got no inclusions at all. On such samples the temperature also rises (simulated look-alikes: 3.7-4.3 against 1.3
  under 0.11.1), and because the evidence check is evidence / T < `--min-evidence`, more short contigs get no call
  (about 12-35% of the contigs, depending on where the temperature lands: look-alike set s1 has 245, 396 and 619 no
  calls at T 3.7, 4.3 and 5.0 of its 1,849 contigs; development data). A no call never changes a bin: the status is
  written last, and flags, inclusions and the refined bins do not depend on it. A large relatives statistic warns
  of such a sample.

- **A calibration at the temperature cap gives no calls.** When the temperature of the calibration ends at the cap
  of the grid (20), the guard above withholds every call. On development data this happened only on stress samples
  of two bins with a single calibration piece without usable sites (or a confidently wrong one) or with half of the
  methylation knocked out. On such a sample a fixed temperature (`--temperature`) can still give calls, but they
  are not calibrated: on one development stress run, `--temperature 1` placed 25 contigs, one of them wrongly.

### Fixed
- **Site-cache key.** The manifest now has a format number (2) and a content fingerprint of the assembly and the
  pileup (SHA-256 of three 1 MiB blocks: start, middle, end; files up to 16 MiB whole). A pileup replaced by one of
  the same size and modification time was reused with the old sites; it is now rebuilt (the strict expected failure
  in `tests/test_cache.py` is fixed). A cache whose part files are missing is rebuilt instead of failing later.
  Caches written before 0.12 (format 1) stay valid: they are reused on path, size and time as before, and
  `epirefine cache upgrade` adds the fingerprints, reading only the blocks.
- **Lock on the cache directory.** Two runs no longer write one cache at once: a run that rebuilds the cache holds
  an exclusive lock, runs that read it a shared one, so readers do not wait for each other.

### Added
- `--calibration-seed` (default: the value of `--seed`): draws the calibration pieces and nothing else. The
  motif-grouping sample and the calibration pieces now have their own random generators; up to 0.11.1 one generator
  drew both, so another `--seed` also changed the motif grouping and, on an assembly larger than the grouping sample
  (`--group-sample-mb`, 50 Mb), rebuilt the site cache (hours on a large pileup). At the default the grouping and
  the pieces are exactly those of 0.11.1 (tested against stored 0.11.1 draws, and on ZymoFecal's two motif tables
  against the groups of their site caches), so no decision changes. `params.json` records `calibration_seed`.
- `params.json` records the code and where it ran: `commit` (from git, `EPIREFINE_COMMIT` or a `COMMIT` file),
  `libraries`, `kernels` (installed, used, build hash), `threads` and `seeds`. `epirefine rerun --params
  params.json --out DIR` repeats a run from it (tested: same decisions).
- Input checks with one-line errors: a contig listed in two bins (it was silently put in the last one), a
  contig-bin table with its columns swapped, a motif table without the needed columns (names the missing ones).
  Bins none of whose contigs are in the assembly are named in the log. Every development input (monoculture,
  ZymoOral, simulated, ZymoFecal, fecal_inhouse contig-bin tables) passes as before.
- `epirefine cache check`: says whether a run with given inputs would reuse a cache ("reused") or rebuild it
  ("would rebuild", with the reason), without the pass over the pileup.

## 0.11.1 (2026-11-12): the unknown-genome model no longer absorbs every unbinned contig

0.11.0 was never published; this is the first published version of the 0.11 series.

### Fixed
- **Inclusion collapse with some motif tables.** With a Nanomotif 1.3.0 motif table on a fecal metagenome
  (ZymoFecal, 34,878 contigs, 490 bins), calibration chose the unknown-genome model with relatives of the bins at a
  low temperature, the estimated share of unbinned contigs without a bin ran to 1 and no unbinned contig was
  included (0.10.1, 0.10.2 and 0.11.0; with a Nanomotif 1.1.3 table the same code included 485). The test that keeps
  the relatives model only when it beats the community-only model by 2 standard errors took its standard error from a
  bootstrap over single calibration pieces. The evidence for relatives comes from sibling pairs of bins (5 of 213
  bins held 44% of it), so the pieces are not independent: the gain 0.142 was 2.9 standard errors over pieces but 1.8
  over bins. The standard error is now taken over bins (calibration feature `bin-parsimony`, the default;
  `--calibration legacy` is the rule of 0.10 and 0.11.0).
  - ZymoFecal, Nanomotif 1.3.0 table: 0 -> 503 inclusions, flags 150 -> 147, contigs moved to another bin 0 -> 18,
    share without a bin 1.000 -> 0.688; six sets of calibration pieces give 503-515 inclusions and the same 147
    flags; with calibration pieces up to 500 kb 501 inclusions. Against Hi-C at >= 10x: 7 of 8 judged flags and 24 of
    26 judged inclusions agree. Nanomotif 1.1.3 table: every decision file identical to 0.11.0 (485 inclusions).
  - Six monoculture training communities, six ZymoOral scenarios and three simulated look-alike communities
    (where the relatives model stays, at 5.0-5.4 standard errors over bins): every benchmark metric identical to
    0.11.0.
  - Known limit: on ZymoFecal the choice rests on a narrow margin (0.9-1.9 standard errors over bins against the
    limit of 2, on five draws of the calibration pieces), and 16 inclusions depend on which calibration pieces are
    drawn. Flags do not.
- The 0.11.0 entry below says that flagged and included contigs were identical on ZymoFecal throughout the speed
  work; those runs made no inclusion at all (the collapse above), so the inclusion path was first checked on a
  metagenome with this version.

### Added (options, off by default)
- `--calibration ...,tempered-bin-prior`: the uniform prior over bins is tempered with the evidence (each bin costs
  log(H) / T instead of log(H)), so the estimated share of unbinned contigs without a bin no longer rises with the
  temperature (ZymoFecal, community-only model: 0.41 at T 1, 0.69 at T 5, 0.94 at T 10 without it). Benchmarks with
  it on: no new false flag or wrong placement, contigs put back 89.3 -> 90.4% on the training communities. On
  ZymoFecal it makes calibration choose the relatives model decisively (gain 1.59 +- 0.15 over bins) and the run then
  includes 1 contig: not a default.
- `--calibration ...,deployed-unknown`: compares the two unknown models at the share each would be deployed with.
  Restores inclusion on ZymoFecal (503) but also replaces the relatives model on the simulated look-alikes (wrongly
  placed contigs 2 -> 19): not a default.

### Documentation (branch docs/review-accuracy)
- README and `docs/`: benchmark figures now say which data they come from. Held-out monoculture communities: 311 of
  316 flags (98.4%) and 1,096 of 1,180 placements (92.9%; validation set 89.6%) right; the development figures
  (every flag right) are labelled as such. The comparison with Nanomotif's refinement is removed.
- Corrected claims without a measurement behind them: "calibrated" probabilities (long contigs remain overconfident),
  methylation "independent of composition and coverage", plasmids and phages methylated "alike" (binned ones can be
  flagged), contaminated bins "handled" (one-motif merges are mostly missed), motifs no bin methylates "harmless",
  a high `p_unknown` as "the right answer", Hi-C precision below 5x (4 of 41, 9.8%, replaces 12.5%), the share of
  contigs without a decision, time and memory per version.
- FAQ: new "Known limitations" section, including the inclusion collapse with some motif tables
  (`p_unknown=1.000`, 0 inclusions) and how to recognise it in the log.

### Benchmarks
- `monoculture_sets.py`: an isolate's piece cuts are seeded by the SHA-256 of its name instead of Python's salted
  `hash()`, so they no longer depend on `PYTHONHASHSEED`. The stored sets cannot be regenerated from code; their cuts
  are kept as data (see the module docstring).
- `benchmark_suite.py`: every row records the version the runs report, the motif table's md5 and the calibration's
  choices (`unknown_lam`, temperature and `p_unknown` ranges), as columns appended to older record files.

**Benchmark effect:** mono-train identical to 0.11.0 (contaminants found 0.948, flag precision 1.000, put back 0.893,
wrongly placed 2.0 per scenario).

## 0.11.0 (2026-11-11): performance and robustness; model options are opt-in

**Results are those of 0.10.2.** A run without new options adds and removes the same contigs. Checked on the six
monoculture training scenarios (`new_contig_bin.tsv`, `decontaminated_contig_bin.tsv` and `bin_contamination.tsv`
byte-identical to 0.10.2; `contig_decisions.tsv` and `include_contigs.tsv` identical except the two label changes
below). The ZymoFecal comparison is recorded in the dev log (`proposals/2026-11-11/release_0.11_integration.md`).

### Performance (outputs unchanged)
- Scoring keeps the marginal tables of binned contigs only and bounds the [contigs x bins] work arrays (synthetic
  60,000 x 300 x 64: peak 13.9 -> 6.2 GB, 35.7 -> 18.6 s); the refinement and posterior tables are built in place;
  calibration holds one model's chunk tables at a time; evidence tables are reduced in blocks (synthetic: peak
  4.95 -> 1.30 GB); unit tables keep the expectation table per worker, take results as they complete, and size
  their batches by the number of motif groups.
- Pileup pass: BGZF ranges are inflated block by block with CRC and length checks (inflate 2.4x faster than
  `gzip.GzipFile`); a contig spanning many blocks is joined once; contig names, modification codes and strand
  are read as text (a contig named with digits, or a block of 4mC rows only, no longer fails).
- `logsumexp` without scipy's array-API layer (299 -> 111 us per call, bit-identical); no process pool for small
  detection fits.
- Optional compiled pileup reader: when the package `epirefine-rs` can be imported, the pileup pass uses its
  reader; `EPIREFINE_RS=0` switches it off. Not a dependency; the Python pass is the reference and the fallback,
  and is used for any site option the reader does not implement. mono_train_1 (5.5 GB pileup, 4 workers, warm
  file cache): pileup pass 16 s -> 3.4 s, largest worker 0.79 -> 0.62 GB, all outputs byte-identical.
- Six training scenarios on 4 cores (site caches reused, shared machine, one run each): 508 s -> 218 s in total, peak memory
  0.58-1.01 GB -> 0.63-0.98 GB.
- Heavy steps of large samples (`perf/heavy-steps`): unit tables from packed integer keys; calibration and final
  scoring run their leave-one-out loops in batches on threads and reuse the marginal-table buffer between fits;
  the per-group tau marginal in one pass; contig evidence scores each contig against its own bin only, in batches;
  two-state share, host profiles and the motif scan of the grouping step on threads; the site cache is read ahead
  during motif grouping; threads and workers follow the CPU affinity of the process. With `epirefine-rs`, the
  model fits and the grouping scan also use compiled kernels (optional as before; `EPIREFINE_RS=0` switches all
  of them off). ZymoFecal (34,878 contigs, 490 bins, 710 motif groups; site cache reused, 12 workers, kernels on,
  idle machine): 4,785 s -> 810 s, main-process peak 31.8 -> 25.5 GB; flagged and included contigs, tau and
  temperature identical. Without the kernels the tables are equal as written on the training scenarios; with them
  a few printed `expected_evidence` values of a large sample differ in the fourth decimal.
- Six training scenarios after this work (4 workers, site caches reused): 15.9 s per scenario on average
  (36.7 s in 0.10.1), peak 0.87 GB.

### Robustness
- Inputs are checked before the assembly is read, with messages that name the problem: a missing, empty or
  wrongly compressed file, an unknown `mod_type`, a contig-bin table or a pileup that matches no contig; contig-bin
  rows naming contigs absent from the assembly are counted in the log; a repeated contig name in the assembly is
  an error.
- The site cache's manifest is deleted before a rebuild starts, so an interrupted rebuild is never reused.
- A run with one bin skips contamination detection with a warning instead of writing NaN.
- `params.json` is strict JSON (an infinite `tau` is `"inf"`) with the version, command line, inputs, seed and
  every option; it is written when the run starts and completed at the end.
- The log starts with the version and the inputs and carries peak memory on every line; errors go to the progress
  stream; a source checkout reports the version of its own `pyproject.toml`; `--help` shows defaults in option
  groups.
- **Label changes (no contig is added or removed differently):** a flagged contig whose move to another bin is
  dropped as redundant keeps status `contaminant` (was `redundant`), and `include_contigs.tsv` marks candidates
  dropped by the redundancy check `dropped_redundant`.
- New report columns and files, none of which changes a decision: `bin_evidence.tsv` gains `robust_fit`,
  `flagged_share`, `two_state_share`, `two_state_group` and `warning` (a warning never withholds a flag);
  `motif_noise.tsv` gains `off_bin_share`; `calibration/false_flag_rate.tsv` and
  `calibration/reliability_deployed.tsv` are new.

### Tests and tooling
- Tests 39 -> 169 (plus 2 expected failures that state what is not supported: contamination detection with one
  bin, and a pileup replaced by one of the same size and modification time): a synthetic community with
  end-to-end, golden-snapshot, invariance (workers, cache state, BGZF, row order), cache and edge-case tests, the
  production site path, the calibration features, the rule options, and the compiled reader against the Python
  pass.
- Evidence tooling: `benchmarks/run_evidence.py` (trivial baselines, harder planted contamination, rule ablations,
  per-contig metrics with fixed denominators).
- Benchmark suite: `--extra-args`, `--workers`, `--label`, and per-scenario `args` in a suite file.

### Model options: opt-in, not validated
Each is off by default and reproduces 0.10.2 when left alone. They were measured on development data only
(monoculture training communities and sets built from them); none has been confirmed on a metagenome or on
held-out data. Numbers and verdicts: dev log, `proposals/2026-11-10/` and
`proposals/2026-11-11/release_0.11_integration.md`.
- Sites: `--shared-sites motif`, `--complement-pairs position | separate`, `--strand-sites auto | split`,
  `--noise-max-off-share`.
- Calibration: `--calibration` with `consistent`, `coherent`, or single features (`temper-then-mix`,
  `prior-averaged`, `prior-matched`, `conservative-temperature`, `full-test-b`, `unit-temperature`,
  `sites-temperature`, `common-tau`, `em-priors`).
- Rules: `--redundancy-order sequential`, `--unknown-prior unbinned` (`--unknown-prior-strength`),
  `--robust-fit damped`, `--max-flagged-share` below 1, `--siteless-bin-guard`, `--regional-leave-out`,
  `--regional-withdraw`, `--regional-tempered`, `--gate-leave-out`, `--gate-unweighted`, `--eps-gated`,
  `--tau-support leave_bin_out`, `--unlisted-genome`, `--inclusion-support`; `--two-state-warning` sets the level
  of the bin warning.

**Benchmark effect:** accuracy unchanged by construction (mono-train: contaminants found 0.948, precision 1, put
back 0.893, as 0.10.2). No benchmark row is recorded with this entry; it is added when the release is made.

## 0.10.2 (2026-11-10): no machine paths in the repository

No change to results.
- `benchmarks/benchmark_suite.py` and `benchmarks/monoculture_sets.py` take the data root from `--data-root` or
  `EPIREFINE_DATA` and the dev log from `--devlog` or `EPIREFINE_DEVLOG`, and stop with a clear message when neither
  is given; no tracked file names a location on the development server any more.
- `benchmarks/suite_monocultures.json` holds paths relative to the data root (`root` key); absolute entries still work.
- `CLAUDE.md`, `.gitignore` and the developer notes describe locations by environment variable.
- Two tests for the suite loader.

**Benchmark effect:** none (no package code changed).

## 0.10.1 (2026-11-02): user documentation

No change to results.
- README rewritten for new users: install, a quick start with the four inputs, what the outputs mean, a
  documentation map, and when to trust the results.
- New user guides in `docs/`: `getting-started.md` (from reads to refined bins, with tested commands),
  `usage.md` (every option grouped, with defaults and when to change them), `outputs.md` (every file and column),
  `how-it-works.md` (the model in plain terms) and `faq.md` (common questions and errors), indexed in
  `docs/README.md`.
- Developer notes moved to `docs/development/` (`method-notes.md`, `scalability.md`); the method notes are marked as
  notes on the predecessor model.
- `Documentation` link in the package metadata.

**Benchmark effect:** none (no package code changed).

## 0.10.0 (2026-11-02): first public release (PyPI, Bioconda)

No change to results.
- Packaging: MIT licence (`LICENSE`), author and project metadata, classifiers for Python 3.10-3.15, README
  usable on PyPI (absolute links; `pip install epirefine`). The source distribution carries the package, tests,
  docs and changelog.
- `.github/workflows/release.yml`: on a version tag, checks the tag against the package version, builds and checks
  the sdist and wheel, tests the installed wheel, publishes to TestPyPI and PyPI with trusted publishing, and
  creates a GitHub release.
- `tests/test_evaluate.py` skips when the repository's `benchmarks/` is not there (as in the sdist).
- `dev/hooks/install.sh` sets `core.hooksPath` instead of a symlink in `.git/hooks`, which failed in worktrees.
- README: a known limitation was out of date (sites shared by motif groups count in the most specific group; they
  are not dropped).

**Benchmark effect:** none (no package code changed).

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
