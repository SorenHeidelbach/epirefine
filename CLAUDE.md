# EpiRefine: rules for working on this project

EpiRefine refines metagenome bins from nanopore methylation. It flags binned contigs that do not belong
to their bin (contamination detection) and places unbinned contigs into bins (contig inclusion). Inputs:
an assembly, a contig-bin table, a modkit pileup and a Nanomotif motif table. Outputs: posteriors per
contig and Nanomotif-compatible files.

This repository holds only the tool, its tests and its monoculture benchmark. The benchmark workflow
(Snakemake, Hi-C validation, the metagenome checks, stress sets) is a separate repository that calls the
tool. The optional compiled kernels are a third repository (`epirefine-rs`, Python module `epirefine_rs`).
Keep workflow and benchmark tooling out of this repository.

## Locations

No tracked file may name a location on a particular machine: no absolute local paths in code, configs,
docs or comments. Locations come from the command line or from the environment:

- `EPIREFINE_DATA`: the benchmark data root.
- `EPIREFINE_DEVLOG`: the development log (outside git).

A machine may keep its own untracked `CLAUDE.md` in a directory above this checkout, with its paths,
its compute limits and the data that must not be touched there. If one exists, read it too. Where it is
stricter than this file, it wins.

## Before changing anything

1. Read the development log: the state note at the top of `README.md`, `MISSION.md` if there is one, the
   newest entries of `DECISIONS.md`, and the newest files under `reviews/` and `proposals/`. Every model
   and design choice there was made for a measured reason. Do not reverse one, or remove what it added,
   without rerunning the benchmark that justified it and telling the user first.
2. Check the branch and the worktree (`git branch --show-current`, `git worktree list`). Never commit
   on `main`.

## Git

- **Branches.** All work happens on a branch, named `<type>/<short-topic>`: `feat/`, `fix/`, `perf/`,
  `model/`, `docs/`, `bench/`, `test/`, `chore/`. One topic per branch.
- **One worktree per branch.** Make a new branch with `git worktree add` and give it its own `.venv`
  (`uv venv && uv pip install -e ".[dev]"`). Never switch branches in a worktree whose environment a
  running job uses: the job would import a mix of two versions.
- **Commits.** Commit every change as you make it, in small commits that each leave the tests passing.
  The message says what changed and why. For a model or speed change it also gives the measured effect.
  End every message with the attribution line from the session's instructions.
- **Integration branches.** The changes of one minor version are collected on `release/<x.y>`. Merge
  each finished topic branch there with `--no-ff`, run the checks listed under "Measuring", and merge
  `release/<x.y>` into `main` once.
- **Merging into `main`.** Only when the user agrees, after they have seen the results. First, on the
  branch: bump `version` in `pyproject.toml`, add the version's section to `CHANGELOG.md` (benchmark
  effect and known limitations), and add the version's row to `benchmarks/results.tsv`. Then
  `git merge --no-ff` with a message that names the version. Merging does not tag.
- **Versions.** MAJOR: breaking change to output files, their columns or the CLI. MINOR: new feature,
  new output, or a model change that changes results. PATCH: bug fix, speed-up with identical
  decisions, docs, tests, tooling. While the major version is 0, breaking changes bump MINOR and say
  "Breaking" in the changelog.
- Never rewrite published history on `main`. Push only when the user asks.

## Releasing

- A `v*` tag publishes. The release workflow builds the package, uploads it to TestPyPI and PyPI and
  creates a GitHub release. This cannot be undone. Tag only when the user says so, only after `main`
  is pushed and the test workflow has passed on it, and only with a tag equal to the version in
  `pyproject.toml`.
- Version 0.11.0 is skipped. Never tag or publish it: with Nanomotif 1.3.0 motifs it makes no
  inclusions on a metagenome. The first published version of that series is 0.11.1.
- The Bioconda recipe follows a PyPI release.

## Data rules

- **Development data** may be used freely to build, tune and test: the monoculture training
  communities (`mono_train_1..6`), mock communities, simulated data, and the metagenomes the machine's
  own `CLAUDE.md` lists as development samples. Label every number from them "development" wherever it
  is reported.
- **Spent data.** The monoculture test and validation communities (`benchmark_suite.py --all`) were
  looked at during development. They are no longer an unseen evaluation. Label them "spent".
- **Held-out data** are never opened, run or inspected during development, by any script or agent:
  the ONT NO-MISS isolate set, the Dabernig 2025 monocultures (PRJNA1091452), and the samples the
  machine's own `CLAUDE.md` lists as untouched. The final evaluation on them is run once, on a frozen
  released version, under an analysis plan written beforehand, and only when the user says so. Never
  name a held-out number as a target to move.
- **Truth.** In the monoculture benchmarks the truth of a contig is the isolate assembly it comes
  from. Never relabel contigs that look foreign. A flag on one counts as a false flag.
- **Motif input.** Motif tables come from Nanomotif 1.3.0; older tables are for comparison only. The
  motif table can change the results a lot, so record its checksum with every run.
- **What the metrics leave out.** Contigs below the coverage cut-off (status `low_coverage`) and
  contigs without methylation evidence are reported separately and left out of recall and precision.
  An inclusion into a bin that holds a single contig is valid (plasmid, phage).

## Defaults and options

- Nothing is on by default without a measured benefit. A change with no measurable effect stays an
  option or is dropped.
- A model change starts as an option on its branch. It becomes the default on the integration branch
  only after it passes the checks under "Measuring", including the metagenome-scale ones.
- A default must not sit on a knife edge. If resampling the calibration data (another seed, a
  bootstrap) flips a model choice, the choice is not stable enough to ship.
- Ask the user before a change to outputs, defaults or the CLI reaches `main`.

## Measuring the effect of every change

- Say what result counts as a pass before running a benchmark, and do not move it afterwards.
- `benchmarks/benchmark_suite.py --note "<what changed>"` (needs `EPIREFINE_DATA` and
  `EPIREFINE_DEVLOG`, or `--data-root` and `--devlog`) runs the monoculture training communities in a
  few minutes. It records accuracy (contaminants found, flag precision, contigs put back, contigs
  wrongly placed, bin completeness and contamination), wall time and peak memory, with the version,
  the commit and the motif-table checksum. Run it before merging any change to the model, the pipeline
  or performance. `--suite <file>` runs another suite file, such as the workflow repository's mock and
  simulated scenarios.
- The monoculture benchmark cannot see failures that appear only at metagenome scale: in 0.11.0 the
  inclusions on a real metagenome fell from about 500 to 0 while every monoculture metric stayed the
  same. For every change to the model or the calibration, also run the workflow repository's
  metagenome checks: the sentinel, and the planted-truth set when the change is meant to improve
  results.
- A replay of stored calibration data is a cheap first test, not proof. A replay that holds one model
  choice fixed can predict the opposite of the full run.
- **Speed and memory changes** must leave the decisions, tau and the temperature identical, and the
  outputs byte-identical where possible. Report peak memory and never raise it. Iterate on the smallest
  input that shows the effect, one measurement per step; run the full-size benchmark at milestones.
- Report the effect of every implemented change as numbers: what got faster or slower and by how
  much, which metric moved, on which benchmark. Say so plainly when nothing measurable changed, or
  when something got worse.
- Every suite run appends its summary to `benchmarks/results.tsv` (commit the row of each version that
  reaches `main`) and to the dev log (`benchmarks/history.tsv`).

## Tests

- `.venv/bin/python -m pytest -q tests` must pass before every commit. The pre-commit hook in
  `dev/hooks/pre-commit` enforces this and refuses commits on `main`; install it with
  `dev/hooks/install.sh`.
- Every bug fix comes with a test that fails without the fix. Tests marked as strict expected failures
  record known defects; remove the mark in the commit that fixes the defect.
- Continuous integration runs the tests on Python 3.10 to 3.15, and once with the lowest allowed
  dependency versions. Keep the lower bounds in `pyproject.toml` true.

## Code

- Python ≥ 3.10, numpy, scipy, polars. New dependencies only after asking the user.
- Compiled code lives only in the optional `epirefine_rs` extension. Every kernel has a numpy
  fallback (`EPIREFINE_RS=0` forces it), and both paths must give the same decisions.
- Scale: the target is ~50k contigs, ~1000 motifs, hundreds of bins, pileups of 100+ GB. Nothing may
  hold all motif records or a [units × bins × groups] tensor in memory; stream and batch instead
  (see `docs/development/scalability.md`).
- Reproducibility: a fixed `--seed` must give identical outputs from run to run, and identical
  decisions for any number of workers and with or without the compiled kernels. Do not depend on
  polars `group_by` order (sort, or use `maintain_order=True`), on the iteration order of sets, or on
  the thread count. Give each random step its own generator.
- Keep the Nanomotif-compatible outputs (`bin_contamination.tsv`, `include_contigs.tsv`,
  `new_contig_bin.tsv`, `decontaminated_contig_bin.tsv`) column-compatible with Nanomotif.
- Do not add tooling, plotting or other data types (Hi-C, coverage, reads) to `src/epirefine` without
  asking the user.
- Match the existing style: docstrings that explain the statistics, comments that say why.

## Words

- "call" and "no call" for whether EpiRefine decided on a contig; "recall" and "precision" for how
  well; rates in percent. One term per concept.
- Label every reported number development, spent or held-out.
- The docs (`docs/`) stay in step with the CLI and the outputs, and claim nothing the benchmarks do
  not support.

## Layout

- `src/epirefine/`: the package, bin refinement only. `cli.py` is the command line; `pipeline.py` runs
  everything; `sites.py` streams the pileup into a site cache; `motifs.py` does motif matching and
  grouping; `detection.py` fits the modification-caller model; `model.py` holds the likelihood model
  and `grid.py` its methylation grid; `calibrate.py` the calibration on pieces cut from the bins;
  `assignability.py` the methylation-evidence check; `refine.py` the contamination and inclusion
  decisions; `redundancy.py` the k-mer check against inclusions the bin already holds; `io.py` and
  `runinfo.py` read inputs and record the run.
- `tests/`: unit tests, with a golden snapshot of the decisions on a small synthetic community.
- `benchmarks/`: the monoculture benchmark. `benchmark_suite.py`, `suite_monocultures.json` (paths
  relative to `$EPIREFINE_DATA/benchmarks/mono_bench`), the scorer `evaluate_refinement.py`, the set
  builders (`monoculture_sets.py`, `fragment_assembly.py`, `perturb_bins.py`, `hard_scenarios.py`),
  the comparison with simple baselines (`run_evidence.py`, `evidence_metrics.py`,
  `trivial_baselines.py`) and `results.tsv` (summary per version).
- `dev/`: git hooks and speed prototypes.
- `docs/`: user documentation (`getting-started.md`, `usage.md`, `outputs.md`, `how-it-works.md`,
  `faq.md`). `docs/development/`: developer notes on the model (`method-notes.md`) and on performance
  (`scalability.md`).

## Development log (outside git, `$EPIREFINE_DEVLOG`)

The log holds the reasoning behind the code, for future sessions.

- `README.md`: current state, how the pieces fit, open questions. Keep the state note at its top
  current.
- `MISSION.md`: when a mission is running, its goal, its acceptance gates, the ordered work packages,
  the questions waiting for the user and a ledger of what is done. Update it after every finished step.
- `DECISIONS.md`: one entry per decision (date, version, choice, why, rejected alternatives, measured
  effect). Add an entry for every choice a later change could undo, and for every ruling the user
  gives, on the day it is given.
- `BACKLOG.md`, `TEST_PLAN.md`: method ideas and the stress-test programme.
- `proposals/<date>/`: design notes and results of investigations. `reviews/<date>/`: audits. Mark a
  proposal "superseded" at its top when a later decision overrides it.
- `sessions/`: one summary per working session (`YYYY-MM-DD_<topic>.md`): what was asked, what was
  tried, what worked and what did not, with numbers. Write it before the session ends.
- `DATA.md`: where the datasets live and how the benchmark data were made.
- `benchmarks/`: results per version (written by the benchmark suite).
- `packaging/`: the release checklist and the Bioconda recipe.
