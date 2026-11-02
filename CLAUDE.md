# EpiRefine: rules for working on this project

EpiRefine refines metagenome bins from nanopore methylation: it flags binned contigs that don't belong
to their bin (contamination detection) and places unbinned contigs into bins (contig inclusion), with
calibrated posteriors and Nanomotif-compatible outputs. It builds on Nanomotif's motif output and a
modkit pileup. The package does only that; the workflow, Hi-C validation and benchmarks live in
their own folders and call it.

## Before changing anything

1. Read the development log at `/home/ubuntu/plasmid_binner_devlog/`: `README.md` (current state and
   open questions), then `DECISIONS.md`. Every model and design choice there was made for a measured
   reason. Don't reverse one, or remove what it added, without a benchmark result showing the
   reversal is better, and say so to the user first.
2. Check the branch (`git branch --show-current`). Never commit on `main`.

## Git workflow

- **Branches.** All work happens on a branch off `main`, named `<type>/<short-topic>`: `feat/`, `fix/`,
  `perf/`, `model/`, `docs/`, `bench/`, `chore/`. One topic per branch.
- **Commits.** Commit every change as you make it, in small commits that each leave the tests passing.
  The message says what changed and why; for a model or speed change it also gives the measured
  effect. End every message with the attribution line from the session's instructions.
- **Merging.** A branch is merged into `main` when its topic is done, with a version bump:
  1. on the branch, bump `version` in `pyproject.toml` (semantic versioning, below) and add the
     version's section to `CHANGELOG.md`, including the benchmark effect;
  2. `git checkout main && git merge --no-ff <branch>` with a message naming the version;
  3. `git tag -a v<version> -m "epirefine <version>"`; delete the merged branch.
  Ask the user before merging a change that alters outputs, defaults or the CLI.
- **Versions.** MAJOR: breaking change to output files, their columns or the CLI. MINOR: new feature,
  new output, or a model change that changes results. PATCH: bug fix, speed-up with identical
  results, docs, tests, tooling. While the major version is 0, breaking changes bump MINOR and say
  "Breaking" in the changelog.
- Never rewrite published history on `main` (no force-push, no amending merged commits).

## Tests

- `.venv/bin/python -m pytest -q tests` must pass before every commit. The pre-commit hook in
  `dev/hooks/pre-commit` enforces this; install it with `dev/hooks/install.sh` if it's missing.
- Every bug fix comes with a test that fails without the fix.
- A speed-up must not change results: compare outputs (or the benchmark metrics) before and after.

## Measuring and reporting the effect of every change

- Run `benchmarks/benchmark_suite.py --note "<what changed>"` before merging any change to the model,
  the pipeline or performance. It takes about 15 minutes, records accuracy (contaminants found,
  flag precision, contigs put back, contigs of unbinned genomes wrongly placed, bin completeness
  and contamination), wall time and peak memory, and prints the change against the last recorded
  version. Add `--all` (adds human_gut, about 1 hour) when a change affects speed or memory at scale.
- Report the effect to the user after every implemented change, as numbers: what got faster or
  slower and by how much, which metric moved, on which benchmark. Say so plainly when nothing
  measurable changed, or when something got worse.
- The benchmark history lives in the dev log (`benchmarks/history.tsv`), not in the repo.

## Development log (outside git)

`/home/ubuntu/plasmid_binner_devlog/` holds the reasoning behind the code, for future sessions:

- `README.md`: current state, how the pieces fit, open questions. Keep it current.
- `DECISIONS.md`: one entry per decision (date, version, choice, why, rejected alternatives,
  measured effect). Add an entry whenever a choice is made that a later change could undo.
- `sessions/`: one summary per working session (`YYYY-MM-DD_<topic>.md`): what was asked, what was
  tried, what worked and what didn't, with numbers. Write it before the session ends.
- `DATA.md`: where the datasets live and how the benchmark data were generated.
- `benchmarks/`: results per version (written by the benchmark suite).

## Code

- Python ≥ 3.10, numpy, scipy, polars. New dependencies only after asking the user.
- Scale: the target is ~50k contigs, ~1000 motifs, hundreds of bins, pileups of 100+ GB. Nothing may
  hold all motif records or a [units × bins × groups] tensor in memory; stream and batch instead
  (see `docs/scalability.md`).
- Runs must be reproducible: a fixed `--seed` gives identical outputs. Don't depend on polars
  `group_by` order; sort or use `maintain_order=True`.
- Keep the Nanomotif-compatible outputs (`bin_contamination.tsv`, `include_contigs.tsv`,
  `new_contig_bin.tsv`, `decontaminated_contig_bin.tsv`) column-compatible with Nanomotif.
- Match the existing style: docstrings that explain the statistics, comments that say why.

## Layout

- `src/epirefine/`: the package, bin refinement only. `pipeline.py` runs everything; `sites.py`
  streams the pileup into a site cache; `model.py` holds the likelihood model; `calibrate.py` the
  pseudo-contig calibration; `refine.py` the contamination and inclusion decisions; `redundancy.py`
  the k-mer check against inclusions the bin already holds; `motifs.py` motif matching and grouping.
  Don't add tooling, plotting or other data types (Hi-C, coverage) here without asking the user.
- `tests/`: unit tests.
- `workflow/`: the Snakemake workflow (EpiRefine, Nanomotif, CheckM2, Hi-C validation) and its
  scripts (`write_bins.py`, `compare_bins.R`) and conda envs.
- `hic/`: Hi-C validation (`contacts.py`, `validate.py`, `env.yaml`), run from the workflow. It
  judges EpiRefine from outside; keep it independent of the package.
- `benchmarks/`: `benchmark_suite.py`, `suite.json` (absolute paths on this server) and the scripts
  that build scenarios (`fragment_assembly.py`, `perturb_bins.py`, `simulate_community.py`,
  `make_truth.py`, `evaluate_refinement.py`, `sweep_thresholds.py`, `compare_nanomotif.py`).
- `dev/`: git hooks and speed prototypes.
- `docs/method.md` (the model), `docs/scalability.md` (performance work).
- Environment: `.venv` (`uv venv && uv pip install -e ".[dev]"`); run with `.venv/bin/epirefine`.
