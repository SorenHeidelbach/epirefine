# EpiRefine

[![tests](https://github.com/SorenHeidelbach/epirefine/actions/workflows/tests.yml/badge.svg)](https://github.com/SorenHeidelbach/epirefine/actions/workflows/tests.yml)

*Epi*genetics + *refine*: it refines metagenome bins using the methylation every genome carries. Formerly `pbinner`.

Bin refinement from nanopore methylation, built on Nanomotif output:

- **contamination detection:** flags binned contigs whose methylation doesn't fit their bin;
- **contig inclusion:** places unbinned contigs into the bin they belong to.

Every decision comes with a calibrated posterior, including an explicit "unbinned genome" class, so
contigs of genomes that have no bin stay out of bins. Outputs include Nanomotif-compatible files
(`bin_contamination.tsv`, `include_contigs.tsv`, `new_contig_bin.tsv`,
`decontaminated_contig_bin.tsv`). Calibration is estimated inside each sample from pseudo-contigs
cut from binned contigs.

- Method: [docs/method.md](https://github.com/SorenHeidelbach/epirefine/blob/main/docs/method.md).
- Performance work: [docs/scalability.md](https://github.com/SorenHeidelbach/epirefine/blob/main/docs/scalability.md).
- Changes per version: [CHANGELOG.md](https://github.com/SorenHeidelbach/epirefine/blob/main/CHANGELOG.md).
- Development rules: [CLAUDE.md](https://github.com/SorenHeidelbach/epirefine/blob/main/CLAUDE.md).

## Repository layout

| Folder | Content |
|---|---|
| `src/epirefine/` | The package: methylation-based contamination detection and contig inclusion, nothing else |
| `tests/` | Unit tests (`.venv/bin/python -m pytest -q tests`) |
| `benchmarks/` | Monoculture benchmark: suite runner, the scripts that build the sets, the evaluator; results of every version in [`benchmarks/results.tsv`](https://github.com/SorenHeidelbach/epirefine/blob/main/benchmarks/results.tsv) |
| `dev/` | Development tooling: git hooks, speed prototypes |
| `docs/` | Method and performance notes |

## Install

```bash
pip install epirefine                 # from PyPI
conda install -c bioconda epirefine   # from Bioconda (once the recipe is accepted)
```

From source, for development:

```bash
git clone https://github.com/SorenHeidelbach/epirefine.git && cd epirefine
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q
```

Python 3.10-3.15 (tested on every push, together with the lowest dependency versions allowed:
numpy 1.26, scipy 1.11, polars 1.2).

## Running EpiRefine

| Flag | File | Notes |
|---|---|---|
| `--assembly` | assembly FASTA | |
| `--pileup` | `modkit pileup` bedMethyl (`.bed`/`.bed.gz`) | Space-separated (older modkit) or all-tab formats. Only rows at motif sites are kept while streaming. |
| `--contig-bin` | `contig<TAB>bin`, no header | The bins to refine |
| `--motifs` | Nanomotif `bin-motifs.tsv` | Only the **union** of motifs is used (+ `motif_complement`). Which bin carries which motif is re-estimated from the data. |

```bash
epirefine run --assembly asm.fasta --pileup pileup.bed --contig-bin contig_bin.tsv \
    --motifs nanomotif/bin-motifs.tsv --out epirefine_out --workers 16
```

**Decision thresholds:**
- `--contamination-threshold` (0.5): flag a binned contig when P(foreign) is at least this,
- `--min-methylome-difference` (0.3): ... and its 95% credible interval for some motif group's
  methylated fraction lies at least this far from its bin's level (an effect size, so that many
  weak site-level differences don't add up to a flag);
- `--include-threshold` (0.95): include an unbinned contig when one bin's posterior is at least this;
- `--max-redundancy` (0.3): don't include a contig when this share of its k-mers is already in the
  bin (assembly duplicates; status `redundant`).

**Model options:** `--variation beta|logit` (how a contig's methylation varies around its bin's;
`--tau` overrides the calibrated value), `--temperature`, `--allow-sharpening` (default: the
temperature is clipped at ≥ 1), `--p-unknown`, `--contam-eps`, `--chunk-size`.
`--regional-inclusion` (experimental, off by default) also includes long unbinned contigs whose
50 kb windows match a bin when the whole contig doesn't.

**Running:**
- `--workers N` sets the processes for the pileup and table passes (default: CPUs, at most 16).
- `--cache-dir` holds the site cache (default `<out>/cache`). A rerun with the same assembly, pileup
  and motifs reuses it and skips the pileup pass.
- `--frag-budget` (default 6000) is the total number of calibration fragments, spread over bins and
  lengths with at most `--frags-per-bin` each.
- `--motif-fit` also writes each contig's methylated fraction per motif group next to its bin's.

The pileup must have each contig's rows together (as modkit writes it). Uncompressed pileups are read
in parallel byte ranges; gzipped ones by a single worker.

## Outputs

| File | Content |
|---|---|
| `contig_decisions.tsv` | Per contig: `status`, given and new bin, `p_contaminant`, `methylome_difference`, `redundancy_with_bin`, best alternative bin and its posterior, MAP, `p_unknown`, 95% credible set |
| `bin_contamination.tsv`, `include_contigs.tsv`, `new_contig_bin.tsv`, `decontaminated_contig_bin.tsv` | Nanomotif-compatible results |
| `host_profiles.tsv` | Per bin × motif group: methylated fraction (posterior mean, 95% interval) |
| `host_chunk_outliers.tsv` | P(chunk is foreign to its bin) for every 50 kb chunk of a binned contig |
| `detection_model.tsv` | Learned caller behaviour per motif group (μ0, μ1, κ0, κ1) |
| `motif_groups.tsv` | Which motifs were merged into which feature, with site and read counts |
| `motif_fit.tsv` | With `--motif-fit`: per contig × motif group, observed methylated fraction (95% CI) and the bin's |
| `calibration/` | Pseudo-contig calibration: accuracy, ECE, credible-set coverage per length, temperature and τ fits |
| `params.json` | Fitted parameters and settings |

**Status** in `contig_decisions.tsv`:

| Status | Meaning |
|---|---|
| `kept` | Binned contig that fits its bin |
| `contaminant` | Binned contig flagged as foreign; removed from its bin |
| `moved` | Flagged contaminant that confidently belongs to another bin; moved there |
| `sole_contig_of_bin` | Only contig of its bin: no leave-one-out profile, not judged |
| `included` | Unbinned contig placed into a bin |
| `redundant` | Would be included, but its sequence is already in the bin |
| `unbinned` | Unbinned contig left out (uncertain, or its genome has no bin) |

## Workflow and evaluation

The Snakemake workflow (Nanomotif, EpiRefine, CheckM2), the Hi-C validation and the benchmarks beyond the
monocultures (ZymoOral pieces, simulated look-alikes, chimeras) live in a separate repository that calls this
tool (`epirefine_workflow`; moved out in 0.9.1).

## Benchmarks

`benchmarks/benchmark_suite.py --note "<change>"` runs the monoculture benchmark (`benchmarks/suite_monocultures.json`):
communities built from isolates sequenced alone, so the origin of every contig is known, with planted errors
(contaminants, removed members, genomes without a bin). By default it runs the six training communities
(about 5 minutes); `--all` adds the held-out test and validation communities, for final evaluation only. Every
run appends one row per benchmark group and version to [`benchmarks/results.tsv`](https://github.com/SorenHeidelbach/epirefine/blob/main/benchmarks/results.tsv):
contaminants found, flag precision, contigs put back, contigs of bin-less genomes placed, bin completeness and
contamination after refinement, wall time and peak memory. `benchmarks/monoculture_sets.py` rebuilds the sets.

## Known limitations

- **One bin per contig.** A contig assembled across two genomes is not split.
- **Robust profiles assume a contaminant looks like a "random community genome".** A bin that is a
  50/50 merge of two genomes is not resolved.
- **Motif groups are treated as independent.** A site matched by several groups counts in the most specific
  one, so one methyltransferase can show up in several partly methylated groups and count more than once.
- **No coverage, co-abundance or taxonomy evidence**: methylation only.
- **Dropped constant:** the binomial coefficient is omitted from likelihoods, so reported
  log-likelihoods are relative. Bayes factors and posteriors are unaffected.
